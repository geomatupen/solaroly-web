(() => {
  "use strict";

  const byId = id => document.getElementById(id);
  const api = () => window.PostprocessWorkspace;

  function addOption(select, value, label, workflowId = "") {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    if (workflowId) option.dataset.workflowId = workflowId;
    select.appendChild(option);
  }

  function hierarchySource(workflow) {
    const latestRevision = (workflow.manual_revisions || []).at(-1);
    if (latestRevision?.source_stage === "regularized" && workflow.outputs?.edited?.path) {
      return workflow.outputs.edited.path;
    }
    return workflow.outputs?.regularized?.path || "";
  }

  function refresh(context = api()?.getContext()) {
    if (!context) return;
    const select = byId("ppHierarchySource");
    const previous = select.value;
    const workflows = context.workflows.filter(workflow =>
      workflow.workflow_kind !== "anomaly" && hierarchySource(workflow)
    );
    select.replaceChildren();
    addOption(select, "", "Select a regularized output…");
    workflows.forEach((workflow, index) => {
      const suffix = index === 0 ? " · Latest" : "";
      addOption(
        select,
        hierarchySource(workflow),
        `${workflow.display_name || workflow.id}${suffix}`,
        workflow.id,
      );
    });
    if ([...select.options].some(option => option.value === previous)) select.value = previous;
    else if (select.options.length > 1) select.selectedIndex = 1;
    select.disabled = workflows.length === 0;
    byId("ppBuildHierarchy").disabled = !select.value;
    byId("ppHierarchyStep").hidden = workflows.length === 0;
  }

  async function buildHierarchy() {
    const workspace = api();
    const context = workspace.getContext();
    const select = byId("ppHierarchySource");
    const option = select.selectedOptions[0];
    const workflowId = option?.dataset.workflowId;
    if (!context.resultId || !workflowId || !select.value) return;
    byId("ppBuildHierarchy").disabled = true;
    workspace.setMessage("Starting panel-row hierarchy generation…");
    try {
      workspace.selectWorkflow(workflowId);
      const payload = await workspace.requestJson(
        `/api/results/${encodeURIComponent(context.resultId)}/postprocess/${encodeURIComponent(workflowId)}/hierarchy`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            input_path: select.value,
            max_orientation_difference_deg: Number(byId("ppRowAngle").value),
            max_lateral_distance_factor: Number(byId("ppRowLateral").value),
            max_along_gap_factor: Number(byId("ppRowGap").value),
          }),
        },
      );
      await workspace.runWorkflow(payload);
    } catch (error) {
      workspace.setMessage(error.message, "err");
      byId("ppBuildHierarchy").disabled = false;
    }
  }

  function init() {
    byId("ppBuildHierarchy")?.addEventListener("click", buildHierarchy);
    byId("ppHierarchySource")?.addEventListener("change", event => {
      byId("ppBuildHierarchy").disabled = !event.target.value;
      const workflowId = event.target.selectedOptions[0]?.dataset.workflowId;
      if (workflowId) api()?.selectWorkflow(workflowId);
    });
    document.addEventListener("postprocess:data", event => refresh(event.detail));
    document.addEventListener("postprocess:workflow", event => refresh(event.detail.context));
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
