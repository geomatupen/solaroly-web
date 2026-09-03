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

  function hierarchySource(workflow, geojsonFiles = []) {
    const path = workflow.outputs?.regularized?.path || "";
    if (!path) return "";
    const catalogEntry = geojsonFiles.find(file => file.path === path);
    if (catalogEntry) return catalogEntry.stage === "regularized" ? path : "";
    return path.split("/").pop() === "regularized.geojson" ? path : "";
  }

  function refresh(context = api()?.getContext()) {
    if (!context) return;
    const select = byId("ppHierarchySource");
    const previous = select.value;
    const workflows = context.workflows.filter(workflow =>
      workflow.workflow_kind !== "anomaly" && hierarchySource(workflow, context.geojsonFiles)
    );
    select.replaceChildren();
    addOption(select, "", "Select a regularized output…");
    workflows.forEach((workflow, index) => {
      const created = workflow.created_at
        ? new Date(workflow.created_at).toLocaleString()
        : workflow.id;
      const latest = index === 0 ? " · Latest" : "";
      addOption(
        select,
        hierarchySource(workflow, context.geojsonFiles),
        `Regularized · ${created}${latest}`,
        workflow.id,
      );
    });
    if ([...select.options].some(option => option.value === previous)) select.value = previous;
    else if (select.options.length > 1) select.selectedIndex = 1;
    select.disabled = workflows.length === 0;
    const selectedWorkflowId = select.selectedOptions[0]?.dataset.workflowId;
    const selectedWorkflow = workflows.find(workflow => workflow.id === selectedWorkflowId);
    byId("ppBuildHierarchy").disabled = !select.value
      || ["queued", "running"].includes(selectedWorkflow?.status);
  }

  async function buildHierarchy() {
    const workspace = api();
    const context = workspace.getContext();
    const select = byId("ppHierarchySource");
    const option = select.selectedOptions[0];
    const workflowId = option?.dataset.workflowId;
    if (!context.resultId || !workflowId || !select.value) return;
    let workflow = context.workflows.find(item => item.id === workflowId);
    workspace.setMessage("Checking the existing Rows GeoJSON before replacement…");
    try {
      workflow = await workspace.requestJson(
        `/api/results/${encodeURIComponent(context.resultId)}/postprocess/${encodeURIComponent(workflowId)}`,
        { cache: "no-store" },
      );
    } catch (error) {
      workspace.setMessage(`Could not check the existing Rows output: ${error.message}`, "err");
      return;
    }
    const hasExistingHierarchy = Boolean(workflow?.outputs?.solar_rows);
    if (hasExistingHierarchy) {
      const confirmed = await workspace.confirmReplacement(
        "Replace rows and assigned IDs?",
        "The existing Rows GeoJSON will be replaced, and panel/row IDs in the selected Regularized GeoJSON will be assigned again. Manual changes in those files may be lost.",
      );
      if (!confirmed) return;
    }
    const inputPath = select.value;
    byId("ppBuildHierarchy").disabled = true;
    workspace.setMessage("Starting row generation and ID assignment…");
    try {
      workspace.selectWorkflow(workflowId);
      const payload = await workspace.requestJson(
        `/api/results/${encodeURIComponent(context.resultId)}/postprocess/${encodeURIComponent(workflowId)}/hierarchy`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            input_path: inputPath,
            max_orientation_difference_deg: Number(byId("ppRowAngle").value),
            max_lateral_distance_factor: Number(byId("ppRowLateral").value),
            max_along_gap_factor: Number(byId("ppRowGap").value),
            max_inner_row_gap_factor: Number(byId("ppInnerRowGap").value),
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
