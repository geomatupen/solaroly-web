(function initializeRowAlignmentModule(){
  "use strict";

  const state = {
    sources: [],
    postprocessJobCount: 0,
    loaded: false,
    loading: false,
    inputType: null,
  };
  const byId = id => document.getElementById(id);

  function setStatus(message, tone = ""){
    const element = byId("rowAlignmentSourceStatus");
    if(!element) return;
    element.textContent = message || "";
    element.className = `statusLine tiny${tone ? ` ${tone}` : ""}`;
  }

  function selectedSource(){
    const jobId = byId("selRowAlignmentJob")?.value || "";
    const path = byId("selRowAlignmentGeoJson")?.value || "";
    return state.sources.find(item => item.job_id === jobId && item.path === path) || null;
  }

  function renderRows(){
    const jobId = byId("selRowAlignmentJob")?.value || "";
    const select = byId("selRowAlignmentGeoJson");
    if(!select) return;
    const matching = state.sources.filter(item => item.job_id === jobId);
    select.innerHTML = '<option value="">Select a Rows GeoJSON…</option>';
    matching.forEach(item => {
      const option = document.createElement("option");
      option.value = item.path;
      option.textContent = `${item.name} · ${Number(item.feature_count || 0).toLocaleString()} rows`;
      select.appendChild(option);
    });
    select.disabled = !jobId || !matching.length;
    if(matching.length === 1){
      select.value = matching[0].path;
      setStatus(`${matching[0].feature_count.toLocaleString()} mapped rows available.`, "ok");
    }else if(jobId && !matching.length){
      setStatus("This job does not have a completed Rows GeoJSON.", "warn");
    }else{
      setStatus("");
    }
  }

  function renderJobs(){
    const select = byId("selRowAlignmentJob");
    if(!select) return;
    const jobs = new Map();
    state.sources.forEach(item => jobs.set(item.job_id, item.job_name));
    select.innerHTML = '<option value="">Select a post-processing job…</option>';
    [...jobs.entries()].sort((a, b) => a[1].localeCompare(b[1])).forEach(([id, name]) => {
      const option = document.createElement("option");
      option.value = id;
      option.textContent = name;
      select.appendChild(option);
    });
    select.disabled = jobs.size === 0;
    const rowsSelect = byId("selRowAlignmentGeoJson");
    if(rowsSelect && jobs.size === 0) rowsSelect.disabled = true;
    if(jobs.size === 1){
      select.value = [...jobs.keys()][0];
      renderRows();
    }
  }

  async function loadSources(force = false){
    if(state.loading || (state.loaded && !force)) return;
    state.loading = true;
    setStatus("Loading available row layers…");
    try{
      const response = await fetch("/api/row-alignment/sources", { cache: "no-store" });
      const payload = await response.json().catch(() => ({}));
      if(!response.ok || !payload.ok) throw new Error(payload.detail || "Could not load row layers.");
      state.sources = Array.isArray(payload.sources) ? payload.sources : [];
      state.postprocessJobCount = Number(payload.postprocess_job_count || 0);
      state.loaded = true;
      renderJobs();
      if(!state.sources.length){
        const message = state.postprocessJobCount === 0
          ? "No post-processing job is available. First run solar-panel segmentation, create a Post-process job, combine and regularize the panels, then run Assign panel and row IDs."
          : "No completed Rows GeoJSON is available. Open a Post-process job, combine and regularize the segmentation panels, then run Assign panel and row IDs.";
        setStatus(message, "warn");
      }else if(!byId("selRowAlignmentJob")?.value){
        setStatus(`${state.sources.length} row layer${state.sources.length === 1 ? "" : "s"} available.`);
      }
    }catch(error){
      setStatus(error.message || String(error), "err");
    }finally{
      state.loading = false;
    }
  }

  function setInfoOpen(open){
    byId("rowAlignmentInfoModal")?.classList.toggle("hidden", !open);
  }

  function syncAvailability(inputType){
    if(arguments.length) state.inputType = inputType || null;
    const checkbox = byId("chkAlignImagesToRows");
    const options = byId("rowAlignmentOptions");
    const hint = byId("rowAlignmentHint");
    if(!checkbox) return;

    const orthophotoSelected = state.inputType === "tif";
    const mosaicSelected = byId("chkMosaicImages")?.checked === true;
    const unavailable = orthophotoSelected || mosaicSelected;
    checkbox.disabled = unavailable;

    if(unavailable){
      checkbox.checked = false;
      if(options) options.hidden = true;
      const reason = orthophotoSelected
        ? "Image-to-row alignment is unavailable for orthophoto input."
        : "Image-to-row alignment is unavailable while approximate mosaic creation is enabled.";
      checkbox.title = reason;
      if(hint) hint.textContent = reason;
      return;
    }

    checkbox.title = "";
    if(hint) hint.textContent = "Refine prepared thermal-image positions and orientation before inference.";
  }

  function validate(){
    if(!byId("chkAlignImagesToRows")?.checked) return null;
    if(byId("chkMosaicImages")?.checked){
      return "Disable approximate mosaic creation before aligning individual images to solar rows.";
    }
    if(!selectedSource()) return "Select a post-processing job and Rows GeoJSON for image alignment.";
    const maximumPosition = Number(byId("rowAlignmentMaxPosition")?.value);
    const maximumRotation = Number(byId("rowAlignmentMaxRotation")?.value);
    if(!Number.isFinite(maximumPosition) || maximumPosition < 0.5 || maximumPosition > 50){
      return "Maximum position correction must be between 0.5 and 50 metres.";
    }
    if(!Number.isFinite(maximumRotation) || maximumRotation < 0 || maximumRotation > 45){
      return "Maximum orientation correction must be between 0 and 45 degrees.";
    }
    return null;
  }

  function appendFormData(formData){
    const checkbox = byId("chkAlignImagesToRows");
    const enabled = checkbox?.checked === true && checkbox.disabled !== true;
    formData.append("align_images_to_rows", enabled ? "true" : "false");
    if(!enabled) return;
    const source = selectedSource();
    formData.append("row_alignment_job_id", source.job_id);
    formData.append("row_alignment_path", source.path);
    formData.append("row_alignment_max_position_m", byId("rowAlignmentMaxPosition").value);
    formData.append("row_alignment_max_rotation_deg", byId("rowAlignmentMaxRotation").value);
  }

  function init(){
    const checkbox = byId("chkAlignImagesToRows");
    const options = byId("rowAlignmentOptions");
    checkbox?.addEventListener("change", () => {
      if(options) options.hidden = !checkbox.checked;
      if(checkbox.checked) loadSources();
    });
    byId("chkMosaicImages")?.addEventListener("change", () => syncAvailability());
    byId("selRowAlignmentJob")?.addEventListener("change", renderRows);
    byId("selRowAlignmentGeoJson")?.addEventListener("change", () => {
      const source = selectedSource();
      setStatus(source ? `${source.feature_count.toLocaleString()} mapped rows selected.` : "");
    });
    byId("btnRowAlignmentInfo")?.addEventListener("click", () => setInfoOpen(true));
    byId("btnCloseRowAlignmentInfo")?.addEventListener("click", () => setInfoOpen(false));
    byId("btnCloseRowAlignmentInfoFooter")?.addEventListener("click", () => setInfoOpen(false));
    byId("rowAlignmentInfoModal")?.addEventListener("click", event => {
      if(event.target === byId("rowAlignmentInfoModal")) setInfoOpen(false);
    });
    syncAvailability();
  }

  window.TestRowAlignment = { init, validate, appendFormData, loadSources, syncAvailability };
  if(document.readyState === "loading") document.addEventListener("DOMContentLoaded", init, { once: true });
  else init();
})();
