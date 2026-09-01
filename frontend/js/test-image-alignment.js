(function initializeImageAlignmentModule(){
  "use strict";

  const byId = id => document.getElementById(id);
  let inputType = null;

  function setInfoOpen(open){
    byId("imageAlignmentInfoModal")?.classList.toggle("hidden", !open);
  }

  function syncAvailability(nextInputType){
    if(arguments.length) inputType = nextInputType || null;
    const checkbox = byId("chkAlignImages");
    const options = byId("imageAlignmentOptions");
    const hint = byId("imageAlignmentHint");
    if(!checkbox) return;

    const orthophotoSelected = inputType === "tif";
    const unavailable = orthophotoSelected;
    checkbox.disabled = unavailable;

    if(unavailable){
      checkbox.checked = false;
      if(options) options.hidden = true;
      const reason = "Individual-image alignment is unavailable for orthophoto input.";
      checkbox.title = reason;
      if(hint) hint.textContent = reason;
      window.updatePhotogrammetryExportVisibility?.();
      return;
    }

    checkbox.title = "";
    if(hint) hint.textContent = "Refine prepared-image positions and orientation with LightGlue before generating prediction coordinates.";
    window.updatePhotogrammetryExportVisibility?.();
  }

  function validate(){
    if(!byId("chkAlignImages")?.checked) return null;
    const quality = byId("imageAlignmentQuality")?.value;
    const strictness = byId("imageAlignmentStrictness")?.value;
    const temporalNeighbors = Number(byId("imageAlignmentTemporalNeighbors")?.value);
    const lateralNeighbors = Number(byId("imageAlignmentLateralNeighbors")?.value);
    const maximumPairRotation = Number(byId("imageAlignmentMaxPairRotation")?.value);
    if(!["standard", "high"].includes(quality)) return "Select a valid matching quality.";
    if(!["strict", "balanced", "lenient"].includes(strictness)) return "Select a valid matching strictness.";
    if(!Number.isInteger(temporalNeighbors) || temporalNeighbors < 1 || temporalNeighbors > 10){
      return "Temporal neighbours must be a whole number between 1 and 10.";
    }
    if(!Number.isInteger(lateralNeighbors) || lateralNeighbors < 0 || lateralNeighbors > 12){
      return "Lateral neighbours must be a whole number between 0 and 12.";
    }
    if(!Number.isFinite(maximumPairRotation) || maximumPairRotation < 0 || maximumPairRotation > 30){
      return "Maximum pair rotation difference must be between 0 and 30 degrees.";
    }
    return null;
  }

  function appendFormData(formData){
    const checkbox = byId("chkAlignImages");
    const enabled = checkbox?.checked === true && checkbox.disabled !== true;
    formData.append("image_alignment_mode", enabled ? "lightglue" : "none");
    if(!enabled) return;
    formData.append("image_alignment_quality", byId("imageAlignmentQuality").value);
    formData.append("image_alignment_strictness", byId("imageAlignmentStrictness").value);
    formData.append("image_alignment_temporal_neighbors", byId("imageAlignmentTemporalNeighbors").value);
    formData.append("image_alignment_lateral_neighbors", byId("imageAlignmentLateralNeighbors").value);
    formData.append("image_alignment_max_pair_rotation_deg", byId("imageAlignmentMaxPairRotation").value);
  }

  function init(){
    const checkbox = byId("chkAlignImages");
    const options = byId("imageAlignmentOptions");
    checkbox?.addEventListener("change", () => {
      if(options) options.hidden = !checkbox.checked;
      window.updatePhotogrammetryExportVisibility?.();
    });
    byId("btnImageAlignmentInfo")?.addEventListener("click", () => setInfoOpen(true));
    byId("btnCloseImageAlignmentInfo")?.addEventListener("click", () => setInfoOpen(false));
    byId("btnCloseImageAlignmentInfoFooter")?.addEventListener("click", () => setInfoOpen(false));
    byId("imageAlignmentInfoModal")?.addEventListener("click", event => {
      if(event.target === byId("imageAlignmentInfoModal")) setInfoOpen(false);
    });
    syncAvailability();
  }

  window.TestImageAlignment = { init, validate, appendFormData, syncAvailability };
  if(document.readyState === "loading") document.addEventListener("DOMContentLoaded", init, { once: true });
  else init();
})();
