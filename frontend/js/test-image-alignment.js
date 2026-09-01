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
      window.updateWebodmExportVisibility?.();
      return;
    }

    checkbox.title = "";
    if(hint) hint.textContent = "Refine prepared-image positions and orientation with LightGlue before generating prediction coordinates.";
    window.updateWebodmExportVisibility?.();
  }

  function validate(){
    if(!byId("chkAlignImages")?.checked) return null;
    const maximumPosition = Number(byId("imageAlignmentMaxPosition")?.value);
    const maximumRotation = Number(byId("imageAlignmentMaxRotation")?.value);
    if(!Number.isFinite(maximumPosition) || maximumPosition < 0.5 || maximumPosition > 50){
      return "Maximum position correction must be between 0.5 and 50 metres.";
    }
    if(!Number.isFinite(maximumRotation) || maximumRotation < 0 || maximumRotation > 45){
      return "Maximum orientation correction must be between 0 and 45 degrees.";
    }
    return null;
  }

  function appendFormData(formData){
    const checkbox = byId("chkAlignImages");
    const enabled = checkbox?.checked === true && checkbox.disabled !== true;
    formData.append("image_alignment_mode", enabled ? "lightglue" : "none");
    if(!enabled) return;
    formData.append("image_alignment_max_position_m", byId("imageAlignmentMaxPosition").value);
    formData.append("image_alignment_max_rotation_deg", byId("imageAlignmentMaxRotation").value);
  }

  function init(){
    const checkbox = byId("chkAlignImages");
    const options = byId("imageAlignmentOptions");
    checkbox?.addEventListener("change", () => {
      if(options) options.hidden = !checkbox.checked;
      window.updateWebodmExportVisibility?.();
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
