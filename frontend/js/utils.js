/* Shared frontend helpers */

window.api = {
  datasets: "/api/test_datasets",
  models: "/api/models",
  upload: "/api/test_upload",
  testRun: "/api/test_run",
  train: "/api/train",
  cancel: "/api/cancel",
  logs: "/api/logs",
  sessions: "/api/sessions",
  sessionSummary: "/api/session_summary",
  sessionTiles: "/api/session_tiles",
  colmapState: "/api/colmap/state",
  colmapCameras: "/api/colmap/cameras",
  colmapStart: "/api/colmap/start",
  colmapFinish: "/api/colmap/finish",
  projects: "/api/projects",
  activeProject: "/api/active-project"
};

window.$ = (sel) => document.querySelector(sel);
window.$$ = (sel) => Array.from(document.querySelectorAll(sel));

window.setHidden = function setHidden(el, hidden = true){
  if(!el) return;
  el.style.display = hidden ? "none" : "block";
};

window.setText = function setText(sel, txt){
  const el = $(sel);
  if(el) el.textContent = txt;
};

window.escapeHtml = function escapeHtml(value){
  return (value || "").replace(/[&<>"']/g, (m) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    "\"": "&quot;",
    "'": "&#39;"
  })[m]);
};

window.closeBtn = function closeBtn(){
  return '<button class="iconBtn alertClose" aria-label="Close">×</button>';
};

window.metersToDeg = function metersToDeg(lat, meters){
  const metersPerDegLat = 111320;
  const metersPerDegLon = 111320 * Math.cos(lat * Math.PI / 180);
  return { dLat: meters / metersPerDegLat, dLon: meters / metersPerDegLon };
};

window.headingToRadians = function headingToRadians(headingDeg, negate = true){
  const rad = Number.isFinite(headingDeg) ? (negate ? -headingDeg : headingDeg) * Math.PI / 180 : 0;
  return rad;
};

window.appendLog = function appendLog(line){
  const pane = $("#logStream");
  if(!pane) return;
  pane.textContent += (pane.textContent ? "\n" : "") + line;
  pane.scrollTop = pane.scrollHeight;
};

window.appendMiniLog = function appendMiniLog(sel, line){
  const pane = $(sel);
  if(!pane) return;
  pane.textContent += (pane.textContent ? "\n" : "") + line;
  pane.scrollTop = pane.scrollHeight;
};

window.clearAlerts = function clearAlerts(prefix){
  for(const id of [`#${prefix}Warn`, `#${prefix}Err`, `#${prefix}Ok`]){
    const el = $(id);
    if(el){
      el.innerHTML = "";
      setHidden(el, true);
    }
  }
};

window.ok = function ok(prefix, msg){
  const el = $(`#${prefix}Ok`);
  if(!el) return;
  el.innerHTML = closeBtn() + escapeHtml(msg);
  setHidden(el, false);
};

window.err = function err(prefix, msg){
  const el = $(`#${prefix}Err`);
  if(!el) return;
  el.innerHTML = closeBtn() + escapeHtml(msg);
  setHidden(el, false);
};

window.warn = function warn(prefix, msg){
  const el = $(`#${prefix}Warn`);
  if(!el) return;
  el.innerHTML = closeBtn() + escapeHtml(msg);
  setHidden(el, false);
};

window.wireAlertClose = function wireAlertClose(){
  $$(".alert").forEach((el) => {
    el.addEventListener("click", (evt) => {
      if(evt.target.classList.contains("alertClose")){
        setHidden(el, true);
      }
    });
  });
};
