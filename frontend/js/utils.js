/* Shared frontend helpers */

window.api = {
  datasets: "/api/test_datasets",
  testDatasets: "/api/test-datasets",
  models: "/api/models",
  trainingDatasets: "/api/training-datasets",
  trainingDatasetUpload: "/api/training-datasets/upload",
  upload: "/api/test_upload",
  testRun: "/api/test_run",
  train: "/api/train",
  cancel: "/api/cancel",
  logs: "/api/logs",
  sessions: "/api/sessions",
  results: "/api/results",
  sessionSummary: "/api/session_summary",
  sessionTiles: "/api/session_tiles",
  projects: "/api/projects",
  activeProject: "/api/active-project",
  features: "/api/features"
};

let appConfirmationResolver = null;
let appConfirmationWired = false;

function finishAppConfirmation(confirmed){
  const modal = document.getElementById("appConfirmModal");
  modal?.classList.remove("show");
  modal?.classList.add("hidden");
  const resolve = appConfirmationResolver;
  appConfirmationResolver = null;
  if(resolve) resolve(Boolean(confirmed));
}

function wireAppConfirmation(){
  if(appConfirmationWired) return;
  appConfirmationWired = true;
  document.getElementById("appConfirmAccept")?.addEventListener("click", ()=>finishAppConfirmation(true));
  document.getElementById("appConfirmCancel")?.addEventListener("click", ()=>finishAppConfirmation(false));
  document.getElementById("appConfirmClose")?.addEventListener("click", ()=>finishAppConfirmation(false));
  document.getElementById("appConfirmModal")?.addEventListener("click", event=>{
    if(event.target === document.getElementById("appConfirmModal")) finishAppConfirmation(false);
  });
}

window.showAppConfirmation = function showAppConfirmation({
  title = "Confirm action",
  message = "Continue?",
  detail = "",
  confirmLabel = "Continue",
  danger = false,
} = {}){
  wireAppConfirmation();
  if(appConfirmationResolver) finishAppConfirmation(false);
  const modal = document.getElementById("appConfirmModal");
  if(!modal) return Promise.resolve(false);
  document.getElementById("appConfirmTitle").textContent = title;
  document.getElementById("appConfirmMessage").textContent = message;
  const detailNode = document.getElementById("appConfirmDetail");
  detailNode.textContent = detail;
  detailNode.hidden = !detail;
  const accept = document.getElementById("appConfirmAccept");
  accept.textContent = confirmLabel;
  accept.classList.toggle("danger", danger);
  modal.classList.remove("hidden");
  modal.classList.add("show");
  accept.focus();
  return new Promise(resolve=>{ appConfirmationResolver = resolve; });
};

let appTextInputResolver = null;
let appTextInputWired = false;

function finishAppTextInput(accepted){
  const modal = document.getElementById("appTextInputModal");
  const input = document.getElementById("appTextInputValue");
  const error = document.getElementById("appTextInputError");
  if(accepted && !input.value.trim()){
    error.textContent = "Name cannot be empty.";
    error.hidden = false;
    input.focus();
    return;
  }
  modal?.classList.remove("show");
  modal?.classList.add("hidden");
  const resolve = appTextInputResolver;
  appTextInputResolver = null;
  if(resolve) resolve(accepted ? input.value.trim() : null);
}

function wireAppTextInput(){
  if(appTextInputWired) return;
  appTextInputWired = true;
  document.getElementById("appTextInputAccept")?.addEventListener("click", ()=>finishAppTextInput(true));
  document.getElementById("appTextInputCancel")?.addEventListener("click", ()=>finishAppTextInput(false));
  document.getElementById("appTextInputClose")?.addEventListener("click", ()=>finishAppTextInput(false));
  document.getElementById("appTextInputModal")?.addEventListener("click", event=>{
    if(event.target === document.getElementById("appTextInputModal")) finishAppTextInput(false);
  });
  document.getElementById("appTextInputValue")?.addEventListener("keydown", event=>{
    if(event.key === "Enter"){
      event.preventDefault();
      finishAppTextInput(true);
    }
  });
}

window.showAppTextInput = function showAppTextInput({
  title = "Update name",
  label = "Name",
  value = "",
  confirmLabel = "Save",
  maxLength = 128,
} = {}){
  wireAppTextInput();
  if(appTextInputResolver) finishAppTextInput(false);
  const modal = document.getElementById("appTextInputModal");
  if(!modal) return Promise.resolve(null);
  document.getElementById("appTextInputTitle").textContent = title;
  document.getElementById("appTextInputLabel").textContent = label;
  const input = document.getElementById("appTextInputValue");
  input.maxLength = Number(maxLength) || 128;
  input.value = value;
  document.getElementById("appTextInputError").hidden = true;
  document.getElementById("appTextInputAccept").textContent = confirmLabel;
  modal.classList.remove("hidden");
  modal.classList.add("show");
  window.setTimeout(()=>{ input.focus(); input.select(); }, 0);
  return new Promise(resolve=>{ appTextInputResolver = resolve; });
};

const featureDefaults = {
  detectron: true,
  yolo: false,
  thermal_data_extraction: true,
  backends: ["detectron"],
};

window.featureDefaults = featureDefaults;
window.featureFlags = { ...featureDefaults };

window.getEnabledBackends = function getEnabledBackends(){
  const flags = window.featureFlags || {};
  if(Array.isArray(flags.backends) && flags.backends.length){
    return flags.backends.slice();
  }
  const list = [];
  if(flags.detectron) list.push("detectron");
  if(flags.yolo) list.push("yolo");
  return list;
};

window.isThermalExtractionEnabled = function isThermalExtractionEnabled(){
  const flags = window.featureFlags || {};
  if(typeof flags.thermal_data_extraction === "boolean"){
    return flags.thermal_data_extraction;
  }
  if(typeof flags.thermal === "boolean"){
    return flags.thermal;
  }
  return true;
};

window.loadFeatureFlags = async function loadFeatureFlags(){
  try{
    const resp = await fetch(window.api.features, { cache: "no-store" });
    if(resp.ok){
      const data = await resp.json().catch(()=>({}));
      if(data && data.features){
        const merged = { ...featureDefaults, ...data.features };
        if(typeof merged.thermal_data_extraction !== "boolean" && typeof merged.thermal === "boolean"){
          merged.thermal_data_extraction = merged.thermal;
        }
        if(typeof merged.thermal !== "boolean" && typeof merged.thermal_data_extraction === "boolean"){
          merged.thermal = merged.thermal_data_extraction;
        }
        window.featureFlags = merged;
      }
    }
  }catch(err){
    console.warn("Failed to load feature flags", err);
  }
  return window.featureFlags;
};

function updateBackendSelectOptions(selectEl, allowed){
  if(!selectEl) return;
  const allowedSet = new Set(allowed);
  Array.from(selectEl.options).forEach(opt => {
    if(opt.value && !allowedSet.has(opt.value)){
      opt.remove();
    }
  });
  if(allowed.length && !allowedSet.has(selectEl.value)){
    selectEl.value = allowed[0];
  }
  if(!allowed.length){
    selectEl.value = "";
  }
  selectEl.disabled = !allowed.length;
}

window.applyFeatureFlags = function applyFeatureFlags(){
  const allowedBackends = window.getEnabledBackends();
  const selects = [
    document.getElementById("selBackend"),
    document.getElementById("selBackendTrain"),
    document.getElementById("selBackendTest")
  ];
  selects.forEach(sel => updateBackendSelectOptions(sel, allowedBackends));

  const disableAll = allowedBackends.length === 0;
  const trainBtn = document.getElementById('btnStartTraining');
  const testBtn = document.getElementById('btnRunTest');
  if(trainBtn) trainBtn.disabled = disableAll;
  if(testBtn) testBtn.disabled = disableAll;
  if(disableAll){
    setText("#trainStatus", "No training backends enabled on this server.");
    setText("#testStatus", "No inference backends enabled on this server.");
  }

  const yoloAvailable = allowedBackends.includes("yolo");
  if(!yoloAvailable){
    ["yoloOptions", "yoloSizeOption"].forEach(id => {
      const el = document.getElementById(id);
      if(el) el.style.display = "none";
    });
  }

  const thermalEnabled = window.isThermalExtractionEnabled ? window.isThermalExtractionEnabled() : true;
  const toggleThermalInput = (inputId) => {
    const input = document.getElementById(inputId);
    if(!input) return;
    if(!thermalEnabled){
      input.checked = false;
    }
    input.disabled = !thermalEnabled;
    const wrapper = input.closest('.switchRow') || input.parentElement;
    if(wrapper && wrapper.style){
      wrapper.style.display = thermalEnabled ? '' : 'none';
    }
  };
  toggleThermalInput('chkUseThermalTrain');
  toggleThermalInput('chkUseThermalTest');
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

const LOG_DEDUP_WINDOW_MS = 3000;

function appendLineToPane(pane, rawLine){
  if(!pane) return;
  if(rawLine == null) return;
  const line = String(rawLine);
  if(!line.trim()) return;
  const now = Date.now();
  const lastLine = pane.dataset.lastLogLine || "";
  const lastStamp = Number(pane.dataset.lastLogStamp || 0);
  if(line === lastLine && now - lastStamp < LOG_DEDUP_WINDOW_MS){
    return; // skip rapid duplicate log entries
  }
  pane.dataset.lastLogLine = line;
  pane.dataset.lastLogStamp = String(now);
  pane.textContent += (pane.textContent ? "\n" : "") + line;
  pane.scrollTop = pane.scrollHeight;
}

window.appendLog = function appendLog(line){
  const pane = $("#logStream");
  appendLineToPane(pane, line);
};

window.appendMiniLog = function appendMiniLog(sel, line){
  const pane = $(sel);
  appendLineToPane(pane, line);
};

window.resetLogPane = function resetLogPane(sel){
  const pane = typeof sel === "string" ? $(sel) : sel;
  if(!pane) return;
  pane.textContent = "";
  delete pane.dataset.lastLogLine;
  delete pane.dataset.lastLogStamp;
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
