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
  activeProject: "/api/active-project",
  features: "/api/features"
};

const featureDefaults = {
  colmap: false,
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

  const colmapEnabled = !!(window.featureFlags && window.featureFlags.colmap);
  const btnOptimize = document.getElementById('btnOptimize');
  if(btnOptimize){
    btnOptimize.style.display = colmapEnabled ? 'inline-block' : 'none';
    if(!colmapEnabled && btnOptimize.classList.contains('active') && typeof window.switchToTab === 'function'){
      window.switchToTab('tab-test');
    }
  }
  const tabOptimize = document.getElementById('tab-optimize');
  if(tabOptimize){
    tabOptimize.style.display = colmapEnabled ? '' : 'none';
  }
  const goOptimizeBtn = document.getElementById('btnGoOptimize');
  if(goOptimizeBtn){
    goOptimizeBtn.style.display = colmapEnabled ? '' : 'none';
  }
  const accurateControls = document.getElementById('accurateControls');
  if(accurateControls){
    accurateControls.style.display = colmapEnabled ? '' : 'none';
  }
  const chkAccurate = document.getElementById('chkAccurateLocations');
  if(chkAccurate){
    if(!colmapEnabled){
      chkAccurate.checked = false;
    }
    chkAccurate.disabled = !colmapEnabled;
  }
  const accurateRows = [
    document.getElementById('accurateModeRow'),
    document.getElementById('useOptimizationFromRow'),
    document.getElementById('mosaicControls')
  ];
  if(!colmapEnabled){
    accurateRows.forEach(row => row && setHidden(row, true));
    const badge = document.getElementById('accurateStatusBadge');
    if(badge){
      badge.className = 'pill pill-muted';
      badge.textContent = 'Disabled';
    }
    const hint = document.getElementById('accurateHint');
    if(hint){
      hint.textContent = 'Accurate poses disabled on this server.';
    }
    if(typeof window.clearColmapPoll === 'function'){
      window.clearColmapPoll();
    }
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
  if(typeof window.updateAccurateUI === 'function'){
    window.updateAccurateUI();
  }
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
