/* global L */

// Project management
let allProjects = [];
// ================= Project Selection Persistence =================
function saveSelectedProject(projectId) {
  /**
   * Save selected project to localStorage
   */
  if (projectId) {
    localStorage.setItem("selectedProjectId", projectId);
  } else {
    localStorage.removeItem("selectedProjectId");
  }
}

function getSelectedProjectFromStorage() {
  /**
   * Retrieve selected project from localStorage
   */
  return localStorage.getItem("selectedProjectId");
}
function getProjectIdFromURL() {
  /**
   * Parse project ID from URL query parameter: ?projectId=<id>
   * Also supports path format as fallback: /projects/<id>/
   */
  const params = new URLSearchParams(window.location.search);
  const projectIdFromQuery = params.get('projectId');
  if (projectIdFromQuery) return projectIdFromQuery;
  
  const pathname = window.location.pathname;
  const match = pathname.match(/\/projects\/([a-f0-9-]+)\//i);
  if (match) return match[1];
  return null;
}

function navigateToProject(projectId) {
  /**
   * Navigate to project page: project.html?projectId=<id>
   */
  console.log("navigateToProject called with projectId:", projectId);
  
  if (!projectId) {
    console.error("navigateToProject: No projectId provided");
    return;
  }
  
  const newUrl = `/project.html?projectId=${projectId}`;
  console.log("Setting window.location.href to:", newUrl);
  window.location.href = newUrl;
  console.log("Navigation triggered (this may not log if page reloads)");
}

function addPopstateListener() {
  /**
   * Listen for browser back button (popstate event)
   * Allows navigation between projects using browser back/forward
   */
  window.addEventListener('popstate', async (event) => {
    const projectId = getProjectIdFromURL();
    if (projectId && activeProject?.id !== projectId) {
      // Switch to the project from URL
      try {
        await activateProject(projectId);
        updateProjectUI();
        switchToTab("tab-test");
      } catch (err) {
        console.error("Failed to switch project from URL:", err);
      }
    }
  });
}

async function loadProjects() {
  try {
    const resp = await fetch(api.projects);
    const data = await resp.json();
    allProjects = data.projects || [];
    allProjectsOriginal = [...allProjects];  // Keep a copy of the original list
    return allProjects;
  } catch (err) {
    console.error("Failed to load projects:", err);
    return [];
  }
}

async function getActiveProject() {
  try {
    const resp = await fetch(api.activeProject);
    if (resp.ok) {
      activeProject = await resp.json();
      return activeProject;
    }
  } catch (err) {
    console.error("Failed to get active project:", err);
  }
  return null;
}

async function createProject(name, description, rootPath) {
  const formData = new FormData();
  formData.append("name", name);
  formData.append("description", description);
  formData.append("root_path", rootPath);
  
  try {
    const resp = await fetch(api.projects, {
      method: "POST",
      body: formData
    });
    if (resp.ok) {
      const project = await resp.json();
      allProjects.push(project);
      return project;
    } else {
      const err = await resp.json();
      throw new Error(err.detail || "Failed to create project");
    }
  } catch (err) {
    console.error("Failed to create project:", err);
    throw err;
  }
}

async function activateProject(projectId) {
  try {
    const resp = await fetch(`${api.projects}/${projectId}/activate`, {
      method: "POST"
    });
    if (resp.ok) {
      const data = await resp.json();
      activeProject = data.project;
      updateProjectUI();
      return data.project;
    } else {
      throw new Error("Failed to activate project");
    }
  } catch (err) {
    console.error("Failed to activate project:", err);
    throw err;
  }
}

async function deleteProject(projectId) {
  try {
    const resp = await fetch(`${api.projects}/${projectId}`, {
      method: "DELETE"
    });
    if (resp.ok) {
      allProjects = allProjects.filter(p => p.id !== projectId);
      return true;
    } else {
      throw new Error("Failed to delete project");
    }
  } catch (err) {
    console.error("Failed to delete project:", err);
    throw err;
  }
}

function updateProjectUI() {
  const currentProjectInfo = $("#currentProjectInfo");
  const currentProjectName = $("#currentProjectName");
  
  if (activeProject) {
    if (currentProjectInfo) currentProjectInfo.style.display = "block";
    if (currentProjectName) currentProjectName.textContent = activeProject.name;
    
    // Show project-specific tabs
    $$(".tabs button[data-tab]").forEach(btn => {
      if (btn.id === "btnProjectsHome") {
        btn.style.display = "none";
      } else {
        btn.style.display = "inline-block";
      }
    });
  } else {
    if (currentProjectInfo) currentProjectInfo.style.display = "none";
    // Hide project-specific tabs, show only Projects tab
    $$(".tabs button[data-tab]").forEach(btn => {
      if (btn.id === "btnProjectsHome") {
        btn.style.display = "inline-block";
      } else {
        btn.style.display = "none";
      }
    });
    // Make sure Projects tab is active
    switchToTab("tab-projects");
  }
}

// runtime caches & UI flags
let testAbort = null;
let modelsCache = {};                // name -> model metadata returned by /api/models
let datasetsCache = [];              // cached dataset list with colmap_ready flag
let testResultsCache = [];            // cached inference result sessions
let userToggledThermalTrain = false; // whether user manually toggled the train thermal checkbox
let userToggledThermalTest = false;  // whether user manually toggled the test thermal checkbox

function populateFolders(list){
  const selTest = $("#selTestFolder");
  const prevTest = selTest?.value || null;
  if(selTest){
    selTest.innerHTML = "";
    list.forEach(d => {
      const o = document.createElement("option");
      o.value = d.name;
      o.textContent = `${d.display_name || d.name} (${d.count})`;
      o.title = `${d.display_name || d.name} · ID: ${d.id || d.name}`;
      selTest.appendChild(o);
    });
    if(prevTest && list.some(d => d.name === prevTest)){
      selTest.value = prevTest;
    }
  }
  const selOpt = document.getElementById('selOptimizeDataset');
  const prevOpt = selOpt?.value || null;
  if(selOpt){
    selOpt.innerHTML = "";
    list.forEach(d => {
      const o = document.createElement("option");
      o.value = d.name;
      o.textContent = d.display_name || d.name;
      selOpt.appendChild(o);
    });
    if(prevOpt && list.some(d => d.name === prevOpt)){
      selOpt.value = prevOpt;
    }else if(selTest && selTest.value){
      selOpt.value = selTest.value;
    }
  }
}
function populateModels(list){
  // Populate a models <select>. Default target is the Test tab model selector.
  const target = "#selModelFolder";
  return populateModelsInto(list, target);
}

// more flexible populate: target selector can be provided
function populateModelsInto(list, selSelector){
  const sel = document.querySelector(selSelector);
  if(!sel) return;
  sel.innerHTML = "";
  // refresh models cache for client-side lookups
  try { modelsCache = {}; } catch(_) { modelsCache = {}; }
  const fmt = (m) => {
    // Prefer explicit model_name from meta (often contains a readable run name)
    function prettyModelType(t){
      if(!t) return '';
      t = String(t);
      // Use compact detectron-style short names for dropdown labels
      const tl = t.toLowerCase();
      if(tl.includes('faster')) return 'fastrcnn';
      // fallback: remove spaces/underscores and non-alphanumerics for compactness
      return tl.replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '');
    }

    if (m.display_name) return String(m.display_name);
    if (m.model_name) {
      const mt = prettyModelType(m.model_type || '');
      // try to pick up num_classes from several possible metadata locations
      const nc = (m.num_classes != null) ? Number(m.num_classes)
                 : (m.model_meta && m.model_meta.num_classes != null) ? Number(m.model_meta.num_classes)
                 : null;
      const ncs = (nc != null) ? `-${nc}cls` : '';
      return `${m.model_name}${mt ? `_${mt}` : ''}${ncs}`;
    }
    // Fallback: construct a compact run-like label: <run>_<backend>_<input_mode>
    const parts = [m.name];
    if (m.backend) parts.push(String(m.backend));
  const mtf = m.model_type ? prettyModelType(m.model_type) : '';
    if (mtf) parts.push(mtf);
    let out = parts.join('_');
  const nc2 = (m.num_classes != null) ? Number(m.num_classes)
        : (m.model_meta && m.model_meta.num_classes != null) ? Number(m.model_meta.num_classes)
        : null;
  if (nc2 != null) out = `${out}-${nc2}cls`;
    return out;
  };

  list.forEach(m => {
    // cache metadata by run name for later decisions (e.g., channel_count)
    if (m && m.name) modelsCache[m.name] = m;
    const o = document.createElement("option");
    o.value = m.name;
    o.textContent = fmt(m);
    sel.appendChild(o);
  });

  // If populating the primary model selector (test-run), auto-set the test thermal checkbox
  // according to the selected model's thermal_used or thermal_only flags
  if (selSelector === '#selModelFolder'){
    const selected = sel.value || sel.options[0]?.value;
    const m = modelsCache[selected];
    const defaultRequiresThermal = (m && m.channel_count) ? (
      Number(m.channel_count) === 3 && (!!(m && m.thermal_only) || !!(m && m.thermal_used))
    ) : false;
    const chkTest = document.getElementById('chkUseThermalTest');
    const thermalEnabled = (typeof window.isThermalExtractionEnabled === 'function')
      ? window.isThermalExtractionEnabled()
      : true;
    if (chkTest){
      if (!thermalEnabled){
        try { chkTest.checked = false; } catch(_){ }
      } else if (!userToggledThermalTest){
        try { chkTest.checked = !!defaultRequiresThermal; } catch(_){ }
      }
    }
  }
}

function getSelectedDataset(){ return $("#selTestFolder").value || null; }
function getSelectedModel(){ return $("#selModelFolder").value || null; }
function getSelectedBackend(){
  // Decide backend based on the currently active tab.
  const activeTabBtn = document.querySelector('.tabs button.active');
  const activeTab = activeTabBtn?.dataset?.tab;
  const selGlobal = $("#selBackend");
  const selTrain = $("#selBackendTrain");
  const selTest  = $("#selBackendTest");
  const fallbackList = (typeof window.getEnabledBackends === 'function') ? window.getEnabledBackends() : [];
  const fallback = fallbackList[0] || 'detectron';

  if(activeTab === 'tab-train'){
    if(selTrain && selTrain.value) return selTrain.value;
    if(selGlobal && selGlobal.value) return selGlobal.value;
    return fallback;
  }

  if(activeTab === 'tab-test'){
    if(selTest && selTest.value) return selTest.value;
    if(selGlobal && selGlobal.value) return selGlobal.value;
    return fallback;
  }

  // default fallback (prefer train selector if present)
  if(selTrain && selTrain.value) return selTrain.value;
  if(selTest && selTest.value) return selTest.value;
  if(selGlobal && selGlobal.value) return selGlobal.value;
  return fallback;
}
function getYoloOptions(){
  return {
    family: $("#selYoloFamily") ? $("#selYoloFamily").value : 'v8',
    size: $("#selYoloSize") ? $("#selYoloSize").value : 's',
  };
}

// ---------- tabs ----------
function setupTabs(){
  $$(".tabs button").forEach(btn=>{
    btn.addEventListener("click", ()=>{
      const id = btn.dataset.tab;
      switchToTab(id);
    });
  });
}

let _resultsTabLoaded = false;
let _resultsTabLoading = null;
let _mapTabLoaded = false;
let _mapTabLoading = null;
const _mapDetachedRasterLayers = new Set();

async function ensureResultsTabLoaded(force = false){
  if(!force && _resultsTabLoaded) return;
  if(!force && _resultsTabLoading) return _resultsTabLoading;

  _resultsTabLoading = (async ()=>{
    await loadSessions(true);
    if($("#selResults").value){
      await showResultsForSelected();
    }
    _resultsTabLoaded = true;
  })().finally(()=>{ _resultsTabLoading = null; });

  return _resultsTabLoading;
}

async function ensureMapTabLoaded(force = false){
  if(!force && _mapTabLoaded) return;
  if(!force && _mapTabLoading) return _mapTabLoading;

  _mapTabLoading = (async ()=>{
    await loadSessions(true);
    if($("#selMapSession").value){
      await refreshMapSessionSelected();
    }
    _mapTabLoaded = true;
  })().finally(()=>{ _mapTabLoading = null; });

  return _mapTabLoading;
}

async function showSharedOverlayOnMap(overlay){
  switchToTab("tab-map");
  await ensureMapTabLoaded();
  if(typeof window.addSavedGeoJsonOverlay !== "function"){
    throw new Error("Map overlay support is not available.");
  }
  await window.addSavedGeoJsonOverlay(overlay, { show: true, focus: true });
  setTimeout(()=> MAP?.invalidateSize(), 30);
}

window.showSharedOverlayOnMap = showSharedOverlayOnMap;

function switchToTab(tabId){
  $$(".tabs button").forEach(b=>b.classList.toggle("active", b.dataset.tab === tabId));
  $$(".tabPanel").forEach(p=>p.classList.toggle("active", p.id === tabId));
  setMapRasterLayersAttached(tabId === "tab-map");
  if(tabId === "tab-map" && MAP){ 
    setTimeout(()=>{
      MAP.invalidateSize();
      if(pendingMapBounds){
        MAP.fitBounds(pendingMapBounds.bounds, pendingMapBounds.options);
        pendingMapBounds = null;
      }
    }, 30);
  }
  if(tabId === "tab-optimize"){ invalidateOptimizeMap(); }
  if(tabId === "tab-logs"){
    const pane = $("#logStream");
    if (pane) pane.scrollTop = pane.scrollHeight;
  }
  if(tabId === "tab-results"){
    ensureResultsTabLoaded();
  }
  if(tabId === "tab-postprocess"){
    window.PostProcessTab?.activate();
  }
  if(tabId === "tab-map"){
    ensureMapTabLoaded().then(() => window.refreshReferencedMapOverlays?.());
  }
  if(tabId === "tab-train"){
    if(document.getElementById('btnAssetData')?.classList.contains('active')) loadTrainingDatasets();
    else loadTrainedModels();
  }
  if(tabId === "tab-test"){
    if(document.getElementById('btnTestAssetResults')?.classList.contains('active')) loadSessions(false);
    else loadDatasets();
  }
}

function setMapRasterLayersAttached(attached){
  if(!MAP) return;
  if(attached){
    for(const layer of _mapDetachedRasterLayers){
      try{ if(!MAP.hasLayer(layer)) layer.addTo(MAP); }catch(_){ }
    }
    _mapDetachedRasterLayers.clear();
    return;
  }
  for(const layer of [imagesLayerGroup, TIF_TILE_GROUP]){
    if(!layer || !MAP.hasLayer(layer)) continue;
    _mapDetachedRasterLayers.add(layer);
    try{ MAP.removeLayer(layer); }catch(_){ }
  }
}

// ---------- datasets/models/sessions ----------
const _inFlightApi = new Map();

function _dedupeRequest(key, requestFn){
  if (_inFlightApi.has(key)) return _inFlightApi.get(key);
  const p = Promise.resolve()
    .then(requestFn)
    .finally(() => _inFlightApi.delete(key));
  _inFlightApi.set(key, p);
  return p;
}

async function _fetchDatasetsOnce(){
  return _dedupeRequest("datasets", async () => {
    const res = await fetch(api.datasets);
    return res.json();
  });
}

async function _fetchModelsOnce(backend){
  const key = `models:${backend || ''}`;
  return _dedupeRequest(key, async () => {
    let url = api.models;
    if(backend) url = `${api.models}?backend=${encodeURIComponent(backend)}`;
    const res = await fetch(url);
    return res.json();
  });
}

async function _fetchSessionsOnce(){
  return _dedupeRequest("sessions", async () => {
    const res = await fetch(api.sessions);
    return res.json();
  });
}

async function loadDatasets(){
  const js = await _fetchDatasetsOnce();
  if(js.ok){
    datasetsCache = js.datasets || [];
    populateFolders(js.datasets);
    renderTestDatasets(datasetsCache);
    onTestDatasetChange();
    if(window.featureFlags?.colmap){
      onOptimizeDatasetChange();
    }else if(typeof updateOptimizePanel === 'function'){
      updateOptimizePanel(null);
    }
  }
}

function switchTestAssetPanel(panel){
  const showData = panel !== 'results';
  const dataTab = document.getElementById('btnTestAssetData');
  const resultsTab = document.getElementById('btnTestAssetResults');
  const dataPanel = document.getElementById('testDataPanel');
  const resultsPanel = document.getElementById('testResultsPanel');
  dataTab?.classList.toggle('active', showData);
  resultsTab?.classList.toggle('active', !showData);
  dataTab?.setAttribute('aria-selected', String(showData));
  resultsTab?.setAttribute('aria-selected', String(!showData));
  if(dataTab) dataTab.tabIndex = showData ? 0 : -1;
  if(resultsTab) resultsTab.tabIndex = showData ? -1 : 0;
  if(dataPanel) dataPanel.hidden = !showData;
  if(resultsPanel) resultsPanel.hidden = showData;
  if(showData) loadDatasets();
  else loadSessions(false);
}

function testAssetCard(asset, kind){
  const isResult = kind === 'result';
  const item = document.createElement('article');
  item.className = 'trainedModelItem';
  const main = document.createElement('div');
  main.className = 'trainedModelMain testAssetTrigger';
  main.tabIndex = 0;
  main.setAttribute('role', 'button');
  const displayName = asset.display_name || asset.name;
  main.setAttribute('aria-label', isResult ? `Open result ${displayName}` : `Select test dataset ${displayName}`);
  const name = document.createElement('strong');
  name.className = 'trainedModelName';
  name.textContent = displayName;
  name.title = displayName;
  const id = document.createElement('small');
  id.className = 'trainedModelId';
  id.textContent = `ID: ${asset.id || asset.name}`;
  id.title = asset.id || asset.name;
  const meta = document.createElement('div');
  meta.className = 'trainedModelMeta';
  const resultComplete = asset.complete !== false && asset.status !== 'incomplete';
  const details = isResult
    ? [resultComplete ? 'Complete' : 'Incomplete', asset.mtime ? new Date(asset.mtime * 1000).toLocaleDateString() : null]
    : [`${asset.count || 0} images`, asset.colmap_ready ? 'Optimized poses' : null, asset.mtime ? new Date(asset.mtime * 1000).toLocaleDateString() : null];
  if(isResult) item.classList.add(resultComplete ? 'complete' : 'incomplete');
  details.filter(Boolean).forEach(value => {
    const span = document.createElement('span');
    span.textContent = value;
    meta.appendChild(span);
  });
  main.append(name, id, meta);

  const open = async () => {
    if(isResult){
      switchToTab('tab-results');
      await ensureResultsTabLoaded();
      const select = document.getElementById('selResults');
      if(select){ select.value = asset.name; await showResultsForSelected(); }
    }else{
      const select = document.getElementById('selTestFolder');
      if(select){ select.value = asset.name; onTestDatasetChange(); }
    }
  };
  main.addEventListener('click', open);
  main.addEventListener('keydown', event => {
    if(event.key === 'Enter' || event.key === ' '){ event.preventDefault(); open(); }
  });

  const menuButton = document.createElement('button');
  menuButton.type = 'button';
  menuButton.className = 'iconDots';
  menuButton.textContent = '⋮';
  menuButton.title = `Actions for ${displayName}`;
  menuButton.setAttribute('aria-label', menuButton.title);
  menuButton.setAttribute('aria-expanded', 'false');
  const menu = document.createElement('div');
  menu.className = 'trainedModelMenu';
  menu.hidden = true;
  const renameButton = document.createElement('button');
  renameButton.type = 'button';
  renameButton.textContent = 'Rename';
  renameButton.addEventListener('click', async event => {
    event.stopPropagation();
    try{ await renameTestAsset(asset, kind); }
    catch(error){ alert(error.message || `Could not rename ${isResult ? 'result' : 'test dataset'}.`); }
  });
  const deleteButton = document.createElement('button');
  deleteButton.type = 'button';
  deleteButton.className = 'danger';
  deleteButton.textContent = 'Delete';
  deleteButton.addEventListener('click', async event => {
    event.stopPropagation();
    try{ await deleteTestAsset(asset, kind); }
    catch(error){ alert(error.message || `Could not delete ${isResult ? 'result' : 'test dataset'}.`); }
  });
  menu.append(renameButton, deleteButton);
  menuButton.addEventListener('click', event => {
    event.stopPropagation();
    const willOpen = menu.hidden;
    closeTrainedModelMenus(menu);
    menu.hidden = !willOpen;
    menuButton.setAttribute('aria-expanded', String(willOpen));
  });
  item.append(main, menuButton, menu);
  return item;
}

function renderTestAssetList(listId, assets, kind){
  const list = document.getElementById(listId);
  if(!list) return;
  list.replaceChildren();
  list.classList.remove('muted');
  if(!assets.length){
    const empty = document.createElement('p');
    empty.className = 'trainedModelsEmpty muted tiny';
    empty.textContent = kind === 'result' ? 'No model results yet.' : 'No test datasets yet.';
    list.appendChild(empty);
    return;
  }
  assets.forEach(asset => list.appendChild(testAssetCard(asset, kind)));
}

function renderTestDatasets(datasets){ renderTestAssetList('testDatasetsList', datasets || [], 'dataset'); }
function renderTestResults(results){ renderTestAssetList('testResultsList', results || [], 'result'); }

async function renameTestAsset(asset, kind){
  closeTrainedModelMenus();
  const oldName = asset.display_name || asset.name;
  const nextName = prompt(kind === 'result' ? 'Rename model result' : 'Rename test dataset', oldName);
  if(nextName === null || !nextName.trim() || nextName.trim() === oldName) return;
  const base = kind === 'result' ? api.results : api.testDatasets;
  const body = new FormData();
  body.append('name', nextName.trim());
  const response = await fetch(`${base}/${encodeURIComponent(asset.id || asset.name)}/rename`, {method:'POST', body});
  const result = await response.json().catch(()=>({}));
  if(!response.ok || !result.ok) throw new Error(result.detail || 'Rename failed.');
  if(kind === 'result') await loadSessions(false);
  else await loadDatasets();
}

async function deleteTestAsset(asset, kind){
  closeTrainedModelMenus();
  const label = asset.display_name || asset.name;
  const noun = kind === 'result' ? 'model result' : 'test dataset';
  const explanation = kind === 'result'
    ? 'This removes the saved inference result files. The model and test data are not deleted.'
    : 'This removes the uploaded test files. Trained models and saved results are not deleted.';
  if(!confirm(`Delete ${noun} "${label}"?\n\n${explanation}`)) return;
  const base = kind === 'result' ? api.results : api.testDatasets;
  const response = await fetch(`${base}/${encodeURIComponent(asset.id || asset.name)}`, {method:'DELETE'});
  const result = await response.json().catch(()=>({}));
  if(!response.ok || !result.ok) throw new Error(result.detail || 'Delete failed.');
  if(kind === 'result'){
    _resultsTabLoaded = false;
    await loadSessions(true);
  }else await loadDatasets();
}
// Load models, optionally filtered by backend (e.g., ?backend=yolo)
// targetSel - optional selector string for which <select> to populate (defaults to '#selModelFolder')
async function loadModels(backend, targetSel = '#selModelFolder'){
  try{
    const js = await _fetchModelsOnce(backend);
    if(js.ok){
      let models = js.models || [];
      // Server may not honor backend query; do client-side filtering as a safe fallback
      if(backend && models.length && models[0].backend !== undefined){
        models = models.filter(m => String(m.backend || '').toLowerCase() === String(backend || '').toLowerCase());
      }
      populateModelsInto(models, targetSel);
    }
  }catch(e){ console.warn('loadModels failed', e); }
}

function closeTrainedModelMenus(exceptMenu = null){
  document.querySelectorAll('.trainedModelMenu').forEach(menu => {
    if(menu !== exceptMenu) menu.hidden = true;
  });
}

function prettyModelType(value){
  const raw = String(value || '').trim();
  const normalized = raw.toLowerCase().replace(/[_-]+/g, '');
  if(normalized.includes('maskrcnn')) return 'Mask R-CNN';
  if(normalized.includes('fasterrcnn')) return 'Faster R-CNN';
  if(normalized.startsWith('yolo')) return raw.toUpperCase().replace('-SEG', ' Segmentation');
  return raw;
}

function trainedModelType(model){
  const backend = String(model.backend || 'unknown').toUpperCase();
  const type = String(model.model_type || '').trim();
  const task = String(model.task || '').toLowerCase();
  const taskLabel = task === 'segment' ? 'Instance segmentation' : task === 'detect' ? 'Object detection' : '';
  return [backend, type ? prettyModelType(type) : '', taskLabel].filter(Boolean).join(' · ');
}

async function refreshModelViews(){
  const testBackend = document.getElementById('selBackendTest')?.value || getSelectedBackend();
  await Promise.all([
    loadTrainedModels(),
    loadModels(testBackend, '#selModelFolder'),
  ]);
}

async function renameTrainedModel(model){
  closeTrainedModelMenus();
  const oldName = model.display_name || model.model_name || model.name;
  const nextName = prompt('Rename model', oldName);
  if(nextName === null || !nextName.trim() || nextName.trim() === oldName) return;

  const body = new FormData();
  body.append('name', nextName.trim());
  const response = await fetch(`${api.models}/${encodeURIComponent(model.id || model.name)}/rename`, {
    method: 'POST',
    body,
  });
  const result = await response.json().catch(()=>({}));
  if(!response.ok || !result.ok) throw new Error(result.detail || 'Could not rename model.');
  await refreshModelViews();
}

async function deleteTrainedModel(model){
  closeTrainedModelMenus();
  const label = model.display_name || model.model_name || model.name;
  const kind = model.complete === false ? 'incomplete training run' : 'model';
  const confirmed = confirm(`Delete ${kind} "${label}"?\n\nThis removes its trained-model output files. Training data will not be deleted.`);
  if(!confirmed) return;

  const response = await fetch(`${api.models}/${encodeURIComponent(model.id || model.name)}`, {
    method: 'DELETE',
  });
  const result = await response.json().catch(()=>({}));
  if(!response.ok || !result.ok) throw new Error(result.detail || 'Could not delete model.');
  await refreshModelViews();
}

async function openModelDetails(modelId){
  const modal = document.getElementById('modelDetailsModal');
  const body = document.getElementById('modelDetailsBody');
  const title = document.getElementById('modelDetailsTitle');
  if(!modal || !body) return;
  body.textContent = 'Loading model details…';
  modal.classList.add('show');
  modal.classList.remove('hidden');
  try{
    const response = await fetch(`${api.models}/${encodeURIComponent(modelId)}`, {cache:'no-store'});
    const result = await response.json().catch(()=>({}));
    if(!response.ok || !result.ok) throw new Error(result.detail || 'Could not load model details.');
    const model = result.model || {};
    const meta = result.meta || {};
    const metrics = result.latest_metrics || {};
    if(title) title.textContent = model.display_name || model.name || 'Model details';
    const lossDetails = {};
    if(metrics.iteration != null) lossDetails.iteration = metrics.iteration;
    if(metrics.lr != null) lossDetails.lr = metrics.lr;
    Object.entries(metrics).forEach(([key, value]) => {
      const normalized = key.toLowerCase();
      if(normalized.includes('loss') || normalized.includes('/ap')) lossDetails[key] = value;
    });
    const metaCard = Object.keys(meta).length
      ? `<section class="info-card"><h4>Model Meta</h4>${renderJsonList(meta)}</section>`
      : '<section class="info-card"><h4>Model Meta</h4><p class="muted tiny">No model_meta.json metadata is available for this run.</p></section>';
    const lossCard = Object.keys(lossDetails).length
      ? `<section class="info-card"><h4>Training Losses</h4>${renderJsonList(lossDetails)}</section>`
      : '<section class="info-card"><h4>Training Losses</h4><p class="muted tiny">No loss metrics were recorded for this run.</p></section>';
    body.innerHTML = `<div class="info-grid">${metaCard}${lossCard}</div>`;
  }catch(error){
    body.textContent = error.message || 'Could not load model details.';
  }
}

function closeModelDetails(){
  const modal = document.getElementById('modelDetailsModal');
  modal?.classList.remove('show');
  modal?.classList.add('hidden');
}

function renderTrainedModels(models){
  const list = document.getElementById('trainedModelsList');
  if(!list) return;
  list.classList.remove('muted');
  list.replaceChildren();
  if(!models.length){
    const empty = document.createElement('p');
    empty.className = 'trainedModelsEmpty muted tiny';
    empty.textContent = 'No trained models yet.';
    list.appendChild(empty);
    return;
  }

  models.forEach(model => {
    const item = document.createElement('article');
    item.className = 'trainedModelItem';

    const main = document.createElement('div');
    main.className = 'trainedModelMain modelDetailsTrigger';
    main.tabIndex = 0;
    main.setAttribute('role', 'button');
    main.setAttribute('aria-label', `View details for ${model.display_name || model.name}`);
    const name = document.createElement('strong');
    name.className = 'trainedModelName';
    name.textContent = model.display_name || model.model_name || model.name;
    name.title = name.textContent;
    const id = document.createElement('small');
    id.className = 'trainedModelId';
    id.textContent = `ID: ${model.id || model.name}`;
    id.title = model.id || model.name;
    const meta = document.createElement('div');
    meta.className = 'trainedModelMeta';
    const date = model.mtime ? new Date(model.mtime * 1000).toLocaleDateString() : null;
    const details = [];
    if(model.complete === false){
      details.push('Incomplete', trainedModelType(model), date);
      item.classList.add('incomplete');
    }else{
      const input = model.thermal_used || model.input_mode === 'thermal' ? 'Thermal' : 'RGB';
      const classes = model.num_classes != null ? `${model.num_classes} classes` : null;
      details.push('Complete', trainedModelType(model), input, classes, date);
      item.classList.add('complete');
    }
    details.filter(Boolean).forEach(value => {
      const span = document.createElement('span');
      span.textContent = value;
      meta.appendChild(span);
    });
    main.append(name, id, meta);
    const openDetails = () => openModelDetails(model.id || model.name);
    main.addEventListener('click', openDetails);
    main.addEventListener('keydown', event => {
      if(event.key === 'Enter' || event.key === ' '){ event.preventDefault(); openDetails(); }
    });

    const menuButton = document.createElement('button');
    menuButton.type = 'button';
    menuButton.className = 'iconDots';
    menuButton.textContent = '⋮';
    menuButton.title = `Actions for ${name.textContent}`;
    menuButton.setAttribute('aria-label', menuButton.title);
    menuButton.setAttribute('aria-expanded', 'false');

    const menu = document.createElement('div');
    menu.className = 'trainedModelMenu';
    menu.hidden = true;
    const renameButton = document.createElement('button');
    renameButton.type = 'button';
    renameButton.textContent = 'Rename';
    if(model.complete === false){
      renameButton.disabled = true;
      renameButton.title = 'Incomplete training runs cannot be renamed.';
    }
    renameButton.addEventListener('click', async () => {
      try{ await renameTrainedModel(model); }
      catch(error){ alert(error.message || 'Could not rename model.'); }
    });
    const deleteButton = document.createElement('button');
    deleteButton.type = 'button';
    deleteButton.className = 'danger';
    deleteButton.textContent = 'Delete';
    deleteButton.addEventListener('click', async () => {
      try{ await deleteTrainedModel(model); }
      catch(error){ alert(error.message || 'Could not delete model.'); }
    });
    menu.append(renameButton, deleteButton);
    menuButton.addEventListener('click', event => {
      event.stopPropagation();
      const willOpen = menu.hidden;
      closeTrainedModelMenus(menu);
      menu.hidden = !willOpen;
      menuButton.setAttribute('aria-expanded', String(willOpen));
    });
    item.append(main, menuButton, menu);
    list.appendChild(item);
  });
}

async function loadTrainedModels(){
  const list = document.getElementById('trainedModelsList');
  if(!list) return;
  try{
    const response = await fetch(`${api.models}?include_incomplete=true`, {cache:'no-store'});
    const result = await response.json();
    if(!response.ok || !result.ok) throw new Error(result.detail || 'Could not load models.');
    renderTrainedModels(result.models || []);
  }catch(error){
    list.textContent = error.message || 'Could not load trained models.';
    list.classList.add('muted');
  }
}

let trainingDatasetsCache = [];
let trainingUploadXhr = null;
let preferredTrainingDatasetId = '';
let selectedTrainingFolderFiles = [];
let selectedTrainingFolderName = '';

function switchTrainingAssetPanel(panel){
  const showData = panel === 'data';
  const modelButton = document.getElementById('btnAssetModels');
  const dataButton = document.getElementById('btnAssetData');
  const modelPanel = document.getElementById('trainingModelsPanel');
  const dataPanel = document.getElementById('trainingDataPanel');
  if(modelButton){
    modelButton.classList.toggle('active', !showData);
    modelButton.setAttribute('aria-selected', String(!showData));
    modelButton.tabIndex = showData ? -1 : 0;
  }
  if(dataButton){
    dataButton.classList.toggle('active', showData);
    dataButton.setAttribute('aria-selected', String(showData));
    dataButton.tabIndex = showData ? 0 : -1;
  }
  if(modelPanel) modelPanel.hidden = showData;
  if(dataPanel) dataPanel.hidden = !showData;
  if(showData) loadTrainingDatasets();
  else loadTrainedModels();
}

function trainingDatasetImageCount(dataset){
  const splits = dataset?.validation?.splits || {};
  return Object.values(splits).reduce((total, split) => total + Number(split?.images || 0), 0);
}

function trainingDatasetFormatLabel(dataset){
  const validation = dataset?.validation || {};
  const backends = validation.training_backends || [];
  if(backends.includes('detectron') && backends.includes('yolo')) return 'COCO + YOLO';
  return (validation.compatible_formats || []).map(value => String(value).toUpperCase()).join(' + ') || 'Unknown format';
}

async function renameTrainingDataset(dataset){
  closeTrainedModelMenus();
  const oldName = dataset.display_name || 'Training dataset';
  const nextName = prompt('Rename training data', oldName);
  if(nextName === null || !nextName.trim() || nextName.trim() === oldName) return;
  const body = new FormData();
  body.append('name', nextName.trim());
  const response = await fetch(`${api.trainingDatasets}/${encodeURIComponent(dataset.id)}/rename`, {
    method: 'POST',
    body,
  });
  const result = await response.json().catch(()=>({}));
  if(!response.ok || !result.ok) throw new Error(result.detail || 'Could not rename training data.');
  await loadTrainingDatasets();
}

async function deleteTrainingDataset(dataset){
  closeTrainedModelMenus();
  const label = dataset.display_name || 'Training dataset';
  const confirmed = confirm(
    `Delete training data "${label}"?\n\n` +
    'This permanently removes the uploaded dataset folder and its registered files. ' +
    'The original source folder you selected is not affected.'
  );
  if(!confirmed) return;
  const response = await fetch(`${api.trainingDatasets}/${encodeURIComponent(dataset.id)}`, {
    method: 'DELETE',
  });
  const result = await response.json().catch(()=>({}));
  if(!response.ok || !result.ok) throw new Error(result.detail || 'Could not delete training data.');
  closeTrainingDatasetSummary();
  await loadTrainingDatasets();
}

function formatBytes(value){
  const bytes = Number(value || 0);
  if(bytes < 1024) return `${bytes} B`;
  const units = ['KB', 'MB', 'GB', 'TB'];
  let amount = bytes / 1024;
  let index = 0;
  while(amount >= 1024 && index < units.length - 1){ amount /= 1024; index += 1; }
  return `${amount.toFixed(amount >= 10 ? 1 : 2)} ${units[index]}`;
}

function renderTrainingDatasets(datasets){
  const list = document.getElementById('trainingDatasetsList');
  if(!list) return;
  list.classList.remove('muted');
  list.replaceChildren();
  if(!datasets.length){
    const empty = document.createElement('p');
    empty.className = 'trainedModelsEmpty muted tiny';
    empty.textContent = 'No uploaded training data yet.';
    list.appendChild(empty);
    return;
  }
  datasets.forEach(dataset => {
    const item = document.createElement('article');
    item.className = 'trainedModelItem trainingDatasetItem';

    const main = document.createElement('div');
    main.className = 'trainedModelMain';
    main.tabIndex = 0;
    main.setAttribute('role', 'button');
    main.setAttribute('aria-label', `View ${dataset.display_name || 'training dataset'}`);
    const name = document.createElement('strong');
    name.className = 'trainedModelName';
    name.textContent = dataset.display_name || 'Training dataset';
    const id = document.createElement('small');
    id.className = 'trainedModelId';
    id.textContent = `ID: ${dataset.id}`;
    id.title = dataset.id;
    const meta = document.createElement('div');
    meta.className = 'trainedModelMeta';
    const formats = trainingDatasetFormatLabel(dataset);
    const imageCount = trainingDatasetImageCount(dataset);
    const classCount = dataset.validation?.classes?.length || 0;
    [formats, `${imageCount} images`, `${classCount} classes`].forEach(value => {
      const span = document.createElement('span');
      span.textContent = value;
      meta.appendChild(span);
    });
    const status = document.createElement('span');
    const datasetReady = Boolean(dataset.available && dataset.validation?.valid);
    status.className = `trainingDatasetStatus ${datasetReady ? 'ready' : dataset.available ? 'invalid' : 'missing'}`;
    status.textContent = datasetReady ? 'Ready' : dataset.available ? 'Validation failed' : 'Folder missing';
    meta.appendChild(status);
    main.append(name, id, meta);

    const open = () => openTrainingDatasetSummary(dataset.id);
    main.addEventListener('click', open);
    main.addEventListener('keydown', event => {
      if(event.key === 'Enter' || event.key === ' '){ event.preventDefault(); open(); }
    });

    const menuButton = document.createElement('button');
    menuButton.type = 'button';
    menuButton.className = 'iconDots';
    menuButton.textContent = '⋮';
    menuButton.title = `Actions for ${name.textContent}`;
    menuButton.setAttribute('aria-label', menuButton.title);
    menuButton.setAttribute('aria-expanded', 'false');

    const menu = document.createElement('div');
    menu.className = 'trainedModelMenu';
    menu.hidden = true;
    const renameButton = document.createElement('button');
    renameButton.type = 'button';
    renameButton.textContent = 'Rename';
    renameButton.addEventListener('click', async event => {
      event.stopPropagation();
      try{ await renameTrainingDataset(dataset); }
      catch(error){ alert(error.message || 'Could not rename training data.'); }
    });
    const deleteButton = document.createElement('button');
    deleteButton.type = 'button';
    deleteButton.className = 'danger';
    deleteButton.textContent = 'Delete';
    if(dataset.source === 'legacy_project_data'){
      deleteButton.disabled = true;
      deleteButton.title = 'Built-in project training data cannot be deleted here.';
    }
    deleteButton.addEventListener('click', async event => {
      event.stopPropagation();
      if(deleteButton.disabled) return;
      try{ await deleteTrainingDataset(dataset); }
      catch(error){ alert(error.message || 'Could not delete training data.'); }
    });
    menu.append(renameButton, deleteButton);
    menuButton.addEventListener('click', event => {
      event.stopPropagation();
      const willOpen = menu.hidden;
      closeTrainedModelMenus(menu);
      menu.hidden = !willOpen;
      menuButton.setAttribute('aria-expanded', String(willOpen));
    });
    item.append(main, menuButton, menu);
    list.appendChild(item);
  });
}

function populateTrainingDatasetOptions(){
  const select = document.getElementById('selTrainingDataset');
  const hint = document.getElementById('trainingDatasetHint');
  const trainButton = document.getElementById('btnStartTraining');
  if(!select) return;
  const backend = document.getElementById('selBackendTrain')?.value || getSelectedBackend();
  const task = document.getElementById('selTrainingTask')?.value || 'detect';
  const previous = preferredTrainingDatasetId || select.value;
  const compatible = trainingDatasetsCache.filter(dataset => {
    const validation = dataset.validation || {};
    const capabilities = validation.training_capabilities || {};
    const tasks = capabilities[backend];
    return Boolean(dataset.available && validation.valid && (validation.training_backends || []).includes(backend)
      && (!Array.isArray(tasks) || tasks.includes(task)));
  });
  select.replaceChildren();
  if(!compatible.length){
    const option = document.createElement('option');
    option.value = '';
    option.textContent = `No validated ${task === 'segment' ? 'segmentation' : 'detection'} data for ${backend === 'yolo' ? 'YOLO' : 'Detectron'}`;
    select.appendChild(option);
    select.disabled = true;
    if(hint) hint.textContent = `Upload or repair training data compatible with ${backend === 'yolo' ? 'YOLO' : 'COCO / Detectron'}.`;
  }else{
    compatible.forEach(dataset => {
      const option = document.createElement('option');
      option.value = dataset.id;
      const images = trainingDatasetImageCount(dataset);
      option.textContent = `${dataset.display_name || 'Training dataset'} (${images} images)`;
      option.title = `${dataset.display_name || 'Training dataset'} · ID: ${dataset.id}`;
      select.appendChild(option);
    });
    const selected = compatible.find(dataset => dataset.id === previous)
      || compatible.find(dataset => dataset.source === 'legacy_project_data')
      || compatible[0];
    select.value = selected.id;
    select.disabled = false;
    preferredTrainingDatasetId = '';
    if(hint) hint.textContent = `${selected.display_name || 'Training dataset'} · ID: ${selected.id}`;
  }
  const enabledBackends = typeof window.getEnabledBackends === 'function' ? window.getEnabledBackends() : [backend];
  if(trainButton) trainButton.disabled = !compatible.length || !enabledBackends.includes(backend);
}

async function loadTrainingDatasets(){
  const list = document.getElementById('trainingDatasetsList');
  if(!list) return;
  try{
    const response = await fetch(api.trainingDatasets, {cache:'no-store'});
    const result = await response.json().catch(()=>({}));
    if(!response.ok || !result.ok) throw new Error(result.detail || 'Could not load training data.');
    trainingDatasetsCache = result.datasets || [];
    renderTrainingDatasets(trainingDatasetsCache);
    populateTrainingDatasetOptions();
  }catch(error){
    list.textContent = error.message || 'Could not load training data.';
    list.classList.add('muted');
    trainingDatasetsCache = [];
    populateTrainingDatasetOptions();
  }
}

function addDatasetSummaryRow(grid, label, value){
  const term = document.createElement('dt');
  term.textContent = label;
  const description = document.createElement('dd');
  description.textContent = value == null || value === '' ? '—' : String(value);
  grid.append(term, description);
}

function addDatasetMessages(body, title, messages, className = ''){
  if(!Array.isArray(messages) || !messages.length) return;
  const section = document.createElement('section');
  section.className = `trainingDatasetSummarySection ${className}`.trim();
  const heading = document.createElement('h4');
  heading.textContent = title;
  const list = document.createElement('ul');
  list.className = 'trainingDatasetMessageList';
  messages.forEach(message => {
    const item = document.createElement('li');
    item.textContent = message;
    list.appendChild(item);
  });
  section.append(heading, list);
  body.appendChild(section);
}

async function openTrainingDatasetSummary(datasetId){
  const modal = document.getElementById('trainingDatasetSummaryModal');
  const body = document.getElementById('trainingDatasetSummaryBody');
  const title = document.getElementById('trainingDatasetSummaryTitle');
  if(!modal || !body) return;
  body.textContent = 'Loading summary…';
  modal.classList.add('show');
  modal.classList.remove('hidden');
  try{
    const response = await fetch(`${api.trainingDatasets}/${encodeURIComponent(datasetId)}`, {cache:'no-store'});
    const result = await response.json().catch(()=>({}));
    if(!response.ok || !result.ok) throw new Error(result.detail || 'Could not load dataset summary.');
    const dataset = result.dataset;
    const validation = dataset.validation || {};
    if(title) title.textContent = dataset.display_name || 'Training dataset';
    body.replaceChildren();
    const grid = document.createElement('dl');
    grid.className = 'trainingDatasetSummaryGrid';
    addDatasetSummaryRow(grid, 'Dataset ID', dataset.id);
    addDatasetSummaryRow(
      grid,
      'Status',
      dataset.available && validation.valid ? 'Ready' : dataset.available ? 'Validation failed' : 'Folder missing'
    );
    addDatasetSummaryRow(grid, 'Format', trainingDatasetFormatLabel(dataset));
    addDatasetSummaryRow(grid, 'Training backends', (validation.training_backends || []).map(value => value === 'yolo' ? 'YOLO' : 'Detectron').join(', '));
    const backendSet = new Set(validation.training_backends || []);
    const sidecarReady = Boolean(validation.yolo_sidecar?.valid);
    const nativeYoloReady = Boolean(validation.format_details?.yolo?.valid);
    addDatasetSummaryRow(grid, 'Detectron / COCO', backendSet.has('detectron') ? 'Ready' : 'Not available');
    addDatasetSummaryRow(
      grid,
      'YOLO',
      backendSet.has('yolo') ? `Ready (${sidecarReady ? 'shared sidecar layout' : nativeYoloReady ? 'native layout' : 'validated'})` : 'Not available'
    );
    addDatasetSummaryRow(grid, 'Storage path', dataset.storage_path);
    addDatasetSummaryRow(grid, 'Files', validation.file_count || 0);
    addDatasetSummaryRow(grid, 'Size', formatBytes(Number(validation.size_bytes || 0)));
    addDatasetSummaryRow(grid, 'Classes', (validation.classes || []).join(', '));
    addDatasetSummaryRow(grid, 'Validated', validation.validated_at ? new Date(validation.validated_at).toLocaleString() : '—');
    body.appendChild(grid);
    Object.entries(validation.splits || {}).forEach(([splitName, split]) => {
      const section = document.createElement('section');
      section.className = 'trainingDatasetSummarySection';
      const heading = document.createElement('h4');
      heading.textContent = `${splitName[0].toUpperCase()}${splitName.slice(1)} split`;
      const text = document.createElement('div');
      text.className = 'muted tiny';
      const details = [`${split.images || 0} images`];
      if(split.annotations != null) details.push(`${split.annotations} annotations`);
      if(split.label_files != null) details.push(`${split.label_files} label files`);
      if(split.missing_labels) details.push(`${split.missing_labels} missing labels`);
      if(split.unreadable_images) details.push(`${split.unreadable_images} unreadable images`);
      text.textContent = details.join(' · ');
      section.append(heading, text);
      body.appendChild(section);
    });
    const warnings = (validation.warnings || [])
      .filter(message => !(backendSet.has('yolo') && String(message).startsWith('YOLO compatibility:')))
      .map(message => {
        const text = String(message);
        if(/YOLO-compatible .* images without labels\.?$/i.test(text)){
          return `${text.replace(/\.$/, '')} (valid background images; add labels only if those images contain objects).`;
        }
        return text;
      });
    addDatasetMessages(body, validation.valid ? 'Compatibility notes' : 'Warnings', warnings, 'warn');
    addDatasetMessages(body, 'Errors', validation.errors, 'err');
  }catch(error){
    body.textContent = error.message || 'Could not load dataset summary.';
  }
}

function closeTrainingDatasetSummary(){
  const modal = document.getElementById('trainingDatasetSummaryModal');
  modal?.classList.remove('show');
  modal?.classList.add('hidden');
}

async function openTrainingUploadModal(){
  const modal = document.getElementById('trainingUploadModal');
  const name = document.getElementById('inpTrainingDatasetName');
  if(name && !name.value) name.value = makeStamp('training_data_');
  modal?.classList.add('show');
  modal?.classList.remove('hidden');
}

function setTrainingUploadError(message = ''){
  const box = document.getElementById('trainingUploadError');
  if(!box) return;
  box.textContent = message;
  box.hidden = !message;
}

function setTrainingUploadRunning(running){
  [
    'inpTrainingDatasetName',
    'fileTrainingZip', 'btnChooseTrainingFolder', 'btnStartTrainingUpload'
  ].forEach(id => {
    const element = document.getElementById(id);
    if(element) element.disabled = running;
  });
  const cancel = document.getElementById('btnCancelTrainingUpload');
  if(cancel){
    cancel.disabled = false;
    cancel.textContent = running ? 'Cancel upload' : 'Cancel';
  }
  const close = document.getElementById('btnCloseTrainingUpload');
  if(close) close.disabled = running;
}

function resetTrainingUpload(){
  setTrainingUploadRunning(false);
  setTrainingUploadError();
  ['fileTrainingZip'].forEach(id => {
    const input = document.getElementById(id);
    if(input) input.value = '';
  });
  selectedTrainingFolderFiles = [];
  selectedTrainingFolderName = '';
  const selection = document.getElementById('trainingUploadSelection');
  const progress = document.getElementById('trainingUploadProgress');
  const bar = document.getElementById('trainingUploadBar');
  if(selection) selection.textContent = '';
  if(progress) progress.hidden = true;
  if(bar) bar.style.width = '0%';
}

function closeTrainingUploadModal(){
  if(trainingUploadXhr) return;
  const modal = document.getElementById('trainingUploadModal');
  modal?.classList.remove('show');
  modal?.classList.add('hidden');
  resetTrainingUpload();
}

function trainingUploadErrorText(detail){
  if(typeof detail === 'string') return detail;
  if(!detail || typeof detail !== 'object') return 'Training dataset upload failed.';
  const messages = [];
  if(detail.message) messages.push(detail.message);
  const validation = detail.validation || {};
  if(Array.isArray(validation.errors)) messages.push(...validation.errors);
  return messages.join(' ') || 'Training dataset upload failed.';
}

function startTrainingDatasetUpload(){
  const zipInput = document.getElementById('fileTrainingZip');
  const zipFiles = Array.from(zipInput?.files || []);
  const folderFiles = selectedTrainingFolderFiles;
  const name = (document.getElementById('inpTrainingDatasetName')?.value || '').trim();
  setTrainingUploadError();
  if(!name){ setTrainingUploadError('Enter a dataset name.'); return; }
  if((zipFiles.length ? 1 : 0) + (folderFiles.length ? 1 : 0) !== 1){
    setTrainingUploadError('Choose either one ZIP archive or one dataset folder.');
    return;
  }
  const files = zipFiles.length
    ? zipFiles.map(file => ({file, relativePath: file.name}))
    : folderFiles;
  const body = new FormData();
  files.forEach(item => body.append('files', item.file, item.relativePath));
  body.append('display_name', name);
  body.append('dataset_format', 'unified');

  const progress = document.getElementById('trainingUploadProgress');
  const bar = document.getElementById('trainingUploadBar');
  const text = document.getElementById('trainingUploadProgressText');
  if(progress) progress.hidden = false;
  if(bar) bar.style.width = '0%';
  if(text) text.textContent = 'Uploading… 0%';
  setTrainingUploadRunning(true);

  const xhr = new XMLHttpRequest();
  trainingUploadXhr = xhr;
  xhr.open('POST', api.trainingDatasetUpload, true);
  xhr.upload.onprogress = event => {
    if(!event.lengthComputable) return;
    const percent = Math.round((event.loaded / event.total) * 100);
    if(bar) bar.style.width = `${percent}%`;
    if(text) text.textContent = `Uploading… ${percent}%`;
  };
  xhr.upload.onload = () => {
    if(bar) bar.style.width = '100%';
    if(text) text.textContent = 'Upload complete. Validating dataset…';
    const cancel = document.getElementById('btnCancelTrainingUpload');
    if(cancel){ cancel.disabled = true; cancel.textContent = 'Validating…'; }
  };
  xhr.onload = async () => {
    trainingUploadXhr = null;
    let result = {};
    try{ result = JSON.parse(xhr.responseText || '{}'); }catch(_){ }
    if(xhr.status >= 200 && xhr.status < 300 && result.ok){
      if(text) text.textContent = 'Upload and validation complete.';
      setTrainingUploadRunning(false);
      const modal = document.getElementById('trainingUploadModal');
      modal?.classList.remove('show');
      modal?.classList.add('hidden');
      resetTrainingUpload();
      preferredTrainingDatasetId = result.dataset?.id || '';
      await loadTrainingDatasets();
    }else{
      setTrainingUploadRunning(false);
      setTrainingUploadError(trainingUploadErrorText(result.detail));
      if(text) text.textContent = 'Upload rejected during validation.';
    }
  };
  xhr.onerror = () => {
    trainingUploadXhr = null;
    setTrainingUploadRunning(false);
    setTrainingUploadError('Network error while uploading training data.');
  };
  xhr.onabort = () => {
    trainingUploadXhr = null;
    setTrainingUploadRunning(false);
    const modal = document.getElementById('trainingUploadModal');
    modal?.classList.remove('show');
    modal?.classList.add('hidden');
    resetTrainingUpload();
  };
  xhr.send(body);
}

function updateTrainingUploadSelection(){
  const zipInput = document.getElementById('fileTrainingZip');
  const selection = document.getElementById('trainingUploadSelection');
  if(!selection) return;
  const zipCount = zipInput?.files?.length || 0;
  const folderCount = selectedTrainingFolderFiles.length;
  selection.textContent = zipCount
    ? `Selected ZIP: ${zipInput.files[0].name}`
    : folderCount ? `Selected folder: ${selectedTrainingFolderName || 'dataset'} · ${folderCount} files` : '';
}

async function collectTrainingDirectoryFiles(directoryHandle, prefix = directoryHandle.name){
  const files = [];
  for await(const [name, entry] of directoryHandle.entries()){
    const relativePath = `${prefix}/${name}`;
    if(entry.kind === 'file'){
      files.push({file: await entry.getFile(), relativePath});
    }else if(entry.kind === 'directory'){
      files.push(...await collectTrainingDirectoryFiles(entry, relativePath));
    }
  }
  return files;
}

async function chooseTrainingDatasetFolder(){
  const zipInput = document.getElementById('fileTrainingZip');
  if(typeof window.showDirectoryPicker !== 'function'){
    setTrainingUploadError('Folder selection requires Chrome or Edge on localhost/HTTPS. You can upload a ZIP archive instead.');
    return;
  }
  try{
    const directoryHandle = await window.showDirectoryPicker({mode:'read'});
    const button = document.getElementById('btnChooseTrainingFolder');
    if(button){ button.disabled = true; button.textContent = 'Reading folder…'; }
    const files = await collectTrainingDirectoryFiles(directoryHandle);
    selectedTrainingFolderFiles = files;
    selectedTrainingFolderName = directoryHandle.name;
    if(zipInput) zipInput.value = '';
    updateTrainingUploadSelection();
  }catch(error){
    if(error?.name !== 'AbortError'){
      setTrainingUploadError(`Could not read the selected folder: ${error.message || error}`);
    }
  }finally{
    const button = document.getElementById('btnChooseTrainingFolder');
    if(button){ button.disabled = false; button.textContent = 'Choose dataset folder'; }
  }
}
async function loadSessions(selectLatest=true){
  const js = await _fetchSessionsOnce();
  if(!js.ok) return;
  const sel1 = $("#selResults");
  const sel2 = $("#selMapSession");
  const previousResults = sel1.value;
  const previousMap = sel2.value;
  sel1.innerHTML = ""; sel2.innerHTML = "";
  testResultsCache = js.sessions || [];
  js.sessions.forEach(s=>{
    const nm = s.display_name || s.name.split("/").pop();
    for(const sel of [sel1, sel2]){
      const o = document.createElement("option");
      o.value = s.name;
      o.textContent = nm;
      o.title = `${nm} · ID: ${s.id || s.name}`;
      sel.appendChild(o);
    }
  });
  renderTestResults(testResultsCache);
  if(selectLatest && js.sessions.length){
    const latest = js.sessions[0].name;
    sel1.value = latest; sel2.value = latest;
  }else{
    if(js.sessions.some(session => session.name === previousResults)) sel1.value = previousResults;
    if(js.sessions.some(session => session.name === previousMap)) sel2.value = previousMap;
  }
}

function toggleBaseStyleDisabled(disabled){
  ["stColor","stWidth","stOpacity","fiColor","fiOpacity"].forEach(id=>{
    const el = document.getElementById(id);
    if (el){
      el.disabled = !!disabled;
      el.classList.toggle("disabled", !!disabled);
    }
  });
}



// ---------- upload modal ----------
function openUploadModal(){
  const modal = $("#uploadModal");
  if(!modal) return;
  modal.classList.add("show");
  modal.classList.remove("hidden");
}
function closeUploadModal(){
  const modal = $("#uploadModal");
  if(!modal) return;
  modal.classList.remove("show");
  modal.classList.add("hidden");
}
function updateFileCount(){
  const input = $("#filesTest");
  const countDisplay = document.getElementById("fileCountDisplay");
  if(!countDisplay) return;
  
  const count = input.files ? input.files.length : 0;
  if(count === 0){
    countDisplay.textContent = "";
  } else {
    countDisplay.textContent = `(${count} file${count !== 1 ? 's' : ''})`;
  }
}

function resetUploadProgress(){
  $("#testUploadBar").style.width = "0%";
  setText("#testUploadText","Uploading… 0%");
  setHidden($("#testUploadProgress"), true);
  // Reset file input and count display
  const filesInput = $("#filesTest");
  if(filesInput) filesInput.value = "";
  const countDisplay = document.getElementById("fileCountDisplay");
  if(countDisplay) countDisplay.textContent = "";
  const nameInput = document.getElementById("inpUploadName");
  if(nameInput) nameInput.value = "";
}

async function startUpload(){
  const input = $("#filesTest");
  if(!input.files || !input.files.length){
    warn("test","Choose at least one file (.zip or images).");
    return;
  }
  clearAlerts("test");
  closeUploadModal();  // Close modal immediately to show full progress bar
  setHidden($("#testUploadProgress"), false);
  $("#testUploadBar").style.width = "0%";
  setText("#testUploadText","Uploading… 0%");

  const fd = new FormData();
  for(const f of input.files){ fd.append("files", f); }
  const uploadNameEl = document.getElementById("inpUploadName");
  const uploadName = (uploadNameEl?.value || "").trim();
  if (uploadName) fd.append("result_name", uploadName);  // optional name

  await new Promise((resolve, reject)=>{
    const xhr = new XMLHttpRequest();
    xhr.open("POST", api.upload, true);
    xhr.upload.onprogress = (e)=>{
      if(e.lengthComputable){
        const pct = Math.round((e.loaded / e.total)*100);
        $("#testUploadBar").style.width = `${pct}%`;
        setText("#testUploadText", `Uploading… ${pct}%`);
      }
    };
    xhr.onload = ()=> {
      try{
        const js = JSON.parse(xhr.responseText);
        if(js.ok){
          ok("test", `Upload complete. Created: ${js.created.join(", ")}`);
          setHidden($("#testUploadProgress"), true);
          loadDatasets();
          resetUploadProgress();
          resolve();
        }else{
          err("test","Upload failed.");
          setHidden($("#testUploadProgress"), true);
          reject(new Error("upload failed"));
        }
      }catch(ex){
        err("test", "Upload error.");
        setHidden($("#testUploadProgress"), true);
        reject(ex);
      }
    };
    xhr.onerror = ()=>{ err("test","Network error"); setHidden($("#testUploadProgress"), true); reject(new Error("xhr")); };
    xhr.send(fd);
  });
}

// ---------- training ----------
async function startTraining(){
  clearAlerts("train"); wireAlertClose();
  resetLogPane("#trainMiniLog");
  const chkThermalTrain = document.getElementById("chkUseThermalTrain");
  const thermalEnabled = (typeof window.isThermalExtractionEnabled === "function")
    ? window.isThermalExtractionEnabled()
    : true;
  if (!thermalEnabled && chkThermalTrain && chkThermalTrain.checked){
    warn("train","Thermal data extraction is disabled on this server. Set PVRT_ENABLE_THERMAL=1 to train with thermal inputs.");
    return;
  }
  const useThermal = thermalEnabled && chkThermalTrain ? chkThermalTrain.checked : false;
  const iters = parseInt($("#inpIters").value || "500", 10);
  const lr = parseFloat($("#inpLR").value || "0.002");
  const batch = parseInt($("#inpBatch").value || "4", 10);
  const modelName = (document.getElementById("inpModelName")?.value || "").trim() || makeStamp();
  const trainingDatasetId = document.getElementById('selTrainingDataset')?.value || '';
  if(!trainingDatasetId){
    warn('train', 'Select validated training data for the chosen backend.');
    return;
  }

  // Check if model name already exists
  let clearExisting = false;
  try {
    const existingModel = modelsCache[modelName];
    if (existingModel) {
      const shouldClear = confirm(`A model named "${modelName}" already exists.\n\nDo you want to delete the previous model and start fresh?`);
      if (!shouldClear) {
        warn("train", "Training canceled.");
        return;
      }
      clearExisting = true;
    }
  } catch (_) { }

  setHidden($("#spinTrain"), false);
  setText("#trainStatus","Submitting training job…");

  const fd = new FormData();
  fd.append("use_thermal", useThermal ? "true" : "false");
  fd.append("max_iter", String(iters));
  fd.append("base_lr", String(lr));
  fd.append("ims_per_batch", String(batch));
  fd.append("model_name", String(modelName));
  fd.append("clear_existing", clearExisting ? "true" : "false");
  fd.append('dataset_id', trainingDatasetId);
  const backend = document.getElementById('selBackendTrain')?.value || getSelectedBackend();
  const task = document.getElementById('selTrainingTask')?.value || 'detect';
  fd.append("backend", backend);
  fd.append('task', task);
  fd.append('model_type', task === 'segment' ? 'maskrcnn' : 'fasterrcnn');
  fd.append('yolo_seg', task === 'segment' ? 'true' : 'false');
  if(backend === 'yolo'){
    const yo = getYoloOptions();
    fd.append('yolo_family', yo.family);
    fd.append('yolo_size', yo.size);
  }


  try{
    const res = await fetch(api.train, { method:"POST", body:fd });
    const js = await res.json().catch(()=>({}));
    if(res.ok && js.ok){
      ok("train","Training completed.");
      setText("#trainStatus","Training completed.");
      await loadTrainedModels();
    }else{
      const detail = typeof js.detail === 'string' ? js.detail : js.detail?.message;
      err("train", detail || "Failed to start training.");
      setText('#trainStatus', 'Training could not start.');
    }
  }catch(error){
    err('train', error.message || 'Network error while starting training.');
    setText('#trainStatus', 'Training request failed.');
  }finally{
    setHidden($("#spinTrain"), true);
  }
}
async function cancelTraining(){
  try{
    const fd = new FormData();
    fd.append("job","train");
    await fetch(api.cancel, { method:"POST", body:fd });
    warn("train","Cancel requested. Training stops at next safe point.");
    wireAlertClose();
  }catch(_){}
}

// ---------- test run ----------
async function runTest(){
  if(testAbort){
    warn("test", "A test is already running. Wait for it to finish or cancel it first.");
    return;
  }
  clearAlerts("test"); wireAlertClose();
  resetLogPane("#testMiniLog");
  const ds = getSelectedDataset();
  if(!ds){
    warn("test","Please select a dataset.");
    return;
  }
  const model = getSelectedModel();
  const wantsAccurate = document.getElementById("chkAccurateLocations")?.checked;
  let accurateMode = null;
  let optimizationProject = "";
  
  if(wantsAccurate){
    if(document.getElementById("radAccurateColmap")?.checked){
      accurateMode = "colmap";
    }else if(document.getElementById("radAccurateOptical")?.checked){
      accurateMode = "optical";
      optimizationProject = document.getElementById("selUseOptimizationFrom")?.value || "";
    }
    
    if(!accurateMode){
      warn("test","Choose COLMAP or optical sync for accurate poses.");
      return;
    }
    
    if(accurateMode === "colmap"){
      const state = getColmapState(ds);
      if(!state || !state.ready){
        warn("test","Run Optimize Locations and finish before enabling COLMAP mode.");
        return;
      }
    }else if(accurateMode === "optical"){
      if(!optimizationProject){
        warn("test","Select an optimized project for optical sync.");
        return;
      }
    }
  }
  // Decide channels based solely on the selected model's metadata.
  // The frontend will send the model's expected channel_count to the server so the backend
  // can decide whether to decode/use the thermal band. If model metadata is missing,
  // default to 3 channels (RGB) and therefore no thermal decoding.
  const selectedModelName = model;
  const mmeta = selectedModelName ? modelsCache[selectedModelName] : null;
  let modelChannelCount = (mmeta && mmeta.channel_count) ? Number(mmeta.channel_count) : 3;
  // All models use 3 channels: either 3-channel RGB or 3-channel thermal (decoded)
  if (modelChannelCount !== 3) modelChannelCount = 3;
  // use_thermal true when a 3-channel model was trained with thermal
  // (thermal_only or thermal_used flags set).
  const modelRequiresThermal = (modelChannelCount === 3 && (!!(mmeta && mmeta.thermal_only) || !!(mmeta && mmeta.thermal_used)));
  const thermalEnabled = (typeof window.isThermalExtractionEnabled === "function")
    ? window.isThermalExtractionEnabled()
    : true;
  if (!thermalEnabled && modelRequiresThermal){
    warn("test","Thermal data extraction is disabled on this server. Select an RGB-only model or set PVRT_ENABLE_THERMAL=1 to run thermal workflows.");
    setHidden($("#spinTest"), true);
    return;
  }
  const useThermal = thermalEnabled && modelRequiresThermal;
  const resultName = (document.getElementById("inpResultName")?.value || "").trim() || makeStamp();
  
  // Check if result name already exists
  let clearExisting = false;
  try {
    const existingSessions = Array.from(document.querySelectorAll("#selResults option")).map(o => o.value);
    const sessionExists = existingSessions.some(s => s.includes(resultName));
    if (sessionExists) {
      const shouldClear = confirm(`A result named "${resultName}" already exists.\n\nDo you want to delete the previous result and start fresh?`);
      if (!shouldClear) {
        warn("test", "Testing canceled.");
        setHidden($("#spinTest"), true);
        return;
      }
      clearExisting = true;
    }
  } catch (_) { }
  
  const testThreshold = (document.getElementById("testThreshold")?.value);
  const targetSurfaceHeightInput = document.getElementById("testTargetSurfaceHeight");
  const targetSurfaceHeightRaw = targetSurfaceHeightInput?.value?.trim() ?? "4";
  const targetSurfaceHeight = Number(targetSurfaceHeightRaw);
  if(!targetSurfaceHeightRaw || !Number.isFinite(targetSurfaceHeight) || targetSurfaceHeight < 0){
    warn("test", "Target surface height must be zero or a positive number in metres.");
    setHidden($("#spinTest"), true);
    return;
  }
  const rowAlignmentError = window.TestRowAlignment?.validate?.();
  if(rowAlignmentError){
    warn("test", rowAlignmentError);
    setHidden($("#spinTest"), true);
    return;
  }

  setHidden($("#spinTest"), false);
  const runTestButton = document.getElementById('btnRunTest');
  if(runTestButton) runTestButton.disabled = true;
  setText("#testStatus","Preparing test data…");
  const preparingLine = `[test] Preparing dataset "${ds}" for model inference…`;
  appendMiniLog("#testMiniLog", preparingLine);
  appendLog(preparingLine);

  const fd = new FormData();
  fd.append("dataset", ds);
  const projectId = getProjectIdFromURL();
  if(projectId) fd.append("project_id", projectId);
  if(model) fd.append("model", model);
  fd.append("use_thermal", useThermal ? "true":"false");
  // inform backend of the expected model channel count so it can prepare inputs correctly
  // inform backend of the expected model channel count so it can prepare inputs correctly
  fd.append('channel_count', String(modelChannelCount));
  // Note: frontend no longer offers the "extract thermal as RGB" option; backend will
  // only attempt decoding when the selected model declares RGB+thermal input.
  fd.append("result_name", resultName);
  fd.append("clear_existing", clearExisting ? "true" : "false");
  fd.append("test_threshold", testThreshold);
  fd.append("target_surface_height_m", String(targetSurfaceHeight));
  window.TestRowAlignment?.appendFormData?.(fd);
  const correctLensDistortion = document.getElementById("chkUndistortThermal")?.checked === true;
  fd.append("undistort_thermal", correctLensDistortion ? "true" : "false");
  fd.append(
    "export_undistorted_images",
    correctLensDistortion && document.getElementById("chkExportUndistortedImages")?.checked ? "true" : "false"
  );
  if(accurateMode === "colmap") fd.append("accurate_locations", "true");
  if(accurateMode === "optical" && optimizationProject) fd.append("optimization_project", optimizationProject);
  const createMosaic = document.getElementById('chkMosaicImages')?.checked === true;
  const inferenceSource = createMosaic
    ? (document.querySelector('input[name="inferenceSource"]:checked')?.value || 'mosaic')
    : 'individual';
  fd.append('inference_source', inferenceSource);
  fd.append('create_mosaic', createMosaic ? 'true' : 'false');
  if(createMosaic){
    fd.append("mosaic_enabled", "true");
    fd.append(
      "refine_mosaic_alignment",
      document.getElementById("chkRefineMosaicAlignment")?.checked ? "true" : "false"
    );
  }
  const backend = getSelectedBackend();
  fd.append('backend', backend);
  if(backend === 'yolo'){
    const yo = getYoloOptions();
    fd.append('yolo_family', yo.family);
  }

  try{
    testAbort = new AbortController();
    const res = await fetch(api.testRun, { method:"POST", body: fd, signal: testAbort.signal });
    const js = await res.json();
    const totalPreds = Array.isArray(js.manifest)
      ? js.manifest.reduce((sum, it) => sum + (it.n || 0), 0)
      : 0;

    if(!res.ok || !js.ok){
      const detail = typeof js.detail === 'string' ? js.detail : js.detail?.message;
      throw new Error(detail || "Test failed.");
    }

    // If backend reports which channel configuration it actually used, show it briefly
    try{
      if (js.used_channel_count != null){
        const uc = Number(js.used_channel_count);
        // Show whether thermal or RGB was used
        let msg = 'Run used: 3ch';
        if (js.thermal_used || js.final_mode === 'thermal') msg += ' (thermal)';
        else msg += ' (RGB)';
        ok("test", msg);
      }
    }catch(_){ }

    // The server-side run is complete; keep progress visible while its result is
    // loaded into the map, result grid, and session selectors.
    currentSession = js.session;
    setText("#testStatus", "Loading completed test result…");
    const loadingLine = "[test] Processing complete. Loading map and result assets…";
    appendMiniLog("#testMiniLog", loadingLine);
    appendLog(loadingLine);

    await applySessionToMap(currentSession);
    renderResultsGrid(js.manifest && js.manifest.length ? js.manifest : pairThumbs(js.assets));
    loadResultsInfo(currentSession);
    
    await loadSessions(true);
    $("#selResults").value = currentSession;
    $("#selMapSession").value = currentSession;
    ok("test", "Testing completed.");
    setText("#testStatus", `Test complete. ${totalPreds} predictions.`);
    switchToTab("tab-results");
  }catch(ex){
    if(ex.name === "AbortError"){
      warn("test","Test canceled.");
    }else{
      err("test", String(ex));
    }
    await loadSessions(false).catch(()=>{});
  }finally{
    setHidden($("#spinTest"), true);
    testAbort = null;
    if(typeof updateAccurateUI === 'function') updateAccurateUI();
    else if(runTestButton) runTestButton.disabled = false;
  }
}
function cancelTest(){ if(testAbort){ testAbort.abort(); } }

// ---------- results ----------
async function showResultsForSelected(){
  const session = $("#selResults").value;
  // console.log(session)
  if(!session) return;
  currentSession = session;
  const res = await fetch(`${api.sessionSummary}?session=${encodeURIComponent(session)}`);
  const js = await res.json();
  if(!js.ok) return;
  lastLoadedSessionSummary = js || null;
  rotatedImagesLookup = null;
  // console.log(js)
  renderResultsGrid(js.manifest && js.manifest.length ? js.manifest : pairThumbs(js.assets));
  loadResultsInfo(currentSession);
}

function pairThumbs(assets){
  const mapThumb = new Map((assets?.thumbs || []).map(u => [u.split("/").pop(), u]));
  const overlays = Array.isArray(assets?.overlays) ? assets.overlays : [];
  const out = [];
  for(const ov of overlays){
    const fn = ov.split("/").pop();
    // Use overlays as-is (they contain annotations/predictions)
    // Do not replace with rotated_images (those are for inference only)
    const thumbUrl = mapThumb.get(fn) || ov;
    out.push({ file: fn, overlay: ov, thumb: thumbUrl });
  }
  return out;
}

function normalizeImageStem(name){
  if(!name) return null;
  const base = extractAssetBasename(name);
  if(!base) return null;
  return base.replace(/\.[^.]+$/, "").toLowerCase();
}

function decodeUrlComponentSafe(value){
  try { return decodeURIComponent(String(value)); }
  catch(_) { return String(value); }
}

function decodeUrlComponentDeep(value, maxRounds = 3){
  let out = String(value ?? '');
  for (let i = 0; i < maxRounds; i++){
    const dec = decodeUrlComponentSafe(out);
    if (dec === out) break;
    out = dec;
  }
  return out;
}

function toProjectFileUrlFromValue(value){
  if (!value) return null;
  const raw = String(value).split('?')[0];
  const decoded = decodeUrlComponentDeep(raw);
  if (decoded.startsWith('/')) {
    return `/api/project_file/${encodeURIComponent(decoded)}`;
  }
  return null;
}

function extractAssetBasename(value){
  if (!value) return "";
  let raw = String(value).split('?')[0];

  if (raw.includes('/api/project_file/')) {
    const marker = '/api/project_file/';
    const idx = raw.indexOf(marker);
    const encodedAbs = idx >= 0 ? raw.slice(idx + marker.length) : raw;
    const absPath = decodeUrlComponentSafe(encodedAbs);
    const base = absPath.split(/[\\/]/).pop();
    if (base) return base;
  }

  raw = decodeUrlComponentDeep(raw);
  const tail = raw.split('/').pop() || raw;
  const base = tail.split(/[\\/]/).pop() || tail;
  return base;
}

function buildNameVariantSet(...values){
  const tokens = new Set();
  const addVariant = (val) => {
    if (!val && val !== 0) return;
    const norm = String(val).trim().toLowerCase();
    if (norm) tokens.add(norm);
  };
  const stripExt = (val) => (typeof val === 'string') ? val.replace(/\.[^.]+$/, '') : '';

  for (const value of values){
    if (value == null) continue;
    let raw = String(value);
    if (!raw) continue;
    raw = raw.split('?')[0];
    addVariant(raw);
    const decoded = decodeUrlComponentDeep(raw);
    addVariant(decoded);
    const base = extractAssetBasename(raw);
    addVariant(base);
    const decodedBase = decodeUrlComponentDeep(base);
    addVariant(decodedBase);
    addVariant(stripExt(base));
    addVariant(stripExt(decodedBase));
  }

  tokens.delete('');
  return tokens;
}

function resolveFeatureOverlayUrl(featureProps, overlayByName, sessionRoot){
  const preparedImage = featureProps?.prepared_image;
  if (typeof preparedImage === 'string' && preparedImage) {
    if (preparedImage.startsWith('/api/project_file/')) return preparedImage;
    const projected = toProjectFileUrlFromValue(preparedImage);
    if (projected) return projected;
  }
  const overlayProp = featureProps?.overlay;
  if (typeof overlayProp === 'string' && overlayProp) {
    if (overlayProp.startsWith('/api/project_file/')) return overlayProp;
    const projected = toProjectFileUrlFromValue(overlayProp);
    if (projected) return projected;
  }

  const srcRaw = featureProps?.src || featureProps?.image || featureProps?.file || featureProps?.name;
  const srcFile = extractAssetBasename(srcRaw);
  if (!srcFile) return null;

  const stem = srcFile.replace(/\.[^.]+$/, '');
  const overlayName = `${stem}.png`;
  const fromSummary = overlayByName.get(overlayName);
  if (fromSummary) return fromSummary;

  const rotated = getRotatedImageUrl(srcFile);
  if (rotated) return rotated;

  const canUseSessionRoot = sessionRoot && !sessionRoot.includes('/api/project_file/');
  return canUseSessionRoot ? `${sessionRoot}overlays/${encodeURIComponent(overlayName)}` : null;
}

function ensureRotatedLookup(){
  if(rotatedImagesLookup !== null) return rotatedImagesLookup;
  rotatedImagesLookup = new Map();
  const rotatedList = lastLoadedSessionSummary?.assets?.rotated_images;
  if(Array.isArray(rotatedList)){
    rotatedList.forEach(url => {
      const stem = normalizeImageStem(url);
      if(stem) rotatedImagesLookup.set(stem, url);
    });
  }
  return rotatedImagesLookup;
}

function getRotatedImageUrl(name){
  const lookup = ensureRotatedLookup();
  if(!lookup.size) return null;
  const stem = normalizeImageStem(name);
  if(!stem) return null;
  return lookup.get(stem) || null;
}

function preferRotatedOverlays(manifest){
  // Use overlays as-is (they contain annotations/predictions)
  // Do not replace with rotated_images (those are for inference only)
  return manifest;
}
function renderResultsGrid(manifest){
  currentManifest = manifest;  // Store for filter re-rendering
  const grid = $("#resultsGrid");
  grid.innerHTML = "";
  
  if(!manifest || !manifest.length){
    grid.innerHTML = `<div class="muted">No overlays generated.</div>`;
  }

  manifest = preferRotatedOverlays(manifest);
  const showOnlyDetections = document.getElementById('chkShowOnlyDetections')?.checked || false;
  
  // Build filtered list for lightbox navigation
  const filteredItems = showOnlyDetections
    ? manifest.filter(item => item.n && item.n > 0)
    : manifest;

  manifest.forEach((item, idx) => {
    const div = document.createElement("div");
    div.className = "thumb";

    // Add detection indicator badge if detections exist
    const detectionBadge = (item.n && item.n > 0)
      ? `<div class="detection-badge">${item.n}</div>`
      : '';
    const correctionLabels = {
      corrected: 'Lens corrected',
      skipped_already_corrected: 'Already corrected',
      skipped_below_threshold: 'Correction &lt;2 px',
    };
    const correctionLabel = correctionLabels[item.lens_correction_status] || '';
    const correctionBadge = correctionLabel
      ? `<div class="correction-badge" title="This overlay uses the prepared inference image">${correctionLabel}</div>`
      : '';

    div.innerHTML = `
      <img src="${item.thumb}" alt="${item.file}">
      <div class="meta" title="${item.file}">${item.file}</div>
      ${detectionBadge}
      ${correctionBadge}
    `;

    // Hide if filter is on and no detections
    if (showOnlyDetections && (!item.n || item.n === 0)) {
      div.classList.add('hidden-by-filter');
    }

    div.addEventListener("click", () => {
      // Find the index of this item in the filtered list
      const filteredIdx = filteredItems.indexOf(item);
      _openLightboxWithGallery(filteredItems, filteredIdx);
    });
    grid.appendChild(div);
  });
}

function _applyDetectionFilter(){
  if(!currentManifest) return;
  renderResultsGrid(currentManifest);
}

// ---------- map ----------
function initMap(){
  const street = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", { maxZoom: 22, attribution: "&copy; OpenStreetMap" });
  const sat = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", { maxZoom: 22, attribution: "&copy; Esri" });

  MAP = L.map("map", { layers: [street] });
  baseLayers = { "Street": street, "Satellite": sat };
  L.control.layers(baseLayers, {}, { position: "topleft" }).addTo(MAP);
  window.addStandardMeasureControl?.(MAP);
  MAP.setView([0,0], 2);

  // Create custom panes for tile layers
  // Session orthophoto pane - should be below overlay layers
  if (!MAP.getPane('sessionOrthophotoPane')) {
    const pane = MAP.createPane('sessionOrthophotoPane');
    pane.style.zIndex = 250; // Above tile pane (200) but below overlay pane (400)
  }
  // Overlay TIF pane - should be above base layers but not too high to hide other overlays
  if (!MAP.getPane('overlayTifPane')) {
    const pane = MAP.createPane('overlayTifPane');
    pane.style.zIndex = 210; // Just above tile pane (200)
  }
  // Predictions must remain above reference polygons so they stay visible and clickable.
  if (!MAP.getPane('predictionsPane')) {
    const pane = MAP.createPane('predictionsPane');
    pane.style.zIndex = 640; // Below tooltips/popups, above vector and marker panes.
  }

  if (!imagesLayerGroup) imagesLayerGroup = L.layerGroup().addTo(MAP);
  imageMarkersLayer = L.layerGroup().addTo(MAP);
  // overlayRegistry["Image markers"] = { layer: imageMarkersLayer, type: "markers" };
  renderLegend();
  refreshLayersPanel();
}

function clearImageMarkers(){
  if (!imageMarkersLayer) imageMarkersLayer = L.layerGroup().addTo(MAP);
  imageMarkersLayer.clearLayers();
  imageMarkers = new Map();
}


function computeLayerBounds(layer){
  // Works for GeoJSON/FeatureGroup/LayerGroup/Marker/Polyline/Polygon
  if (layer && typeof layer.getBounds === 'function') {
    try { return layer.getBounds(); } catch {}
  }
  let bounds = null;
  if (layer && typeof layer.eachLayer === 'function') {
    layer.eachLayer(child => {
      const b = computeLayerBounds(child);
      if (!b) return;
      bounds = bounds ? bounds.extend(b) : b;
    });
    return bounds;
  }
  if (layer && typeof layer.getLatLng === 'function') {
    const ll = layer.getLatLng();
    return L.latLngBounds(ll, ll);
  }
  if (layer && typeof layer.getLatLngs === 'function') {
    const lls = layer.getLatLngs().flat(Infinity);
    if (lls.length) return L.latLngBounds(lls);
  }
  return bounds;
}

function toHex(c){
  if (!c) return '#000000';
  if (c[0] === '#') return c;
  // Convert named/rgb(...) to hex
  const ctx = document.createElement('canvas').getContext('2d');
  ctx.fillStyle = c;
  const rgb = ctx.fillStyle;              // normalized css color
  const m = rgb.match(/\d+/g);
  if (!m) return '#000000';
  return '#' + m.slice(0,3).map(x => (+x).toString(16).padStart(2,'0')).join('');
}

function uniqueValuesFromGJ(gj, prop){
  const s = new Set();
  (gj.features||[]).forEach(f=>{
    if (f?.properties?.[prop] != null) s.add(String(f.properties[prop]));
  });
  return Array.from(s).sort((a,b)=> a.localeCompare(b));
}

function toHex(c){
  if (!c) return '#000000';
  if (c[0] === '#') return c;
  const ctx = document.createElement('canvas').getContext('2d');
  ctx.fillStyle = c;
  const rgb = ctx.fillStyle;
  const m = rgb.match(/\d+/g);
  if (!m) return '#000000';
  return '#' + m.slice(0,3).map(x => (+x).toString(16).padStart(2,'0')).join('');
}




function rebuildCategoryEditors(prop, layerKey = "Predictions"){
  const rec  = overlayRegistry[layerKey];
  const host = document.getElementById("stCatList");
  if (!rec || !rec.data || !host) return;

  const CATEGORY_NONE = '__none__';
  const ALL_KEY = '__ALL__';

  // Build list of unique values for the chosen prop
  let values = [];
  if (prop && prop !== CATEGORY_NONE){
    values = uniqueValuesFromGJ(rec.data, prop);
  }

  // Auto-fallback to "None" if invalid or too many categories
  const useNone = (!prop || prop === CATEGORY_NONE || values.length === 0 || values.length > 19);
  host.innerHTML = "";

  const base = rec.style || { color:"#ff5722", weight:1, opacity:1, fillColor:"#ff5722", fillOpacity:0.25 };
  const prev = (rec.categorical && (rec.categorical.prop === prop || (useNone && rec.categorical.prop === CATEGORY_NONE)))
               ? (rec.categorical.classes || {}) : {};
  const classes = {};

  if (useNone){
    // Single "All features" row acts as whole-layer style
    const s = prev[ALL_KEY] || {
      strokeColor: base.color, strokeWidth: base.weight, strokeOpacity: base.opacity,
      fillColor: base.fillColor, fillOpacity: base.fillOpacity,
    };
    classes[ALL_KEY] = {...s};

    const row = document.createElement("div");
    row.className = "catRow";
    row.innerHTML = `
      <div class="catName">All features</div>
      <div class="catControls">
        <div class="ctrl"><label>Stroke</label><input type="color"  class="cStroke" value="${toHex(s.strokeColor)}"></div>
        <div class="ctrl"><label>Width</label> <input type="number" class="cW"  min="0" step="0.5" value="${s.strokeWidth}"></div>
        <div class="ctrl"><label>Opacity</label><input type="number" class="cSO" min="0" max="1" step="0.05" value="${s.strokeOpacity}"></div>
        <div class="ctrl"><label>Fill</label>  <input type="color"  class="cFill"  value="${toHex(s.fillColor)}"></div>
        <div class="ctrl"><label>Opacity</label><input type="number" class="cFO" min="0" max="1" step="0.05" value="${s.fillOpacity}"></div>
      </div>`;
    host.appendChild(row);

    const iStroke = row.querySelector(".cStroke");
    const iW      = row.querySelector(".cW");
    const iSO     = row.querySelector(".cSO");
    const iFill   = row.querySelector(".cFill");
    const iFO     = row.querySelector(".cFO");

    const apply = ()=>{
      classes[ALL_KEY] = {
        strokeColor: iStroke.value,
        strokeWidth: parseFloat(iW.value || "1"),
        strokeOpacity: Math.max(0, Math.min(1, parseFloat(iSO.value || "1"))),
        fillColor: iFill.value,
        fillOpacity: Math.max(0, Math.min(1, parseFloat(iFO.value || "0.25"))),
      };
      rec.categorical = { prop: CATEGORY_NONE, classes, values:[ALL_KEY] };
      if (rec.layer?.setStyle) rec.layer.setStyle(f => styleForCategoricalFeature(f, rec.style || {}, rec.categorical));
      renderLegend();
    };
    [iStroke, iW, iSO, iFill, iFO].forEach(inp => inp.addEventListener("input", apply));

    // commit
    rec.categorical = { prop: CATEGORY_NONE, classes, values:[ALL_KEY] };
    if (rec.layer?.setStyle) rec.layer.setStyle(f => styleForCategoricalFeature(f, rec.style || {}, rec.categorical));
    renderLegend();
    return;
  }

  // Real categories (<=19)
  values.forEach((v)=>{
    const s = prev[v] || {
      strokeColor: base.color, strokeWidth: base.weight, strokeOpacity: base.opacity,
      fillColor: base.fillColor, fillOpacity: base.fillOpacity,
    };
    classes[v] = {...s};

    const row = document.createElement("div");
    row.className = "catRow";
    row.innerHTML = `
      <div class="catName">${escapeHtml(v)}</div>
      <div class="catControls">
        <div class="ctrl"><label>Stroke</label><input type="color"  class="cStroke" value="${toHex(s.strokeColor)}"></div>
        <div class="ctrl"><label>Width</label> <input type="number" class="cW"  min="0" step="0.5" value="${s.strokeWidth}"></div>
        <div class="ctrl"><label>Opacity</label><input type="number" class="cSO" min="0" max="1" step="0.05" value="${s.strokeOpacity}"></div>
        <div class="ctrl"><label>Fill</label>  <input type="color"  class="cFill"  value="${toHex(s.fillColor)}"></div>
        <div class="ctrl"><label>Opacity</label><input type="number" class="cFO" min="0" max="1" step="0.05" value="${s.fillOpacity}"></div>
      </div>`;
    host.appendChild(row);

    const iStroke = row.querySelector(".cStroke");
    const iW      = row.querySelector(".cW");
    const iSO     = row.querySelector(".cSO");
    const iFill   = row.querySelector(".cFill");
    const iFO     = row.querySelector(".cFO");

    const apply = ()=>{
      classes[v] = {
        strokeColor: iStroke.value,
        strokeWidth: parseFloat(iW.value || "1"),
        strokeOpacity: Math.max(0, Math.min(1, parseFloat(iSO.value || "1"))),
        fillColor: iFill.value,
        fillOpacity: Math.max(0, Math.min(1, parseFloat(iFO.value || "0.25"))),
      };
      rec.categorical = { prop, classes, values };
      if (rec.layer?.setStyle) rec.layer.setStyle(f => styleForCategoricalFeature(f, rec.style || {}, rec.categorical));
      renderLegend();
    };
    [iStroke, iW, iSO, iFill, iFO].forEach(inp => inp.addEventListener("input", apply));
  });

  rec.categorical = { prop, classes, values };
  if (rec.layer?.setStyle) rec.layer.setStyle(f => styleForCategoricalFeature(f, rec.style || {}, rec.categorical));
  renderLegend();
}




function propsTable(props = {}) {
  const rows = Object.entries(props).map(([k, v]) => {
    const val = (v == null) ? "" : (typeof v === "string" ? v : JSON.stringify(v));
    return `<tr><th>${escapeHtml(k)}</th><td>${escapeHtml(val)}</td></tr>`;
  }).join("");
  return `<table class="propTable">${rows}</table>`;
}

function featurePopupHTML(f) {
  const props = f?.properties || {};
  const previewURL = props.overlay || props.thumb || props.url;
  const title = props.image || props.file || props.name || "Feature";
  const toggleBtn = popupImageToggleButtonHTML(props);
  return `
    <div class="popupWrap">
      <div class="popupTitle"><b>${escapeHtml(title)}</b></div>
      ${previewURL ? `<div class="popupPreview" style="margin:.5rem 0">
        <img src="${previewURL}" style="max-width:240px;max-height:180px;border:1px solid var(--border);border-radius:6px;">
      </div>` : ``}
      ${toggleBtn ? `<div class="popupActions" style="margin:0.25rem 0 0.5rem;">
        ${toggleBtn}
      </div>` : ``}
      ${propsTable(props)}
    </div>
  `;
}

function popupImageToggleButtonHTML(props){
  const rec = findImageRecordForFeature(props);
  if (!rec) return "";
  const label = rec.on ? "Remove image" : "View image";
  const state = rec.on ? "on" : "off";
  const eyeSvg = `<svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true">
    <path d="M2 12s4-8 10-8 10 8 10 8-4 8-10 8S2 12 2 12z" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" />
    <circle cx="12" cy="12" r="3" fill="none" stroke="currentColor" stroke-width="1.6" />
  </svg>`;
  return `<button type="button" class="popupImageToggle ${state}" data-image-id="${escapeHtml(rec.id)}" data-state="${state}" title="${escapeHtml(label)}" aria-label="${escapeHtml(label)}">${eyeSvg}<span class="popupImageToggleText">${escapeHtml(label)}</span></button>`;
}

function findImageRecordForFeature(props){
  if (!props) return null;
  const targetTokens = buildNameVariantSet(props.image, props.file, props.name, props.src);
  if (!targetTokens.size) return null;
  for (const rec of imageCatalog){
    if (!rec) continue;
    if (!rec.matchTokens || !(rec.matchTokens instanceof Set)){
      rec.matchTokens = buildNameVariantSet(rec.id, rec.name);
    }
    for (const token of rec.matchTokens){
      if (targetTokens.has(token)) return rec;
    }
  }
  return null;
}



let anomaliesProp = 'class_name';  // current property to color by

async function loadGeoJSON(url){
  const res = await fetch(url);
  const gj = await res.json();

  const base = overlayRegistry["Predictions"]?.style || {
    color: "#ff5722", weight: 1, opacity: 1,
    fillColor: "#ff5722", fillOpacity: 0.25
  };

  // remove the previous Predictions layer so we do not create duplicates
  try{
    const prev = overlayRegistry["Predictions"];
    if (prev && prev.layer){ try { MAP.removeLayer(prev.layer); } catch(_){} }
  }catch(_){ }

  const layer = L.geoJSON(gj, {
    pane: "predictionsPane",
    style: (f)=> styleForAnomalyFeature(f, base),
    pointToLayer: (f, latlng) => L.circleMarker(latlng, {
      pane: "predictionsPane",
      radius: 4,
      color: base.color,
      fillColor: base.fillColor,
      fillOpacity: 0.8,
    }),
    onEachFeature: (feature, layer) => {
      window.addGeoJsonHoverHighlight?.(feature, layer);
      // Resolve the associated image when the popup opens. The image catalog is
      // loaded after predictions, so building this HTML during layer creation
      // permanently omitted the View image action.
      try { layer.bindPopup(() => featurePopupHTML(feature)); } catch(_) {}
    }
  });

  overlayRegistry["Predictions"] = { layer, type: "geojson", style: base, data: gj, categorical: overlayRegistry["Predictions"]?.categorical || null };
  refreshLayersPanel();
  renderLegend();
  // Don't auto-fit bounds here - let applySessionToMap handle it after all layers loaded
}

function styleForCategoricalFeature(f, fallback, categorical){
  const cat = categorical;
  if (!cat) return fallback || { color:"#ff5722", weight:1, opacity:1, fillColor:"#ff5722", fillOpacity:0.25 };

  const CATEGORY_NONE = '__none__';
  const ALL_KEY = '__ALL__';

  if (cat.prop === CATEGORY_NONE){
    const s = cat.classes?.[ALL_KEY];
    if (s){
      return {
        color: s.strokeColor ?? "#ff5722",
        weight: s.strokeWidth ?? 1,
        opacity: s.strokeOpacity ?? 1,
        fillColor: s.fillColor ?? s.strokeColor ?? "#ff5722",
        fillOpacity: s.fillOpacity ?? 0.25,
      };
    }
    return fallback || { color:"#ff5722", weight:1, opacity:1, fillColor:"#ff5722", fillOpacity:0.25 };
  }

  const key = String(f?.properties?.[cat.prop] ?? "");
  const s = cat.classes?.[key];
  if (s){
    return {
      color: s.strokeColor ?? "#ff5722",
      weight: s.strokeWidth ?? 1,
      opacity: s.strokeOpacity ?? 1,
      fillColor: s.fillColor ?? s.strokeColor ?? "#ff5722",
      fillOpacity: s.fillOpacity ?? 0.25,
    };
  }
  return fallback || { color:"#ff5722", weight:1, opacity:1, fillColor:"#ff5722", fillOpacity:0.25 };
}

function styleForAnomalyFeature(f, fallback){
  const rec = overlayRegistry["Predictions"];
  const cat = rec?.categorical;
  return styleForCategoricalFeature(f, fallback, cat);
}



function applyCategoricalStyling(prop='class_name'){
  const rec = overlayRegistry["Predictions"];
  if (!rec || !rec.data || !rec.layer) { renderLegend(); return; }

  anomaliesProp = prop;

  const values = uniquePropValues(rec.data, prop);
  if (values.length && values.length <= 10){
    const palette = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#a6cee3','#b2df8a','#fb9a99','#fdbf6f','#cab2d6'];
    const mapping = {};
    values.forEach((v,i)=> mapping[v] = palette[i % palette.length]);
    rec.categorical = { prop, mapping, values };
    rec.layer.setStyle((f)=> styleForAnomalyFeature(f, rec.style));
  } else {
    rec.categorical = null;           // too many values → fallback
    rec.layer.setStyle(rec.style);
  }
  renderLegend();
}

function uniquePropValues(gj, prop){
  const s = new Set();
  (gj.features || []).forEach(f=>{
    if (f && f.properties && f.properties[prop] != null){
      s.add(String(f.properties[prop]));
    }
  });
  return Array.from(s).sort((a,b)=> String(a).localeCompare(String(b)));
}


function renderLegend(){
  const el = document.getElementById('legend');
  if (!el) return;
  el.innerHTML = '';

  const rec = overlayRegistry["Predictions"];
  if (!rec || !rec.layer) return;

  const title = document.createElement('div');
  title.className = 'legendHeader';
  const by = rec.categorical?.prop && rec.categorical.prop !== '__none__'
    ? ` — <span class="dim">by <b>${escapeHtml(rec.categorical.prop)}</b></span>` : '';
  title.innerHTML = `<div class="legendTitle">Predictions${by}</div>`;
  el.appendChild(title);

  const body = document.createElement('div'); body.className = 'legendBody'; el.appendChild(body);

  if (rec.categorical){
    const { prop, classes, values } = rec.categorical;
    if (prop === '__none__'){
      const s = classes?.['__ALL__'] || {};
      const fill = s.fillColor || s.strokeColor || '#ff5722';
      const stroke = s.strokeColor || '#202020';
      body.innerHTML = `<div class="legendItem"><span class="swatch" style="background:${fill}; border:2px solid ${stroke}"></span> All features</div>`;
    } else {
      body.innerHTML = values.map(v=>{
        const s = classes?.[v] || {};
        const fill = s.fillColor || s.strokeColor || '#ff5722';
        const stroke = s.strokeColor || '#202020';
        return `<div class="legendItem"><span class="swatch" style="background:${fill}; border:2px solid ${stroke}"></span> ${escapeHtml(v)}</div>`;
      }).join('');
    }
  }
}




async function loadImagesGeoJSON(url){
  try{
    const res = await fetch(url);
    const gj = await res.json();
    const feats = (gj && gj.type === "FeatureCollection") ? gj.features : [];
    // keep only valid point features
    return feats.filter(f => f?.geometry?.type === "Point" &&
                             Array.isArray(f.geometry.coordinates) &&
                             f.geometry.coordinates.length >= 2);
  }catch(_){
    return [];
  }
}

function populateImagesList(features){
  const ul = $("#imagesList");
  if (!ul) return;
  ul.innerHTML = "";

  features.forEach((f, i)=>{
    const name = f.properties?.name || f.properties?.file || f.properties?.image ||  `image ${i+1}`;
    const url  = f.properties?.url || "";
    const id   = `imgchk_${i}`;

    const [x, y] = f.geometry.coordinates;   // lon, lat
    const latlng = L.latLng(y, x);

    // build marker (not added by default)
    const html = `
      <div style="min-width:200px">
        <b>${escapeHtml(name)}</b>
        ${url ? `<div style="margin-top:6px"><img src="${url}" style="max-width:240px;max-height:180px;display:block;border-radius:6px;border:1px solid var(--border)"></div>` : ""}
        ${url ? `<div style="margin-top:6px"><a href="${url}" target="_blank">open image</a></div>` : ""}
      </div>`;
    const marker = L.marker(latlng).bindPopup(html);
    imageMarkers.set(id, marker);

    // list row
    const li = document.createElement("li");
    li.innerHTML = `
      <label class="chk">
        <input type="checkbox" id="${id}">
        <span>${escapeHtml(name)}</span>
      </label>
      <button class="iconDots" title="Zoom">🔍</button>
    `;

    const chk = li.querySelector("input");
    const zoomBtn = li.querySelector("button");

    chk.addEventListener("change", ()=>{
      if (chk.checked) marker.addTo(imageMarkersLayer);
      else { try{ imageMarkersLayer.removeLayer(marker); }catch(_){} }
    });
    zoomBtn.addEventListener("click", ()=>{
      MAP.setView(latlng, Math.max(MAP.getZoom(), 18));
      marker.openPopup();
    });

    ul.appendChild(li);
  });

  // bulk actions
  $("#btnShowAllImages")?.addEventListener("click", ()=>{
    $$("#imagesList input[type='checkbox']").forEach(cb=>{
      cb.checked = true;
      const m = imageMarkers.get(cb.id);
      if (m) m.addTo(imageMarkersLayer);
    });
  });
  $("#btnHideAllImages")?.addEventListener("click", ()=>{
    $$("#imagesList input[type='checkbox']").forEach(cb=>{
      cb.checked = false;
      const m = imageMarkers.get(cb.id);
      if (m) { try{ imageMarkersLayer.removeLayer(m); }catch(_){} }
    });
  });
}

// ---------------- Camera positions overrides (frontend-only) ----------------
function normalizeBasename(name){
  if(!name) return '';
  // strip path and extension, lower-case
  const b = name.split(/[/\\]/).pop();
  const noext = b.replace(/\.[^.]+$/, '');
  return noext.trim().toLowerCase();
}

function findOverrideForName(name){
  const stem = normalizeBasename(name);
  if(!stem) return null;
  // direct match
  if(cameraPositionOverrides[stem]) return cameraPositionOverrides[stem];

  // swap _v <-> _t suffix (common naming: *_V / *_T)
  const swap = (s) => {
    if(/[_\-]v$/.test(s)) return s.replace(/[_\-]v$/, (m)=> m[0] + 't');
    if(/[_\-]t$/.test(s)) return s.replace(/[_\-]t$/, (m)=> m[0] + 'v');
    return null;
  };
  const s2 = swap(stem);
  if(s2 && cameraPositionOverrides[s2]) return cameraPositionOverrides[s2];

  // try removing trailing _v/_t
  const s3 = stem.replace(/[_\-][vt]$/, '');
  if(s3 && cameraPositionOverrides[s3]) return cameraPositionOverrides[s3];

  return null;
}

function parseCameraPositionsFeatureCollection(gj){
  const out = {};
  if(!gj || gj.type !== 'FeatureCollection' || !Array.isArray(gj.features)) return out;
  for(const f of gj.features){
    try{
      const fn = f?.properties?.filename || f?.properties?.name || f?.properties?.image || f?.properties?.file;
      if(!fn) continue;
      const key = normalizeBasename(fn);
      const coords = f?.geometry?.coordinates;
      if(!coords || !Array.isArray(coords) || coords.length < 2) continue;
      // store as object { lon, lat, alt }
      out[key] = { lon: Number(coords[0]), lat: Number(coords[1]), alt: coords.length>2 ? Number(coords[2]) : null };
    }catch(_){ continue; }
  }
  return out;
}

function updateCameraPositionsInfo(matched, total){
  const el = document.getElementById('cameraPositionsInfo');
  if(!el) return;
  if(!total){ el.textContent = '' ; return; }
  el.textContent = `Camera positions: ${total} features; matched images: ${matched}. Using locations from uploaded file.`;
}

async function clearCameraPositions(){
  cameraPositionOverrides = {};
  updateCameraPositionsInfo(0,0);
  // re-render current session images if any
  if(lastLoadedSessionName){
    await loadImagesCatalog(lastLoadedSessionName, lastLoadedImagesUrl);
  }
}


// function renderLegend(){
//   const st = overlayRegistry["Anomalies"]?.style || { color:"#ff5722", fillColor:"#ff5722" };
//   const el = $("#legend");
//   el.innerHTML = `
//     <div class="legendItem"><span class="swatch" style="background:${st.fillColor}"></span> Anomaly (polygon)</div>
//     <div class="legendItem"><span class="swatch" style="background:#3388ff"></span> Image marker</div>
//   `;
// }


function setMapSectionLoading(loading, message = "Loading map data…"){
  const overlay = document.getElementById('mapSectionLoading');
  const text = document.getElementById('mapSectionLoadingText');
  if(text) text.textContent = message;
  if(overlay) overlay.hidden = !loading;
}

function setImagesSectionLoading(message){
  const list = document.getElementById('imagesList');
  if(!list) return;
  list.innerHTML = `<li class="mapListLoading"><span class="spinner" aria-hidden="true"></span><span>${escapeHtml(message)}</span></li>`;
}

function waitForMapTiles(layers, timeoutMs = 10000){
  const pending = (layers || []).map(layer => new Promise(resolve => {
    if(!layer?.once) return resolve();
    const finish = () => resolve();
    layer.once('load', finish);
    layer.once('tileerror', finish);
  }));
  if(!pending.length) return Promise.resolve();
  return Promise.race([
    Promise.all(pending),
    new Promise(resolve => setTimeout(resolve, timeoutMs)),
  ]);
}

async function applySessionToMap(sessionName){
  setMapSectionLoading(true, "Loading result map…");
  setImagesSectionLoading("Checking geospatial imagery…");
  try {
  // 0) clear any previous GeoTIFF tiles
  removeTifTiles();

  // 1) session summary (urls for geojsons)
  const res = await fetch(`/api/session_summary?session=${encodeURIComponent(sessionName)}`, { cache: 'no-store' });
  if (!res.ok) { console.warn('session_summary failed'); return; }
  const sum = await res.json();
  // cache the session summary so loadImagesCatalog can prefer rotated_images when available
  lastLoadedSessionSummary = sum || null;
  rotatedImagesLookup = null;

  const _sessionRootFromSummary = () => {
    const candidates = [
      sum?.images_geojson_url,
      sum?.images_geojson,
      sum?.images,
      sum?.images_gj,
      sum?.predictions_geojson,
      sum?.geojson_url,
      sum?.anomalies_geojson,
      sum?.geojson,
    ];
    for (const url of candidates) {
      if (!url || typeof url !== 'string') continue;
      const clean = url.split('?')[0];
      const idx = clean.lastIndexOf('/');
      if (idx > 0) return clean.slice(0, idx + 1);
    }
    return null;
  };

  const sessRoot = _sessionRootFromSummary();
  const anomaliesUrl = sum.predictions_geojson || sum.anomalies_geojson || sum.geojson_url || sum.geojson || (sessRoot ? (sessRoot + 'predictions.geojson') : null);
  const imagesUrl = sum.images_geojson_url || sum.images_geojson || sum.images || sum.images_gj || (sessRoot ? (sessRoot + 'images.geojson') : null);

  // 2) anomalies polygons (load regardless)
  if (anomaliesUrl){
    setMapSectionLoading(true, "Loading detection polygons…");
    try { await loadGeoJSON(anomaliesUrl); }
    catch(e){ console.warn('anomalies fetch failed:', e); }
  }

  // 3) Try ORIGINAL GeoTIFF tiles
  setMapSectionLoading(true, "Checking for an orthophoto…");
  let tiles = null;
  try{
    const r = await fetch(`/api/session_tiles?session=${encodeURIComponent(sessionName)}`, { cache:'no-store' });
    if (r.ok) tiles = await r.json();
  }catch(e){ console.warn('session_tiles failed:', e); }

  const hasTifTiles = !!(tiles?.ok && Array.isArray(tiles.layers) && tiles.layers.length);

  if (hasTifTiles){
    setMapSectionLoading(true, "Loading orthophoto tiles…");
    updateMapDetectionFilterVisibility(true);
    updateImageListButtonsVisibility(true);
    const b = createTifTileGroup(tiles.layers);
    TIF_TILE_GROUP  = b.group;
    TIF_TILE_LAYERS = b.layers;
    TIF_TILE_BOUNDS = b.firstBounds;
    TIF_TILE_BOUNDS = b.firstBounds;

    // show controller row inside Images list (replaces normal images there)
    await loadImagesCatalog(sessionName, imagesUrl);
    installTilesIntoImagesList(sessionName, tiles.layers, false);

    await waitForMapTiles(b.layers);

    // fit to raster on first load
    if (b.firstBounds){
      try{ MAP.fitBounds(b.firstBounds, { padding:[20,20] }); }catch(_){}
    }
  } else {
    updateMapDetectionFilterVisibility(false);
    updateImageListButtonsVisibility(false);
    // Fallback: point markers loaded from images.geojson
    await loadImagesCatalog(sessionName, imagesUrl);
    // Fit to camera locations (image markers) after all layers loaded
    const bounds = L.latLngBounds([]);
    
    try {
      const imagesRec = overlayRegistry["Image markers"];
      if (imagesRec?.layer) bounds.extend(imagesRec.layer.getBounds());
      if (bounds.isValid()) {
        // Store bounds to apply when map tab becomes visible
        pendingMapBounds = { bounds, options: {padding:[50,50]} };
        // Also try to apply now (works if map is already visible)
        try { MAP.fitBounds(bounds, {padding:[50,50]}); } catch(_){}
      }
    } catch(_){}
  }

  refreshLayersPanel();
  } finally {
    setMapSectionLoading(false);
  }
}

function createTifTileGroup(layerDefs, paneName = 'sessionOrthophotoPane'){
  const group = L.layerGroup();
  const layers = [];
  let firstBounds = null;

  layerDefs.forEach((Ldef, i) => {
    const bounds = (Array.isArray(Ldef.bounds) && Ldef.bounds.length === 2)
      ? L.latLngBounds(Ldef.bounds)
      : null;
    const lyr = L.tileLayer(Ldef.template, {
      minZoom: Ldef.minzoom ?? 0,
      maxZoom: Ldef.maxzoom ?? 22,
      bounds: bounds || undefined,
      opacity: 1,
      pane: paneName
    });
    group.addLayer(lyr);
    layers.push(lyr);

    if (!firstBounds && Array.isArray(Ldef.bounds) && Ldef.bounds.length === 2){
      const sw = L.latLng(Ldef.bounds[0][0], Ldef.bounds[0][1]);
      const ne = L.latLng(Ldef.bounds[1][0], Ldef.bounds[1][1]);
      firstBounds = L.latLngBounds(sw, ne);
    }
  });

  group.addTo(MAP);
  return { group, layers, firstBounds };
}

function installTilesIntoImagesList(sessionName, layerDefs, clearExisting = true){
  const list = document.getElementById('imagesList');
  if (!list) return;
  if (clearExisting) list.innerHTML = "";

  const label = layerDefs.length === 1 ? layerDefs[0].name : `Orthophoto (GeoTIFF)`;

  const row = document.createElement('li');
  row.innerHTML = `
    <label class="chk">
      <input type="checkbox" id="chkTifTiles" checked>
      <span>${escapeHtml(label)}</span>
    </label>
    <div style="margin-left:auto;display:flex;gap:.75rem;align-items:center;">
      <button class="iconDots orthoMenu" data-session="${escapeHtml(sessionName)}" title="Options">⋮</button>
    </div>
  `;

  const chk   = row.querySelector('#chkTifTiles');
  const orthoBtn = row.querySelector('.orthoMenu');

  chk.addEventListener('change', ()=>{
    if (!TIF_TILE_GROUP) return;
    if (chk.checked) TIF_TILE_GROUP.addTo(MAP);
    else { try{ MAP.removeLayer(TIF_TILE_GROUP); }catch(_){} }
  });

  orthoBtn.addEventListener('click', ()=>{
    openOrthoMenu(orthoBtn, sessionName, layerDefs);
  });

  if (clearExisting) list.appendChild(row);
  else list.prepend(row);
}



function installTileLayers(layers){
  // clear any previous raster tiles
  for (const tl of tileLayers){ try{ MAP.removeLayer(tl); }catch(_){} }
  tileLayers = [];

  // use a single registry key "Orthophoto" (group of all tile layers)
  const group = L.layerGroup();
  let firstBounds = null;

  layers.forEach((Ldef, i)=>{
    const bounds = (Array.isArray(Ldef.bounds) && Ldef.bounds.length === 2)
      ? L.latLngBounds(Ldef.bounds)
      : null;
    const lyr = L.tileLayer(Ldef.template, {
      minZoom: Ldef.minzoom ?? 0,
      maxZoom: Ldef.maxzoom ?? 22,
      bounds: bounds || undefined
    });
    group.addLayer(lyr);
    tileLayers.push(lyr);
    if (i === 0) lyr.addTo(MAP);   // show first by default

    if (!firstBounds && Array.isArray(Ldef.bounds) && Ldef.bounds.length === 2){
      const sw = L.latLng(Ldef.bounds[0][0], Ldef.bounds[0][1]);
      const ne = L.latLng(Ldef.bounds[1][0], Ldef.bounds[1][1]);
      firstBounds = L.latLngBounds(sw, ne);
    }
  });

  // register (replaces previous "Raster: …" entries)
  overlayRegistry["Orthophoto"] = { layer: group, type: "raster", bounds: firstBounds };

  if (firstBounds){
    try{ MAP.fitBounds(firstBounds, { padding:[20,20] }); }catch(_){}
  }
  refreshLayersPanel();
}

// Homography/helpers removed: backend provides rotated images.


// ---- Image markers (GeoJSON → Leaflet layer + registry) ----
function installImageMarkers(gj) {
  try {
    if (!gj || gj.type !== 'FeatureCollection') return;

    const key = 'Image markers';

    // remove previous instance, if any
    const prev = overlayRegistry[key];
    if (prev && prev.layer) {
      try { MAP.removeLayer(prev.layer); } catch (e) {}
    }

    // keep/restore previous style if user changed it via the Style modal
    const style = (prev && prev.style) || {
      color: '#0ea5e9',
      weight: 1,
      fillColor: '#0ea5e9',
      fillOpacity: 0.6
    };

    const layer = L.geoJSON(gj, {
      pointToLayer: (feat, latlng) =>
        L.circleMarker(latlng, { radius: 4, ...style })
          .bindPopup(`<div class="mini"><b>Image:</b> ${escapeHtml(feat?.properties?.image || feat?.properties?.name || feat?.properties?.file || '')}</div>`)
    }).addTo(MAP);

    overlayRegistry[key] = { layer, type: 'geojson', style };

    // make sure it appears in the panel
    if (typeof refreshLayersPanel === 'function') refreshLayersPanel();
  } catch (err) {
    console.error('installImageMarkers failed:', err);
  }
}

// tiny utility (only if you don't already have one)
function escapeHtml(s) {
  if (s == null) return '';
  return String(s).replace(/[&<>"']/g, m => (
    ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m])
  ));
}



async function loadImagesCatalog(sessionName, imagesUrl){
  imageCatalog = [];
  clearImageOverlays();
  const listEl = document.getElementById('imagesList');
  if (listEl) listEl.innerHTML = '<li class="dim">Loading…</li>';

  if (!imagesUrl){
    if (listEl) listEl.innerHTML = '<li class="dim">No geolocated images in this session</li>';
    return;
  }

  try {
    const gj = await (await fetch(imagesUrl, { cache: 'no-store' })).json();
    // store original and session info so overrides can be reapplied later
    lastLoadedImagesGJ = gj;
    lastLoadedSessionName = sessionName;
    lastLoadedImagesUrl = imagesUrl;

    // Build manifest lookup for detection counts
    const manifestLookup = {};
    const manifestStemLookup = {};
    if (lastLoadedSessionSummary?.manifest && Array.isArray(lastLoadedSessionSummary.manifest)){
      for (const item of lastLoadedSessionSummary.manifest){
        if (item.file && item.n != null){
          manifestLookup[item.file] = item.n;
          const stem = String(item.file).replace(/\.[^.]+$/, '');
          manifestStemLookup[stem] = item.n;
        }
      }
    }

    // Build modified copy of gj where we apply any cameraPositionOverrides
    const modifiedGJ = JSON.parse(JSON.stringify(gj));
    let matched = 0;
    if (Array.isArray(modifiedGJ.features)){
      for (const f of modifiedGJ.features){
        try{
          const file = f?.properties?.image || f?.properties?.file || f?.properties?.name;
          if(!file) continue;
          const ov = findOverrideForName(file);
          if (ov){
            // replace geometry coordinates [lon, lat, (alt)]
            f.geometry = f.geometry || { type: 'Point', coordinates: [ov.lon, ov.lat] };
            f.geometry.coordinates = [ov.lon, ov.lat].concat(ov.alt != null ? [ov.alt] : []);
            matched++;
          }
        }catch(_){ }
      }
    }

    // 2a) populate the “Image markers” layer using modified coordinates
    installImageMarkers(modifiedGJ);

    // 2b) build imageCatalog for actual image overlays (use modifiedGJ)
    imageCatalog = [];
    const overlaysFromSummary = Array.isArray(lastLoadedSessionSummary?.assets?.overlays)
      ? lastLoadedSessionSummary.assets.overlays
      : [];
    const overlayByName = new Map();
    for (const overlayUrl of overlaysFromSummary) {
      if (typeof overlayUrl !== 'string' || !overlayUrl) continue;
      const fileName = extractAssetBasename(overlayUrl);
      if (fileName) overlayByName.set(fileName, overlayUrl);
    }

    const sessionRoot = (() => {
      const seed = lastLoadedSessionSummary?.images_geojson_url
        || lastLoadedSessionSummary?.geojson_url
        || imagesUrl;
      if (!seed || typeof seed !== 'string') return null;
      const clean = seed.split('?')[0];
      const idx = clean.lastIndexOf('/');
      return idx > 0 ? clean.slice(0, idx + 1) : null;
    })();
    const feats = Array.isArray(modifiedGJ?.features) ? modifiedGJ.features : [];
    for (const f of feats){
      if (f?.geometry?.type !== 'Point') continue;

      const [lng, lat] = f.geometry.coordinates || [];
      if (typeof lat !== 'number' || typeof lng !== 'number') continue;

      // filename is in properties.image (per your sample)
      const fileRaw = f?.properties?.src || f?.properties?.image || f?.properties?.file || f?.properties?.name;
      const file = extractAssetBasename(fileRaw);
      if (!file) continue;

      const url = resolveFeatureOverlayUrl(f?.properties || {}, overlayByName, sessionRoot);
      if (!url) continue;

      // Place the prepared image at the corrected GeoJSON centre. Its residual
      // row-alignment orientation is applied by the shared map overlay helper.
      let bounds = null;
      let storedRotation = Number(f?.properties?.rotation ?? f?.properties?.rotation_heading ?? 0);
      const corners = f?.properties?.corners;
      if (f?.properties?.width_m && f?.properties?.height_m){
        try{
          const halfW = Number(f.properties.width_m) / 2.0;
          const halfH = Number(f.properties.height_m) / 2.0;
          const top = lat + (halfH / 111320);
          const bottom = lat - (halfH / 111320);
          const left = lng - (halfW / (111320 * Math.cos(lat * Math.PI / 180)));
          const right = lng + (halfW / (111320 * Math.cos(lat * Math.PI / 180)));
          bounds = L.latLngBounds(L.latLng(bottom, left), L.latLng(top, right));
        }catch(_){ bounds = null; }
      }

      // Older result files may only contain footprint corners.
      if (!bounds && Array.isArray(corners) && corners.length >= 4){
        try{
          const lons = corners.map(c => Number(c[0]));
          const lats = corners.map(c => Number(c[1]));
          const minLon = Math.min(...lons);
          const maxLon = Math.max(...lons);
          const minLat = Math.min(...lats);
          const maxLat = Math.max(...lats);
          const sw = L.latLng(minLat, minLon);
          const ne = L.latLng(maxLat, maxLon);
          bounds = L.latLngBounds(sw, ne);
        }catch(_){ bounds = null; }
      }

      // Fallback: small footprint around point if we don’t have true bounds
      if (!bounds){
        const { dLat, dLon } = metersToDeg(lat, DEFAULT_IMAGE_HALF_SIZE_M);
        const sw = L.latLng(lat - dLat, lng - dLon);
        const ne = L.latLng(lat + dLat, lng + dLon);
        bounds = L.latLngBounds(sw, ne);
      }

      // Get detection count from manifest
      const fileStr = String(file);
      const fileStem = fileStr.replace(/\.[^.]+$/, '');
      const decoded = (()=>{ try{ return decodeURIComponent(fileStr); }catch(_){ return fileStr; }})();
      const decodedStem = decoded.replace(/\.[^.]+$/, '');
      const detectionCount = (
        manifestLookup[fileStr] ??
        manifestLookup[decoded] ??
        manifestStemLookup[fileStem] ??
        manifestStemLookup[decodedStem] ??
        0
      );

      const matchTokens = buildNameVariantSet(file);
      imageCatalog.push({
        id: file,
        name: file,
        url,
        bounds,
        on: false,
        rotation: storedRotation,
        corners: Array.isArray(corners) ? corners : null,
        n: detectionCount,
        reviewOnly: f?.properties?.source_role === 'mosaic_input' && f?.properties?.inference_performed === false,
        matchTokens,
      });
    }

    applyMapDetectionFilter();
    // update info UI if camera positions present
    const totalOverrides = Object.keys(cameraPositionOverrides).length;
    updateCameraPositionsInfo(matched, totalOverrides);
  } catch (e) {
    console.warn('images_geojson parse failed:', e);
    if (listEl) listEl.innerHTML = '<li class="err">Failed to load images</li>';
  }
}


// Images panel interactions
const imagesListEl = document.getElementById('imagesList');
if (imagesListEl){
  imagesListEl.addEventListener('change', (e)=>{
    const t = e.target;
    if (t && t.classList.contains('imgToggle')){
      const id = t.getAttribute('data-id');
      toggleImageOverlay(id, t.checked);
    }
  });
  imagesListEl.addEventListener('click', (e)=>{
    const btn = e.target.closest('.imgMenu');
    if (!btn) return;
    openImageMenu(btn);
  });
}

const btnShowAll = document.getElementById('btnShowAllImages');
const btnHideAll = document.getElementById('btnHideAllImages');
if (btnShowAll) btnShowAll.addEventListener('click', ()=> setAllImageOverlays(true));
if (btnHideAll) btnHideAll.addEventListener('click', ()=> setAllImageOverlays(false));

document.addEventListener('click', (e) => {
  const btn = e.target.closest('.popupImageToggle');
  if (!btn) return;
  const imageId = btn.dataset.imageId;
  if (!imageId) return;
  const rec = imageCatalog.find(r => r.id === imageId);
  if (!rec) return;
  const nextState = !rec.on;
  toggleImageOverlay(imageId, nextState);
  const checkbox = document.querySelector(`.imgToggle[data-id="${CSS.escape(imageId)}"]`);
  if (checkbox){
    checkbox.checked = nextState;
    const row = checkbox.closest('li');
    if (row){
      try{ row.scrollIntoView({ behavior:'smooth', block:'center' }); }catch(_){ }
    }
  }
  btn.dataset.state = nextState ? 'on' : 'off';
  btn.classList.toggle('on', nextState);
  btn.classList.toggle('off', !nextState);
  const label = nextState ? 'Remove image' : 'View image';
  btn.title = label;
  btn.setAttribute('aria-label', label);
  const text = btn.querySelector('.popupImageToggleText');
  if (text) text.textContent = label;
});



// ---------- layers panel + ⋮ menu ----------
function refreshLayersPanel(){
  const list = document.getElementById('layersList');   // your main layers UL
  if (!list) return;

  // Build Layers list from overlayRegistry
  const entries = Object.entries(overlayRegistry);
  entries.sort(([a],[b]) => a.localeCompare(b));

  list.innerHTML = entries.map(([key, rec]) => `
    <li>
      <label class="chk">
        <input type="checkbox" class="layerToggle" data-key="${escapeHtml(key)}" ${MAP.hasLayer(rec.layer) ? 'checked' : ''}>
        <span>${escapeHtml(key)}${rec.reference ? ' <small class="muted">· linked</small>' : ''}</span>
      </label>
      <button class="iconDots layerMenuBtn" data-key="${escapeHtml(key)}" title="Layer menu">⋮</button>
    </li>
  `).join('');

  // Delegate: toggles + menu
  list.addEventListener('change', (e) => {
    const cb = e.target.closest('.layerToggle');
    if (!cb) return;
    const key = cb.dataset.key;
    const rec = overlayRegistry[key];
    if (!rec || !rec.layer) return;
    if (cb.checked) {
      rec.layer.addTo(MAP);
    } else {
      MAP.removeLayer(rec.layer);
    }
  });
  
  list.addEventListener('input', (e) => {
    const slider = e.target.closest('.layerOpacity');
    if (!slider) return;
    const key = slider.dataset.key;
    const rec = overlayRegistry[key];
    if (!rec || !rec.layer) return;
    const opacity = slider.value / 100;
    if (rec.layer.setOpacity) {
      rec.layer.setOpacity(opacity);
    } else if (rec.layer.eachLayer) {
      rec.layer.eachLayer(l => {
        if (l.setOpacity) l.setOpacity(opacity);
      });
    }
    const valSpan = document.getElementById(`opacity-val-${escapeHtml(key)}`);
    if (valSpan) valSpan.textContent = slider.value + '%';
  });

  list.addEventListener('click', (e) => {
    const btn = e.target.closest('.layerMenuBtn');
    if (!btn) return;
    openLayerMenu(btn);
  });

  // Note: the Images sidebar list is managed by renderImagesList() — not here.
}



function setAllImages(show){
  
  imageMarkers.forEach(im => {
    im.shown = show;
    if (show) im.marker.addTo(imageMarkersLayer);
    else imageMarkersLayer.removeLayer(im.marker);
  });
  // reflect in checkboxes
  const imagesUl = document.querySelector('#imagesList');
  if (imagesUl) {
    imagesUl.querySelectorAll('input[type=checkbox][data-img]').forEach((cb, i) => cb.checked = show);
  }
}


// function openLayerMenu(name, info, x, y){
//   const menu = $("#layerMenu");
//   layerMenuState = { name, info };
//   menu.style.left = `${x}px`;
//   menu.style.top = `${y}px`;
//   menu.classList.remove("hidden");
// }
// window.addEventListener("click", ()=> $("#layerMenu").classList.add("hidden"));
// $("#layerMenu [data-act='zoom']").addEventListener("click", ()=>{
//   const { info } = layerMenuState;
//   if(!info) return;
//   try{
//     const b = info.layer.getBounds ? info.layer.getBounds() : null;
//     if(b && b.isValid()) MAP.fitBounds(b, {padding:[20,20]});
//   }catch(_){}
//   $("#layerMenu").classList.add("hidden");
// });
// $("#layerMenu [data-act='style']").addEventListener("click", ()=>{
//   const { name, info } = layerMenuState;
//   if(!info) return;
//   styleTarget = { name, info };
//   const st = info.style || { color:"#ff5722", opacity:1, weight:1, fillColor:"#ff5722", fillOpacity:0.25 };
//   $("#stColor").value = toHex(st.color);
//   $("#stWidth").value = st.weight ?? 1;
//   $("#stOpacity").value = st.opacity ?? 1;
//   $("#fiColor").value = toHex(st.fillColor || st.color || "#ff5722");
//   $("#fiOpacity").value = st.fillOpacity ?? 0.25;
//   $("#styleModal").classList.remove("hidden");
//   $("#layerMenu").classList.add("hidden");
// });
function toHex(c){
  if(!c) return "#ff5722";
  const ctx = document.createElement("canvas").getContext("2d");
  ctx.fillStyle = c;
  return ctx.fillStyle;
}
$("#btnCloseStyle").addEventListener("click", ()=> $("#styleModal").classList.add("hidden"));
$("#btnCancelStyle").addEventListener("click", ()=> $("#styleModal").classList.add("hidden"));

// $("#btnApplyStyle").addEventListener("click", ()=>{
//   if(!styleTarget) return;
//   const st = {
//     color: $("#stColor").value,
//     weight: parseFloat($("#stWidth").value || "1"),
//     opacity: parseFloat($("#stOpacity").value || "1"),
//     fillColor: $("#fiColor").value,
//     fillOpacity: parseFloat($("#fiOpacity").value || "0.25")
//   };
//   styleTarget.info.style = st;

//   if(styleTarget.info.type === "geojson"){
//     styleTarget.info.layer.setStyle(st);   // works for “Anomalies” and now “Image markers”
//     renderLegend();
//   }else if(styleTarget.info.type === "raster"){
//     const op = Math.max(0, Math.min(1, st.opacity));
//     for(const l of tileLayers) l.setOpacity(op);
//   }
//   $("#styleModal").classList.add("hidden");
//   styleTarget = null;
// });

const btnApply = document.getElementById('btnApplyStyle');
if (btnApply) btnApply.style.display = 'none';  // hide the button

// Live-apply on any input change
['stColor','fiColor','stWidth','stOpacity','fiOpacity'].forEach(id=>{
  const el = document.getElementById(id);
  if (!el) return;
  // apply on change & while dragging (range) / typing (color/text/number)
  el.addEventListener('input', applyStyleLive);
  el.addEventListener('change', applyStyleLive);
});


// Camera positions upload (WebODM) - frontend-only overrides
$("#fileCameraPositions")?.addEventListener("change", async (e)=>{
  const f = e.target.files[0];
  if(!f) return;
  try{
    const text = await f.text();
    const gj = JSON.parse(text);
    const parsed = parseCameraPositionsFeatureCollection(gj);
    cameraPositionOverrides = parsed;
    const total = Object.keys(parsed).length;
    // If images already loaded for a session, re-run loadImagesCatalog to apply overrides
    if(lastLoadedSessionName && lastLoadedImagesUrl){
      await loadImagesCatalog(lastLoadedSessionName, lastLoadedImagesUrl);
    } else {
      updateCameraPositionsInfo(0, total);
    }
  }catch(ex){
    alert('Invalid JSON');
    console.warn('camera positions parse failed', ex);
  }finally{
    e.target.value = "";
  }
});

$("#btnClearCameraPositions")?.addEventListener('click', async (e)=>{ e.preventDefault(); await clearCameraPositions(); });

// ---------- logs (SSE) ----------
let evtSource = null;
function connectLogs(){
  if(evtSource){ evtSource.close(); evtSource = null; }
  evtSource = new EventSource(api.logs);
  setText("#logConn","connecting…");
  evtSource.onopen  = ()=> setText("#logConn","connected");
  evtSource.onerror = ()=>{
    setText("#logConn","reconnecting…");
    try{ evtSource.close(); }catch(_){}
    setTimeout(()=> {
      evtSource = new EventSource(api.logs);
    }, 3000);
  };
  evtSource.onmessage = (e)=>{
    const line = e.data;
    appendLog(line);

    // Route detectron2 progress lines into mini-train log (epochs/iters)
    const isTrainLine = (
      line.includes("[train]") ||
      line.includes("UI:INFO:train") || line.includes("UI:OK:train") || line.includes("UI:ERR:train") ||
      /\biter[: ]/i.test(line) || /\bloss[: ]/i.test(line) || /\beta[: ]/i.test(line)
    );
    if(isTrainLine){ appendMiniLog("#trainMiniLog", line); }

    const isTestLine = (line.includes("[test]") || line.includes("UI:INFO:test") || line.includes("UI:OK:test") || line.includes("UI:WARN:test") || line.includes("UI:ERR:test"));
    if(isTestLine){ appendMiniLog("#testMiniLog", line); }
    if(line.includes("Orthomosaic split progress:")){
      const progress = line.split("Orthomosaic split progress:").pop().trim();
      setText("#testStatus", `Splitting orthomosaic: ${progress}`);
    }else if(line.includes("Creating orthomosaic preview")){
      setText("#testStatus", "Creating orthomosaic preview…");
    }else if(line.includes("Finalizing prepared inputs")){
      setText("#testStatus", "Finalizing prepared inputs…");
    }else if(line.includes("Starting model inference")){
      setText("#testStatus", "Running inference…");
    }else if(line.includes("Aligning prepared images to solar rows")){
      setText("#testStatus", "Aligning prepared images to solar rows…");
    }else if(line.includes("Solar-row alignment progress:")){
      const progress = line.split("Solar-row alignment progress:").pop().trim();
      setText("#testStatus", `Aligning images to rows: ${progress}`);
    }else if(line.includes("Solar-row alignment complete:")){
      setText("#testStatus", "Solar-row alignment complete. Preparing inference…");
    }else if(line.includes("Inference complete")){
      setText("#testStatus", "Inference complete. Preparing result files…");
    }else if(line.includes("Preparing prediction manifest")){
      setText("#testStatus", "Preparing prediction manifest…");
    }else if(line.includes("Generating prediction and image GeoJSON")){
      setText("#testStatus", "Generating map GeoJSON…");
    }else if(line.includes("GeoJSON outputs ready")){
      setText("#testStatus", "Map GeoJSON ready. Finalizing result…");
    }else if(line.includes("Finalizing result metadata")){
      setText("#testStatus", "Finalizing result metadata…");
    }

    // Surface server-side warnings to the frontend warning panels so users
    // see when thermal was requested but data is missing (or other test warnings).
    if(line.includes('UI:WARN:test:')){
      const msg = line.split('UI:WARN:test:').pop().trim();
      warn('test', msg);
      wireAlertClose();
    }
    if(line.includes('UI:WARN:train:')){
      const msg = line.split('UI:WARN:train:').pop().trim();
      warn('train', msg);
      wireAlertClose();
    }

    // Spinner/status controls from UI-tagged lines only
    if(line.includes("UI:OK:train: Training started")){
      setText("#trainStatus","Training started…");
      setHidden($("#spinTrain"), false);
    }
    if(line.includes("UI:OK:train: Training completed")){
      setText("#trainStatus","Training completed.");
      setHidden($("#spinTrain"), true);
  ok("train","Training completed.");
  wireAlertClose();
  refreshModelViews();
    }
    if(line.includes("UI:ERR:train:")){
      setHidden($("#spinTrain"), true);
      err("train", line.split("UI:ERR:train:").pop().trim());
      wireAlertClose();
      setText("#trainStatus","Training failed.");
    }

    if(line.includes("UI:OK:test: Test complete")){
      // The request handler still loads the completed result into the UI. Its
      // finally block owns the spinner and its success path sets final status.
      setText("#testStatus","Test processing complete. Loading result…");
    }
    if(line.includes("UI:ERR:test:")){
      setHidden($("#spinTest"), true);
      err("test", line.split("UI:ERR:test:").pop().trim());
      wireAlertClose();
      setText("#testStatus","Test failed.");
    }
  };
}

// ---------- map session selection ----------
async function refreshMapSessionSelected(){
  const session = $("#selMapSession").value;
  if(!session) return;
  await applySessionToMap(session);
}


// --- lightbox gallery state ---
let _gallery = [];     // array of {src, file}
let _gIdx = 0;
let _lightboxOpen = false;
let _lightboxScale = 1.0;
let _lightboxPanX = 0;
let _lightboxPanY = 0;
let _isDragging = false;
let _dragStartX = 0;
let _dragStartY = 0;
let _dragStartPanX = 0;
let _dragStartPanY = 0;

function _applyLightboxScale(){
  const img = document.getElementById('lightboxImg');
  if(!img) return;
  // apply both scale and pan transformations
  img.style.transform = `translate(${_lightboxPanX}px, ${_lightboxPanY}px) scale(${_lightboxScale})`;
  img.style.transition = _isDragging ? 'none' : 'transform 120ms ease-out';
  img.style.display = 'block';
  img.style.maxWidth = 'none';
  img.style.cursor = _lightboxScale > 1.0 ? 'grab' : 'default';
}

function _zoomIn(step = 0.15){
  _lightboxScale = Math.min(5.0, _lightboxScale + step);
  _applyLightboxScale();
}
function _zoomOut(step = 0.15){
  _lightboxScale = Math.max(0.2, _lightboxScale - step);
  _applyLightboxScale();
}
function _resetZoom(){
  _lightboxScale = 1.0;
  _lightboxPanX = 0;
  _lightboxPanY = 0;
  _applyLightboxScale();
}

function _setLightbox(idx){
  _gIdx = Math.max(0, Math.min(idx, _gallery.length - 1));
  const it = _gallery[_gIdx];
  const img = document.getElementById("lightboxImg");
  const ttl = document.getElementById("lightboxTitle");
  const ctr = document.getElementById("imgCounter");
  if (img) img.src = it.src;
  if (ttl) ttl.textContent = it.file || "";
  if (ctr) ctr.textContent = `${_gIdx + 1} / ${_gallery.length}`;

  // Reset zoom and pan when changing images
  _lightboxScale = 1.0;
  _lightboxPanX = 0;
  _lightboxPanY = 0;
  _applyLightboxScale();

  const prev = document.getElementById("imgPrev");
  const next = document.getElementById("imgNext");
  if (prev) prev.disabled = (_gIdx === 0);
  if (next) next.disabled = (_gIdx === _gallery.length - 1);
}

function _openLightboxWithGallery(items, startIdx){
  _gallery = items.map(it => ({ src: it.overlay, file: it.file }));
  _gIdx = Math.max(0, Math.min(startIdx || 0, _gallery.length - 1));
  const lightbox = document.getElementById("lightbox");
  lightbox?.classList.add("show");
  lightbox?.classList.remove("hidden");
  _lightboxOpen = true;
  _setLightbox(_gIdx);
}

function _closeLightbox(){
  const lightbox = document.getElementById("lightbox");
  lightbox?.classList.remove("show");
  lightbox?.classList.add("hidden");
  _lightboxOpen = false;
}

function _nextImg(){ if (_gIdx < _gallery.length - 1) _setLightbox(_gIdx + 1); }
function _prevImg(){ if (_gIdx > 0) _setLightbox(_gIdx - 1); }


// ---------- UI wiring ----------
function setupUI(){
  setupTabs();

  let infoModalReturnFocus = null;
  const closeInfoModal = modal => {
    if(!modal) return;
    modal.classList.remove('show');
    modal.classList.add('hidden');
    const returnTarget = infoModalReturnFocus;
    infoModalReturnFocus = null;
    returnTarget?.focus();
  };
  const wireInfoModal = (buttonId, modalId) => {
    const button = document.getElementById(buttonId);
    const modal = document.getElementById(modalId);
    if(!button || !modal) return;
    button.addEventListener('click', event => {
      event.preventDefault();
      event.stopPropagation();
      infoModalReturnFocus = button;
      modal.classList.add('show');
      modal.classList.remove('hidden');
      modal.querySelector('[data-close-info-modal]')?.focus();
    });
    modal.querySelectorAll('[data-close-info-modal]').forEach(closeButton => {
      closeButton.addEventListener('click', () => closeInfoModal(modal));
    });
    modal.addEventListener('click', event => {
      if(event.target === modal) closeInfoModal(modal);
    });
  };
  wireInfoModal('btnLensCorrectionInfo', 'lensCorrectionInfoModal');
  wireInfoModal('btnTargetSurfaceHeightInfo', 'targetSurfaceHeightInfoModal');
  wireInfoModal('btnMosaicInfo', 'mosaicInfoModal');
  wireInfoModal('btnThermalTrainingInfo', 'thermalTrainingInfoModal');

  const lensCorrectionToggle = document.getElementById('chkUndistortThermal');
  const undistortedExportToggle = document.getElementById('chkExportUndistortedImages');
  const updateUndistortedExportVisibility = () => {
    const enabled = lensCorrectionToggle?.checked === true;
    setHidden(document.getElementById('undistortedExportControls'), !enabled);
    if(!enabled && undistortedExportToggle) undistortedExportToggle.checked = false;
  };
  lensCorrectionToggle?.addEventListener('change', updateUndistortedExportVisibility);
  updateUndistortedExportVisibility();

  const mosaicToggle = document.getElementById('chkMosaicImages');
  const updateMosaicOptionsVisibility = () => {
    setHidden(document.getElementById('mosaicOptions'), !mosaicToggle?.checked);
  };
  mosaicToggle?.addEventListener('change', updateMosaicOptionsVisibility);
  updateMosaicOptionsVisibility();
  document.addEventListener('keydown', event => {
    if(event.key !== 'Escape') return;
    const openInfoModal = document.querySelector('.workflowInfoModal.show');
    if(openInfoModal) closeInfoModal(openInfoModal);
  });

  const btnRefreshFolders = $("#btnRefreshFolders");
  if(btnRefreshFolders) btnRefreshFolders.addEventListener("click", loadDatasets);
  const selTestFolder = document.getElementById('selTestFolder');
  if(selTestFolder){ selTestFolder.addEventListener('change', onTestDatasetChange); }
  document.getElementById('btnTestAssetData')?.addEventListener('click', () => switchTestAssetPanel('data'));
  document.getElementById('btnTestAssetResults')?.addEventListener('click', () => switchTestAssetPanel('results'));
  document.getElementById('btnRefreshTestDataList')?.addEventListener('click', loadDatasets);
  document.getElementById('btnRefreshTestResultsList')?.addEventListener('click', () => loadSessions(false));
  document.getElementById('btnOpenTestUploadFromList')?.addEventListener('click', openUploadModal);
  document.querySelector('.testAssetTabs')?.addEventListener('keydown', event => {
    if(!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault();
    const showResults = event.key === 'ArrowRight' || event.key === 'End';
    switchTestAssetPanel(showResults ? 'results' : 'data');
    document.getElementById(showResults ? 'btnTestAssetResults' : 'btnTestAssetData')?.focus();
  });
  const btnRefreshModels = $("#btnRefreshModels");
  if(btnRefreshModels) btnRefreshModels.addEventListener("click", ()=> loadModels(getSelectedBackend(), '#selModelFolder'));
  const btnRefreshTrainedModels = $("#btnRefreshTrainedModels");
  if(btnRefreshTrainedModels) btnRefreshTrainedModels.addEventListener("click", loadTrainedModels);
  document.getElementById('btnAssetModels')?.addEventListener('click', () => switchTrainingAssetPanel('models'));
  document.getElementById('btnAssetData')?.addEventListener('click', () => switchTrainingAssetPanel('data'));
  document.querySelector('#tab-train .trainingAssetTabs')?.addEventListener('keydown', event => {
    if(!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault();
    const tabs = Array.from(event.currentTarget.querySelectorAll('[role="tab"]'));
    const current = Math.max(0, tabs.indexOf(document.activeElement));
    let next = current;
    if(event.key === 'Home') next = 0;
    else if(event.key === 'End') next = tabs.length - 1;
    else if(event.key === 'ArrowRight') next = (current + 1) % tabs.length;
    else if(event.key === 'ArrowLeft') next = (current - 1 + tabs.length) % tabs.length;
    const target = tabs[next];
    switchTrainingAssetPanel(target?.id === 'btnAssetData' ? 'data' : 'models');
    target?.focus();
  });
  document.getElementById('btnRefreshTrainingData')?.addEventListener('click', loadTrainingDatasets);
  document.getElementById('btnRefreshTrainingDatasetOptions')?.addEventListener('click', loadTrainingDatasets);
  document.getElementById('selTrainingDataset')?.addEventListener('change', event => {
    const selected = trainingDatasetsCache.find(dataset => dataset.id === event.target.value);
    const hint = document.getElementById('trainingDatasetHint');
    if(hint && selected) hint.textContent = `${selected.display_name || 'Training dataset'} · ID: ${selected.id}`;
  });
  document.getElementById('btnOpenTrainingUpload')?.addEventListener('click', openTrainingUploadModal);
  document.getElementById('btnCloseTrainingUpload')?.addEventListener('click', closeTrainingUploadModal);
  document.getElementById('btnCancelTrainingUpload')?.addEventListener('click', () => {
    if(trainingUploadXhr) trainingUploadXhr.abort();
    else closeTrainingUploadModal();
  });
  document.getElementById('btnStartTrainingUpload')?.addEventListener('click', startTrainingDatasetUpload);
  document.getElementById('btnCloseTrainingDatasetSummary')?.addEventListener('click', closeTrainingDatasetSummary);
  document.getElementById('btnCloseModelDetails')?.addEventListener('click', closeModelDetails);
  document.getElementById('fileTrainingZip')?.addEventListener('change', event => {
    if(event.target.files?.length){
      selectedTrainingFolderFiles = [];
      selectedTrainingFolderName = '';
    }
    updateTrainingUploadSelection();
  });
  document.getElementById('btnChooseTrainingFolder')?.addEventListener('click', chooseTrainingDatasetFolder);
  const trainingUploadModal = document.getElementById('trainingUploadModal');
  trainingUploadModal?.addEventListener('click', event => { if(event.target === trainingUploadModal) closeTrainingUploadModal(); });
  const trainingSummaryModal = document.getElementById('trainingDatasetSummaryModal');
  trainingSummaryModal?.addEventListener('click', event => { if(event.target === trainingSummaryModal) closeTrainingDatasetSummary(); });
  const modelDetailsModal = document.getElementById('modelDetailsModal');
  modelDetailsModal?.addEventListener('click', event => { if(event.target === modelDetailsModal) closeModelDetails(); });
  const btnCloseUpload = $("#btnCloseUploadModal");
  if(btnCloseUpload) btnCloseUpload.addEventListener("click", ()=>{ closeUploadModal(); resetUploadProgress(); });
  const btnCancelUpload = $("#btnCancelUpload");
  if(btnCancelUpload) btnCancelUpload.addEventListener("click", ()=>{ closeUploadModal(); resetUploadProgress(); });
  const btnStartUpload = $("#btnStartUpload");
  if(btnStartUpload) btnStartUpload.addEventListener("click", startUpload);
  
  const filesTestInput = $("#filesTest");
  if(filesTestInput) filesTestInput.addEventListener("change", updateFileCount);

  const btnTrain = $("#btnStartTraining");
  if(btnTrain) btnTrain.addEventListener("click", startTraining);
  const btnCancelTrain = $("#btnCancelTrain");
  if(btnCancelTrain) btnCancelTrain.addEventListener("click", cancelTraining);

  const btnTest = $("#btnRunTest");
  if(btnTest) btnTest.addEventListener("click", runTest);
  const btnCancelTest = $("#btnCancelTest");
  if(btnCancelTest) btnCancelTest.addEventListener("click", cancelTest);

  const btnToggleTestAdvanced = document.getElementById('btnToggleTestAdvanced');
  const testAdvancedOptions = document.getElementById('testAdvancedOptions');
  if (btnToggleTestAdvanced && testAdvancedOptions) {
    testAdvancedOptions.hidden = true;
    btnToggleTestAdvanced.textContent = 'Advanced ▸';
    btnToggleTestAdvanced.addEventListener('click', (e) => {
      e.preventDefault();
      const isOpen = !testAdvancedOptions.hidden;
      testAdvancedOptions.hidden = isOpen;
      btnToggleTestAdvanced.textContent = isOpen ? 'Advanced ▸' : 'Advanced ▾';
    });
  }
  
  // Detection filter toggles
  const chkShowOnlyDetections = document.getElementById('chkShowOnlyDetections');
  if(chkShowOnlyDetections){ chkShowOnlyDetections.addEventListener('change', _applyDetectionFilter); }
  const chkMapShowOnlyDetections = document.getElementById('chkMapShowOnlyDetections');
  if(chkMapShowOnlyDetections){ chkMapShowOnlyDetections.addEventListener('change', applyMapDetectionFilter); }
  
  const lnkLogsTest = $("#lnkToLogsFromTest");
  if(lnkLogsTest) lnkLogsTest.addEventListener("click", (e)=>{ e.preventDefault(); switchToTab("tab-logs"); });
  const lnkLogsTrain = $("#lnkToLogsFromTrain");
  if(lnkLogsTrain) lnkLogsTrain.addEventListener("click", (e)=>{ e.preventDefault(); switchToTab("tab-logs"); });

  const btnLogsConnect = $("#btnLogsConnect");
  if(btnLogsConnect) btnLogsConnect.addEventListener("click", connectLogs);
  const btnLogsClear = $("#btnLogsClear");
  if(btnLogsClear) btnLogsClear.addEventListener("click", ()=> resetLogPane("#logStream"));

  const btnRefreshSessions = $("#btnRefreshSessions");
  if(btnRefreshSessions) btnRefreshSessions.addEventListener("click", async ()=>{
    await ensureResultsTabLoaded(true);
  });
  const selResults = $("#selResults");
  if(selResults) selResults.addEventListener("change", showResultsForSelected);

  const btnRefreshMapSessions = $("#btnRefreshMapSessions");
  if(btnRefreshMapSessions) btnRefreshMapSessions.addEventListener("click", async ()=>{
    await ensureMapTabLoaded(true);
  });
  const selMapSession = $("#selMapSession");
  if(selMapSession) selMapSession.addEventListener("change", refreshMapSessionSelected);

  // initial hidden spinners (also hidden attribute in HTML prevents flash)
  setHidden($("#spinTrain"), true);
  setHidden($("#spinTest"), true);

  wireAlertClose();
  document.addEventListener('click', ()=> closeTrainedModelMenus());

  // Backend selector wiring: show YOLO options when YOLO is selected
  const selBackendGlobal = $("#selBackend");
  const selBackendTrain = $("#selBackendTrain");
  const selBackendTest  = $("#selBackendTest");

  // Train-side: show/hide YOLO-specific train options and the Detectron model-type selector
  function _updateYoloUIForTrain(){
    const b = (selBackendTrain && selBackendTrain.value) ? selBackendTrain.value : (selBackendGlobal && selBackendGlobal.value) || 'detectron';
    const show = (b === 'yolo');
    const elOpts = $("#yoloOptions");
    const elSize = $("#yoloSizeOption");
    if(elOpts) elOpts.style.display = show ? 'block' : 'none';
    if(elSize) elSize.style.display = show ? 'block' : 'none';
    const task = document.getElementById('selTrainingTask')?.value || 'detect';
    const hint = document.getElementById('trainingTaskHint');
    if(hint) hint.textContent = show
      ? `YOLO will train a ${task === 'segment' ? 'segmentation' : 'detection'} model.`
      : `Detectron will train ${task === 'segment' ? 'Mask R-CNN' : 'Faster R-CNN'}.`;
    const family = document.getElementById('selYoloFamily');
    if(family){
      if(show && task === 'segment') family.value = 'v8';
      family.disabled = show && task === 'segment';
      family.title = family.disabled ? 'YOLO segmentation currently uses YOLOv8.' : '';
    }
  }
  if(selBackendGlobal) selBackendGlobal.addEventListener('change', _updateYoloUIForTrain);
  if(selBackendTrain) selBackendTrain.addEventListener('change', ()=>{ _updateYoloUIForTrain(); populateTrainingDatasetOptions(); });
  const selTrainingTask = document.getElementById('selTrainingTask');
  if(selTrainingTask) selTrainingTask.addEventListener('change', ()=>{ _updateYoloUIForTrain(); populateTrainingDatasetOptions(); });
  _updateYoloUIForTrain();

  const chkUseThermalTrain = document.getElementById('chkUseThermalTrain');
  if (chkUseThermalTrain){
    chkUseThermalTrain.addEventListener('change', ()=>{ userToggledThermalTrain = true; });
  }

  // Test-side: when test-backend changes, reload the models list filtered for that backend
  async function _onBackendChangeForTest(backend){
    try{ await loadModels(backend, '#selModelFolder'); }catch(_){ /* ignore */ }
  }
  if(selBackendTest) selBackendTest.addEventListener('change', ()=> _onBackendChangeForTest(selBackendTest.value));
  // Initialize test models list according to current test/backend selector
  if(selBackendTest){ _onBackendChangeForTest(selBackendTest.value || (selBackendGlobal && selBackendGlobal.value)); }

  // When the selected model changes in the test selector, update the test thermal checkbox
  // to reflect the model's channel_count unless the user manually toggled the checkbox.
  const selModelFolder = document.getElementById('selModelFolder');
  if (selModelFolder){
    selModelFolder.addEventListener('change', ()=>{
      if (userToggledThermalTest) return;
      const sel = selModelFolder.value;
      const m = modelsCache[sel];
      const def = (m && m.channel_count) ? (
        Number(m.channel_count) === 3 && (!!(m && m.thermal_only) || !!(m && m.thermal_used))
      ) : false;
      const chk = document.getElementById('chkUseThermalTest');
      if (chk){ try{ chk.checked = !!def; }catch(_){ } }
    });
  }

  // Note: the frontend option to extract decoded thermal as 3-channel JPGs
  // has been removed. Backend will only attempt decoding when the selected
  // model explicitly declares RGB+thermal input.

  // Track manual toggles so we don't overwrite user intent when a model is auto-selected
  const chkTrain = document.getElementById('chkUseThermalTrain');
  if (chkTrain) chkTrain.addEventListener('change', ()=> { userToggledThermalTrain = true; });
  const chkTest = document.getElementById('chkUseThermalTest');
  if (chkTest) chkTest.addEventListener('change', ()=> { userToggledThermalTest = true; });

  // lightbox
  $("#btnCloseLightbox").addEventListener("click", _closeLightbox);
  document.getElementById("imgNext")?.addEventListener("click", _nextImg);
  document.getElementById("imgPrev")?.addEventListener("click", _prevImg);

  // Zoom controls wiring (buttons are in HTML as #imgZoomIn/#imgZoomOut/#imgResetZoom)
  document.getElementById("imgZoomIn")?.addEventListener("click", ()=> _zoomIn());
  document.getElementById("imgZoomOut")?.addEventListener("click", ()=> _zoomOut());
  document.getElementById("imgResetZoom")?.addEventListener("click", ()=> _resetZoom());

  // Wheel zoom on image and mouse drag to pan when zoomed (simple implementation)
  const lbImg = document.getElementById('lightboxImg');
  if(lbImg){
    lbImg.style.transformOrigin = 'center center';
    lbImg.addEventListener('wheel', (ev)=>{
      if (!_lightboxOpen) return;
      ev.preventDefault();
      const delta = Math.sign(ev.deltaY) * -0.075; // wheel up -> zoom in
      if (delta > 0) _zoomIn(delta); else _zoomOut(-delta);
    }, { passive: false });

    // Mouse drag to pan when zoomed
    lbImg.addEventListener('mousedown', (ev)=>{
      if (!_lightboxOpen || _lightboxScale <= 1.0) return;
      ev.preventDefault();
      _isDragging = true;
      _dragStartX = ev.clientX;
      _dragStartY = ev.clientY;
      _dragStartPanX = _lightboxPanX;
      _dragStartPanY = _lightboxPanY;
      lbImg.style.cursor = 'grabbing';
    });

    document.addEventListener('mousemove', (ev)=>{
      if (!_isDragging) return;
      ev.preventDefault();
      const dx = ev.clientX - _dragStartX;
      const dy = ev.clientY - _dragStartY;
      _lightboxPanX = _dragStartPanX + dx;
      _lightboxPanY = _dragStartPanY + dy;
      _applyLightboxScale();
    });

    document.addEventListener('mouseup', ()=>{
      if (_isDragging) {
        _isDragging = false;
        const img = document.getElementById('lightboxImg');
        if (img) img.style.cursor = _lightboxScale > 1.0 ? 'grab' : 'default';
      }
    });
  }

  // keyboard: ← - Esc (also allow A/D)
  document.addEventListener("keydown", (e)=>{
    if (!_lightboxOpen) return;
    const k = e.key;
    if (k === "ArrowRight" || k === "d" || k === "D") { _nextImg(); e.preventDefault(); }
    else if (k === "ArrowLeft" || k === "a" || k === "A") { _prevImg(); e.preventDefault(); }
    else if (k === "Escape") { _closeLightbox(); e.preventDefault(); }
  });

  // Wire up images opacity slider
  const imagesOpacitySlider = document.getElementById('imagesOpacitySlider');
  const imagesOpacityValue = document.getElementById('imagesOpacityValue');
  if (imagesOpacitySlider) {
    imagesOpacitySlider.addEventListener('input', (e) => {
      const opacity = e.target.value / 100;
      imagesOpacity = opacity;
      imagesOpacityValue.textContent = e.target.value + '%';
      // Apply to all image overlays
      imageOverlays.forEach(ov => {
        if (ov && ov.setOpacity) {
          ov.setOpacity(opacity);
        }
      });
      // Apply to orthophoto if it exists
      if (TIF_TILE_GROUP && TIF_TILE_GROUP.eachLayer) {
        TIF_TILE_GROUP.eachLayer(l => {
          if (l.setOpacity) {
            l.setOpacity(opacity);
          }
        });
      }
    });
  }

  if (typeof initOptimizeTabControls === 'function') {
    initOptimizeTabControls();
  }
  renderLegend();
}

document.addEventListener("DOMContentLoaded", async ()=>{
  // Check if there's a project ID in the URL query parameter (REQUIRED for project.html)
  const projectIdFromURL = getProjectIdFromURL();
  
  if (!projectIdFromURL) {
    // No projectId in URL - show error and redirect to home
    console.warn("project.html opened without projectId in URL");
    const container = document.querySelector("main") || document.body;
    container.innerHTML = `
      <div style="padding: 2rem; text-align: center; color: var(--error);">
        <h2>No Project Selected</h2>
        <p>Please select a project from the <a href="/index.html">home page</a></p>
      </div>
    `;
    return;
  }
  
  // ProjectId found in URL - activate it
  try {
    console.log("Activating project from URL:", projectIdFromURL);
    await activateProject(projectIdFromURL);
    // Save to localStorage for persistence
    saveSelectedProject(projectIdFromURL);
  } catch (err) {
    console.warn("Failed to activate project from URL:", projectIdFromURL, err);
    saveSelectedProject(null);
  }

  if(typeof loadFeatureFlags === 'function'){
    await loadFeatureFlags();
  }
  if(typeof applyFeatureFlags === 'function'){
    applyFeatureFlags();
  }

  setupUI();
  if(typeof applyFeatureFlags === 'function'){
    applyFeatureFlags();
  }
  initMap();
  if (typeof initMapOverlayUI === 'function') {
    initMapOverlayUI();
  }
  connectLogs();
  await Promise.all([loadDatasets(), loadModels(getSelectedBackend(), '#selModelFolder'), loadTrainedModels(), loadTrainingDatasets()]);
});

// ensure these helpers exist (from earlier step)
function makeStamp(prefix = "") {
  const d = new Date();
  const pad = (n) => String(n).padStart(2, "0");
  const ts = d.getFullYear().toString() + pad(d.getMonth()+1) + pad(d.getDate())
            + "_" + pad(d.getHours()) + pad(d.getMinutes()) + pad(d.getSeconds());
  return `${prefix}${ts}`;
}
function prefillAutofill(el, prefix = "") {
  if (!el) return;
  el.dataset.prefix = prefix;
  if (!el.value) { el.value = makeStamp(prefix); el.dataset.autofill = "1"; }
  el.addEventListener("input", () => { el.dataset.autofill = ""; }, { once: true });
}
function startAutoStamping() {
  const trainEl  = document.getElementById("inpModelName");    // train_
  const testEl   = document.getElementById("inpResultName");   // test_
  const uploadEl = document.getElementById("inpUploadName");   // test_ (upload modal)

  prefillAutofill(trainEl,  "train_");
  prefillAutofill(testEl,   "test_");
  prefillAutofill(uploadEl, "data_");

  const msToNextMinute = 60000 - (Date.now() % 60000);
  setTimeout(() => {
    const updateIfAuto = (el) => {
      if (el && el.dataset.autofill === "1") {
        el.value = makeStamp(el.dataset.prefix || "");
      }
    };
    updateIfAuto(trainEl);  updateIfAuto(testEl);  updateIfAuto(uploadEl);
    setInterval(() => { updateIfAuto(trainEl); updateIfAuto(testEl); updateIfAuto(uploadEl); }, 60000);
  }, msToNextMinute);
}
document.addEventListener("DOMContentLoaded", startAutoStamping);

// ================= Projects UI =================

async function initializeProjectsUI() {
  // Setup popstate listener for back button
  addPopstateListener();
  
  // Load projects
  await loadProjects();
  
  // Check if project ID is in URL (priority 1)
  const projectIdFromURL = getProjectIdFromURL();
  
  // Check if project ID is saved in localStorage (priority 2)
  const projectIdFromStorage = getSelectedProjectFromStorage();
  
  let project = null;
  let projectIdToLoad = projectIdFromURL || projectIdFromStorage;
  
  if (projectIdToLoad) {
    // Try to load project (from URL or localStorage)
    try {
      await activateProject(projectIdToLoad);
      project = activeProject;
      // Save to localStorage and URL so it persists across reloads
      saveSelectedProject(projectIdToLoad);
      // Update URL if projectId is not already there
      if (!projectIdFromURL && projectIdFromStorage) {
        window.history.replaceState({}, '', `/project.html?projectId=${projectIdToLoad}`);
      }
    } catch (err) {
      console.warn("Project not found:", projectIdToLoad, err);
      // Clear storage if project no longer exists
      saveSelectedProject(null);
    }
  }
  
  if (project) {
    activeProject = project;
    updateProjectUI();
    // Load the test tab by default when a project is active
    switchToTab("tab-test");
  } else {
    // No project selected - show projects page (homepage)
    activeProject = null;
    updateProjectUI();
  }
  
  // Render project cards
  renderProjectCards();
  
  // Setup event listeners
  setupProjectEventListeners();
}

function renderProjectCards() {
  const grid = $("#projectsGrid");
  const noMsg = $("#noProjectsMsg");
  
  if (!grid || !noMsg) return;
  
  // Reset search and sort to show all projects
  const searchInput = $("#projectSearch");
  if (searchInput) searchInput.value = "";
  
  if (allProjects.length === 0) {
    grid.innerHTML = "";
    noMsg.style.display = "block";
    return;
  }
  
  // Update original list in case it was modified
  allProjectsOriginal = [...allProjects];
  
  noMsg.style.display = "none";
  grid.innerHTML = allProjects.map(project => `
    <div class="projectCard ${activeProject?.id === project.id ? 'active' : ''}" data-project-id="${project.id}">
      <div class="projectThumbnail">📁</div>
      <div class="projectCardContent">
        <h3>${escapeHtml(project.name)}</h3>
        <div class="description">${escapeHtml(project.description || "No description")}</div>
        <div class="projectCardMeta">
          <span>${new Date(project.modified_at).toLocaleDateString()}</span>
        </div>
        <div class="projectCardActions">
          <a href="/project.html?projectId=${project.id}" class="open" style="display: inline-block; padding: 0.5rem 1rem; text-decoration: none; border-radius: 8px; background: var(--primary); color: white; cursor: pointer; border: none; font-size: 0.95rem;">Open</a>
          <button type="button" id="btnEditProject_${project.id}" class="secondary tiny">Edit</button>
          <button type="button" id="btnDeleteProject_${project.id}" class="secondary tiny" style="color:var(--err);">Delete</button>
        </div>
      </div>
    </div>
  `).join("");
  
  // Attach event listeners to cards
  $$(".projectCard").forEach(card => {
    card.addEventListener("click", (e) => {
      // Don't intercept clicks on buttons or links - let them handle themselves
      if (!e.target.closest("button") && !e.target.closest("a")) {
        const projectId = card.dataset.projectId;
        activateProjectAndSwitch(projectId);
      }
    });
  });
  
  allProjects.forEach(project => {
    const editBtn = $(`#btnEditProject_${project.id}`);
    const deleteBtn = $(`#btnDeleteProject_${project.id}`);
    
    if (editBtn) {
      editBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        showProjectModal(project);
      });
    }
    
    if (deleteBtn) {
      deleteBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        if (confirm(`Delete project "${project.name}"? (Files will not be deleted.)`)) {
          deleteProject(project.id).then(() => {
            renderProjectCards();
          }).catch(err => {
            alert("Error deleting project: " + err.message);
          });
        }
      });
    }
  });
}

function setupProjectEventListeners() {
  const newProjectBtn = $("#btnNewProject");
  const createFirstLink = $("#createFirstProject");
  const modalCancel = $("#projectModalCancel");
  const modalSave = $("#projectModalSave");
  const projectSearch = $("#projectSearch");
  const projectSort = $("#projectSort");
  
  if (newProjectBtn) {
    newProjectBtn.addEventListener("click", () => {
      showProjectModal(null);
    });
  }
  
  if (createFirstLink) {
    createFirstLink.addEventListener("click", (e) => {
      e.preventDefault();
      showProjectModal(null);
    });
  }
  
  if (modalCancel) {
    modalCancel.addEventListener("click", closeProjectModal);
  }
  
  if (modalSave) {
    modalSave.addEventListener("click", saveProject);
  }
  
  if (projectSearch) {
    projectSearch.addEventListener("input", () => {
      applyProjectFilters();
    });
  }
  
  if (projectSort) {
    projectSort.addEventListener("change", () => {
      applyProjectFilters();
    });
  }
}

function showProjectModal(project = null) {
  const modal = $("#projectModal");
  const title = $("#projectModalTitle");
  const nameInput = $("#projectName");
  const descInput = $("#projectDescription");
  const pathInput = $("#projectPath");
  const saveBtn = $("#projectModalSave");
  
  if (!modal) return;
  
  if (project) {
    title.textContent = "Edit Project";
    nameInput.value = project.name;
    descInput.value = project.description || "";
    pathInput.value = project.root_path;
    pathInput.disabled = true;
    saveBtn.textContent = "Save Changes";
  } else {
    title.textContent = "New Project";
    nameInput.value = "";
    descInput.value = "";
    pathInput.value = "";
    pathInput.disabled = false;
    saveBtn.textContent = "Create Project";
  }
  
  modal.classList.add("show");
  nameInput.focus();
  
  // Store project ID for editing
  modal.dataset.projectId = project?.id || "";
}

function closeProjectModal() {
  const modal = $("#projectModal");
  if (modal) {
    modal.classList.remove("show");
    delete modal.dataset.projectId;
  }
}

async function saveProject() {
  const nameInput = $("#projectName");
  const descInput = $("#projectDescription");
  const pathInput = $("#projectPath");
  const modal = $("#projectModal");
  
  const name = nameInput.value.trim();
  const desc = descInput.value.trim();
  const path = pathInput.value.trim();
  
  if (!name) {
    alert("Please enter a project name");
    return;
  }
  
  const projectId = modal.dataset.projectId;
  
  try {
    if (projectId) {
      // Edit mode - update project
      const formData = new FormData();
      formData.append("name", name);
      formData.append("description", desc);
      
      const resp = await fetch(`${api.projects}/${projectId}`, {
        method: "PUT",
        body: formData
      });
      if (!resp.ok) throw new Error("Failed to update project");
    } else {
      // Create mode
      if (!path) {
        alert("Please specify a project directory");
        return;
      }
      await createProject(name, desc, path);
    }
    closeProjectModal();
    await loadProjects();
    renderProjectCards();
  } catch (err) {
    alert("Error: " + err.message);
  }
}

// Keep full list for filtering
let allProjectsOriginal = [];

function applyProjectFilters() {
  const searchTerm = ($("#projectSearch")?.value || "").toLowerCase();
  const sortBy = $("#projectSort")?.value || "modified";
  
  let filtered = allProjectsOriginal.filter(p =>
    p.name.toLowerCase().includes(searchTerm) ||
    p.description?.toLowerCase().includes(searchTerm)
  );
  
  // Sort
  filtered.sort((a, b) => {
    switch (sortBy) {
      case "name":
        return a.name.localeCompare(b.name);
      case "created":
        return new Date(b.created_at) - new Date(a.created_at);
      case "modified":
      default:
        return new Date(b.modified_at) - new Date(a.modified_at);
    }
  });
  
  // Render filtered list without modifying allProjects
  const grid = $("#projectsGrid");
  const noMsg = $("#noProjectsMsg");
  
  if (!grid || !noMsg) return;
  
  if (filtered.length === 0) {
    grid.innerHTML = "";
    noMsg.style.display = "block";
    return;
  }
  
  noMsg.style.display = "none";
  grid.innerHTML = filtered.map(project => `
    <div class="projectCard ${activeProject?.id === project.id ? 'active' : ''}" data-project-id="${project.id}">
      <div class="projectThumbnail">📁</div>
      <div class="projectCardContent">
        <h3>${escapeHtml(project.name)}</h3>
        <div class="description">${escapeHtml(project.description || "No description")}</div>
        <div class="projectCardMeta">
          <span>${new Date(project.modified_at).toLocaleDateString()}</span>
        </div>
        <div class="projectCardActions">
          <a href="/project.html?projectId=${project.id}" class="open" style="display: inline-block; padding: 0.5rem 1rem; text-decoration: none; border-radius: 8px; background: var(--primary); color: white; cursor: pointer; border: none; font-size: 0.95rem;">Open</a>
          <button type="button" id="btnEditProject_${project.id}" class="secondary tiny">Edit</button>
          <button type="button" id="btnDeleteProject_${project.id}" class="secondary tiny" style="color:var(--err);">Delete</button>
        </div>
      </div>
    </div>
  `).join("");
  
  // Attach event listeners (duplicated from renderProjectCards for filtered display)
  $$(".projectCard").forEach(card => {
    card.addEventListener("click", (e) => {
      // Don't intercept clicks on buttons or links - let them handle themselves
      if (!e.target.closest("button") && !e.target.closest("a")) {
        const projectId = card.dataset.projectId;
        activateProjectAndSwitch(projectId);
      }
    });
  });
  
  // Only attach listeners to Edit and Delete buttons
  filtered.forEach(project => {
    const editBtn = $(`#btnEditProject_${project.id}`);
    const deleteBtn = $(`#btnDeleteProject_${project.id}`);
    
    if (editBtn) {
      editBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        showProjectModal(project);
      });
    }
    
    if (deleteBtn) {
      deleteBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        if (confirm(`Delete project "${project.name}"? (Files will not be deleted.)`)) {
          deleteProject(project.id).then(() => {
            applyProjectFilters();  // Refresh the filtered display
          }).catch(err => {
            alert("Error deleting project: " + err.message);
          });
        }
      });
    }
  });
}

async function activateProjectAndSwitch(projectId) {
  if (!projectId) {
    console.warn("No project ID provided");
    return;
  }
  
  console.log("activateProjectAndSwitch called with projectId:", projectId);
  
  // Save to localStorage immediately
  saveSelectedProject(projectId);
  
  // Try to activate on backend (but don't block if it fails)
  try {
    await activateProject(projectId);
    console.log("Project activated on backend");
  } catch (err) {
    console.warn("Failed to activate project on backend, but continuing:", err);
  }
  
  // ALWAYS navigate to the project page, even if backend activation failed
  console.log("Navigating to project.html?projectId=" + projectId);
  navigateToProject(projectId);
}

// Initialize projects on page load (only on index.html, not project.html)
document.addEventListener("DOMContentLoaded", () => {
  // Check if projects grid exists (means we're on index.html)
  if (document.getElementById("projectsGrid")) {
    initializeProjectsUI();
  }
});




// --- helpers to render any JSON as a nice key/value list ---
function flattenForList(obj, prefix = "") {
  const out = {};
  if (!obj || typeof obj !== "object") return out;
  for (const [k, v] of Object.entries(obj)) {
    const key = prefix ? `${prefix}.${k}` : k;
    if (v && typeof v === "object" && !Array.isArray(v)) {
      Object.assign(out, flattenForList(v, key));
    } else {
      out[key] = v;
    }
  }
  return out;
}

function renderJsonList(obj) {
  const flat = flattenForList(obj);
  const rows = Object.entries(flat)
    .sort(([a],[b]) => a.localeCompare(b))
    .map(([k, v]) => {
      const val = (typeof v === "string") ? v : JSON.stringify(v);
      return `<li><span class="key">${escapeHtml(k)}</span><span class="val">${escapeHtml(val ?? '')}</span></li>`;
    })
    .join("");
  return `<ul class="kv-list">${rows}</ul>`;
}

// --- render the two cards (metrics + model meta) inside #resultsInfo ---
function renderResultsInfo(metrics, modelMeta) {
  const root = document.getElementById("resultsInfo");
  if (!root) return;

  root.innerHTML = `
    <div class="info-grid">
      ${metrics ? `
        <section class="info-card">
          <h4>Test Metrics</h4>
          ${renderJsonList(metrics)}
        </section>` : ``}
      ${modelMeta ? `
        <section class="info-card">
          <h4>Model Meta</h4>
          ${renderJsonList(modelMeta)}
        </section>` : ``}
    </div>
  `;
  root.hidden = !(metrics || modelMeta);
}

// Try to pull "train_YYYYMMDD_HHMMSS" out of model_name if we need a fallback
function deriveRunNameFromModelName(name) {
  // const m = /train_\d{8}_\d{6}/.exec(String(name || ""));
  // return m ? m[0] : null;
  return name
}

// Fetch metrics + model meta for a session, then render
async function loadResultsInfo(sessionName) {
  const infoEl = document.getElementById("resultsInfo");
  const predictions_title = document.getElementById("predictionsTitle");
  if (infoEl) { infoEl.hidden = true; infoEl.innerHTML = ""; }

  let metrics = null, meta = null;

  // 1) metrics (required for the panel to be useful)
  try {
    const r = await fetch(`/api/results/${encodeURIComponent(sessionName)}/metrics`, { cache: "no-store" });
    if (r.ok) metrics = await r.json();
  } catch {}

  // 3) fallback: derive run name from metrics.model_name → /api/runs/{run}/meta
  if (!meta && metrics?.model_name) {
    const runName = deriveRunNameFromModelName(metrics.model_name);
    if (runName) {
      try {
        const r2 = await fetch(`/api/runs/${encodeURIComponent(runName)}/meta`, { cache: "no-store" });
        if (r2.ok) meta = await r2.json();
      } catch {}
    }
  }

  renderResultsInfo(metrics, meta);
  if(predictions_title){
    predictions_title.textContent = metrics?.total_detections != null
      ? `Predictions: ${metrics.total_detections}`
      : 'Predictions';
  }
}
