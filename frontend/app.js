/* global L */

const api = {
  datasets: "/api/test_datasets",
  models: "/api/models",
  upload: "/api/test_upload",
  testRun: "/api/test_run",
  train: "/api/train",
  cancel: "/api/cancel",
  logs: "/api/logs",
  sessions: "/api/sessions",
  sessionSummary: "/api/session_summary",
  sessionTiles: "/api/session_tiles"
};

let MAP, baseLayers, overlayRegistry = {};
let imagesLayerGroup = null;           // holds all image markers/overlays
let imageMarkers = new Map();
;          // id -> L.Marker or L.ImageOverlay
let geojsonLayer = null;
let imageMarkersLayer = null;
let tileLayers = [];
let currentSession = null;
let styleTarget = null;
let layerMenuState = { name: null, info: null };
let testAbort = null;

// runtime caches & UI flags
let modelsCache = {};                // name -> model metadata returned by /api/models
let userToggledThermalTrain = false; // whether user manually toggled the train thermal checkbox
let userToggledThermalTest = false;  // whether user manually toggled the test thermal checkbox

// catalog & runtime overlays for photos
let imageCatalog = [];              // [{ id, name, url, bounds, on }]
let imageOverlays = new Map();      // id -> L.ImageOverlay

// last loaded images.geojson and session info (used to reapply overrides)
let lastLoadedImagesGJ = null;
let lastLoadedSessionName = null;
let lastLoadedImagesUrl = null;
let lastLoadedSessionSummary = null;

// camera position overrides loaded from an uploaded WebODM camera-positions JSON
// keyed by normalized basename (no extension), value: { lon, lat, alt }
let cameraPositionOverrides = {};

// fallback sizing if images.geojson has only a Point
const DEFAULT_IMAGE_HALF_SIZE_M = 25; // ~25 m half-width/half-height (adjust if needed)

// --- TIF raster globals ---
let TIF_TILE_GROUP = null;   // Leaflet layerGroup that holds the ZXY tile layers
let TIF_TILE_LAYERS = [];    // underlying L.tileLayer instances

function removeTifTiles(){
  if (TIF_TILE_GROUP) {
    try { MAP.removeLayer(TIF_TILE_GROUP); } catch {}
  }
  TIF_TILE_GROUP = null;
  TIF_TILE_LAYERS = [];
}



// ---------- helpers ----------
const $ = sel => document.querySelector(sel);
const $$ = sel => Array.from(document.querySelectorAll(sel));
function setHidden(el, hidden=true){ if(!el) return; hidden ? el.style.display = 'none' : el.style.display = 'block' ; }
function setText(sel, txt){ const el=$(sel); if(el) el.textContent = txt; }
function escapeHtml(s){ return (s||"").replace(/[&<>"']/g, m=>({ "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;" }[m])); }
function closeBtn(){ return `<button class="iconBtn alertClose" aria-label="Close">×</button>`; }

// convert meters to degrees at a given latitude
function metersToDeg(lat, meters){
  const metersPerDegLat = 111320;
  const metersPerDegLon = 111320 * Math.cos(lat * Math.PI / 180);
  return { dLat: meters / metersPerDegLat, dLon: meters / metersPerDegLon };
}

// remove all currently shown image overlays
function clearImageOverlays(){
  for (const ov of imageOverlays.values()){
    try { imagesLayerGroup.removeLayer(ov); } catch(_){}
  }
  imageOverlays.clear();
}

// turn one image overlay on/off by id
function toggleImageOverlay(id, on){
  const rec = imageCatalog.find(x => x.id === id);
  if (!rec) return;

  let ov = imageOverlays.get(id);
  if (on){
    if (!ov){
      // create overlay using provided bounds (may be conservative bbox of corners)
      ov = L.imageOverlay(rec.url, rec.bounds, { opacity: 0.85, interactive: false });
      imageOverlays.set(id, ov);
    }
    ov.addTo(imagesLayerGroup);
    rec.on = true;

    // No client-side rotation or homography is applied. Backend supplies
    // properly-oriented rotated images when available, and the frontend
    // should simply overlay the provided PNG pixels at the geospatial bounds.

  } else {
    if (ov){ try{ imagesLayerGroup.removeLayer(ov); }catch(_){ } }
    rec.on = false;
  }
}

// turn ALL images on/off (used by Show all / Hide all buttons)
function setAllImageOverlays(on){
  for (const rec of imageCatalog){
    toggleImageOverlay(rec.id, on);
    const sel = `.imgToggle[data-id="${CSS.escape(rec.id)}"]`;
    const cb = document.querySelector(sel);
    if (cb) cb.checked = on;
  }
}

// build the sidebar list from imageCatalog
function renderImagesList(){
  const ul = document.getElementById('imagesList');
  if (!ul) return;

  if (!imageCatalog.length){
    ul.innerHTML = `<li class="dim">No geolocated images in this session</li>`;
    return;
  }

  ul.innerHTML = imageCatalog.map(rec => `
    <li>
      <label class="chk">
        <input type="checkbox" class="imgToggle" data-id="${escapeHtml(rec.id)}">
        <span>${escapeHtml(rec.name)}</span>
      </label>
      <button class="iconDots openImg" data-id="${escapeHtml(rec.id)}" title="Open image">🔍</button>
    </li>
  `).join('');
}

// document.addEventListener('click', (e)=>{
//   const li = e.target.closest('#layerMenu li');
//   if (!li) return;

//   const menu = li.closest('#layerMenu');
//   const key  = menu?.dataset?.key;
//   const rec  = key ? overlayRegistry[key] : null;
//   if (!rec || !rec.layer) { closeLayerMenu(); return; }

//   const action = li.dataset.action;

//   if (action === 'zoom') {
//     const b = computeLayerBounds(rec.layer);
//     if (b && b.isValid && b.isValid()) {
//       MAP.fitBounds(b.pad(0.2));
//     }
//     closeLayerMenu();
//     return;
//   }

//   if (action === 'style') {
//     // Open your style modal and prefill from rec.style (fallbacks provided)
//     styleTarget = { name: key, info: rec };

//     const st = rec.style || {};
//     $('#stColor').value    = toHex(st.color || '#0ea5e9');
//     $('#stWidth').value    = st.weight ?? 1;
//     $('#stOpacity').value  = st.opacity ?? 1;
//     $('#fiColor').value    = toHex(st.fillColor || st.color || '#0ea5e9');
//     $('#fiOpacity').value  = st.fillOpacity ?? 0.25;

//     $('#styleModal').classList.remove('hidden');
//     closeLayerMenu();
//     return;
//   }
// });


function applyVectorColor(layer, color){
  // try to recolor vector layers
  if (layer.setStyle) {
    layer.setStyle({ color, fillColor: color });
  }
  if (layer.eachLayer){
    layer.eachLayer(l => applyVectorColor(l, color));
  }
}

function applyStyleLive(){
  if (!styleTarget || !styleTarget.info) return;

  const st = {
    color:     document.getElementById('stColor')?.value || '#0ea5e9',
    weight:    parseFloat(document.getElementById('stWidth')?.value || '1'),
    opacity:   parseFloat(document.getElementById('stOpacity')?.value || '1'),
    fillColor: document.getElementById('fiColor')?.value || '#0ea5e9',
    fillOpacity: parseFloat(document.getElementById('fiOpacity')?.value || '0.25'),
  };

  styleTarget.info.style = st;

  if (styleTarget.info.type === 'geojson' && styleTarget.info.layer?.setStyle) {
    styleTarget.info.layer.setStyle(st);
    renderLegend();
  } else if (styleTarget.info.type === 'raster') {
    const op = Math.max(0, Math.min(1, st.opacity));
    for (const l of tileLayers) l.setOpacity(op);
  }
}



function closeLayerMenu(){
  const m = document.getElementById('layerMenu');
  if (m) m.remove();
}

function openLayerMenu(btn){
  closeLayerMenu();

  const key = btn.dataset.key;
  const rec = overlayRegistry[key];
  if (!rec || !rec.layer) return;

  const menu = document.createElement('div');
  menu.id = 'layerMenu';
  menu.className = 'layerMenu';
  menu.dataset.key = key;

  menu.innerHTML = `
    <ul>
      <li data-action="zoom">Zoom to layer</li>
      <li data-action="style">Style…</li>
    </ul>
  `;

  document.body.appendChild(menu);

  // --- position: prefer opening to the LEFT of the button ---
  const r  = btn.getBoundingClientRect();
  const vw = document.documentElement.clientWidth;
  const pad = 8;                          // viewport padding
  const mw = menu.offsetWidth || 220;     // menu width (fallback)

  const leftSpace  = r.left - pad;
  const rightSpace = vw - r.right - pad;

  let left;
  if (leftSpace >= mw) {
    // enough room on the left → align menu's right edge with button's right
    left = Math.round(r.right - mw);
  } else if (rightSpace >= mw) {
    // not enough left, open to the right
    left = Math.round(r.left);
  } else {
    // clamp inside viewport if both sides are tight
    left = Math.max(pad, Math.min(Math.round(r.right - mw), vw - mw - pad));
  }

  menu.style.left = `${left}px`;
  menu.style.top  = `${Math.round(r.bottom + 6)}px`;
}


// Close the menu when clicking anywhere outside it (but not on the ⋮ button)
document.addEventListener('mousedown', (e)=>{
  const m = document.getElementById('layerMenu');
  if (!m) return;
  if (!m.contains(e.target) && !e.target.closest('.layerRow .more')) {
    closeLayerMenu();
  }
});


document.addEventListener('click', (e)=>{
  const li = e.target.closest('#layerMenu li');
  if (!li) return;

  const menu = li.closest('#layerMenu');
  const key  = menu?.dataset?.key;
  const rec  = key ? overlayRegistry[key] : null;
  if (!rec || !rec.layer) { closeLayerMenu(); return; }

  const action = li.dataset.action;

  if (action === 'zoom') {
    const b = computeLayerBounds(rec.layer);
    if (b && b.isValid && b.isValid()) {
      MAP.fitBounds(b.pad(0.2));
    }
    closeLayerMenu();
    return;
  }

  if (action === 'style') {
  if (action === 'style') {
  styleTarget = { name: key, info: rec };

  const stSelBlk = document.getElementById('stColorByBlock');
  const stSel    = document.getElementById('stColorBy');
  const stCat    = document.getElementById('stCatList');

  // Only Anomalies supports styling
  if (key !== "Anomalies" || !rec?.data){
    stSelBlk.classList.add('hidden');
    stCat.classList.remove('hidden');
    stCat.innerHTML = `<div class="muted">Styling not available for this layer.</div>`;
    document.getElementById('styleModal').classList.remove('hidden');
    closeLayerMenu();
    return;
  }

  // Show Category UI
  stSelBlk.classList.remove('hidden');
  stCat.classList.remove('hidden');

  // Build property list (with "None")
  const props = new Set(['class_name','class_id','score']);
  try{
    const f0 = rec.data.features.find(f => f?.properties) || null;
    if (f0) Object.keys(f0.properties).forEach(k => props.add(k));
  }catch(_){}

  // Helper: pick a sensible default
  const CATEGORY_NONE = '__none__';
  function pickDefaultProp(){
    const tryProps = ['class_name','class_id','score', ...props];
    for (const p of tryProps){
      if (!p || p === CATEGORY_NONE) continue;
      const u = uniqueValuesFromGJ(rec.data, p);
      if (u.length >= 1 && u.length <= 10) return p;
    }
    return CATEGORY_NONE;
  }

  const currentProp = rec.categorical?.prop ?? pickDefaultProp();

  // Fill dropdown
  stSel.innerHTML = '';
  const optNone = document.createElement('option');
  optNone.value = CATEGORY_NONE; optNone.textContent = 'None';
  stSel.appendChild(optNone);
  Array.from(props).forEach(k=>{
    const o = document.createElement('option');
    o.value = k; o.textContent = k;
    if (k === currentProp) o.selected = true;
    stSel.appendChild(o);
  });

  // Build editor for selected prop
  const choose = (val)=>{
    rebuildCategoryEditors(val);
  };
  choose(currentProp);
  stSel.onchange = ()=> choose(stSel.value);

  document.getElementById('styleModal').classList.remove('hidden');
  closeLayerMenu();
  return;
}

}

});



function appendLog(line){
  const pane = $("#logStream");
  pane.textContent += (pane.textContent ? "\n" : "") + line;
  pane.scrollTop = pane.scrollHeight;
}
function appendMiniLog(sel, line){
  const pane = $(sel);
  if(!pane) return;
  pane.textContent += (pane.textContent ? "\n" : "") + line;
  pane.scrollTop = pane.scrollHeight;
}
function clearAlerts(prefix){
  for(const id of [`#${prefix}Warn`, `#${prefix}Err`, `#${prefix}Ok`]){
    const el = $(id);
    if(el){ el.innerHTML = ""; setHidden(el,true); }
  }
}
function ok(prefix, msg){ const el=$(`#${prefix}Ok`); if(el){ el.innerHTML=closeBtn()+escapeHtml(msg); setHidden(el,false);} }
function err(prefix, msg){ const el=$(`#${prefix}Err`); if(el){ el.innerHTML=closeBtn()+escapeHtml(msg); setHidden(el,false);} }
function warn(prefix, msg){ const el=$(`#${prefix}Warn`); if(el){ el.innerHTML=closeBtn()+escapeHtml(msg); setHidden(el,false);} }
function wireAlertClose(){ $$(".alert").forEach(el=>{ el.addEventListener("click",(e)=>{ if(e.target.classList.contains("alertClose")) setHidden(el,true); }); }); }

function populateFolders(list){
  const sel = $("#selTestFolder");
  sel.innerHTML = "";
  list.forEach(d => {
    const o = document.createElement("option");
    o.value = d.name;
    o.textContent = `${d.name} (${d.count})`;
    sel.appendChild(o);
  });
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
      if(tl.includes('mask')) return 'maskrcnn';
      // fallback: remove spaces/underscores and non-alphanumerics for compactness
      return tl.replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '');
    }

    if (m.model_name) {
      const mt = prettyModelType(m.model_type || '');
      // try to pick up num_classes from several possible metadata locations
      const nc = (m.num_classes != null) ? Number(m.num_classes)
                 : (m.model_meta && m.model_meta.num_classes != null) ? Number(m.model_meta.num_classes)
                 : null;
      const nch = m.channel_count ? `-${m.channel_count}ch` : '';
      const ncs = (nc != null) ? `-${nc}cls` : '';
      return `${m.model_name}${mt ? `_${mt}` : ''}${nch}${ncs}`;
    }
    // Fallback: construct a compact run-like label: <run>_<backend>_<input_mode>_<nch>
    const parts = [m.name];
    if (m.backend) parts.push(String(m.backend));
  const mtf = m.model_type ? prettyModelType(m.model_type) : '';
    if (mtf) parts.push(mtf);
    let out = parts.join('_');
  if (m.channel_count) out = `${out}-${m.channel_count}ch`;
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
  // according to the selected model's channel_count unless the user explicitly toggled it.
  if (selSelector === '#selModelFolder'){
    const selected = sel.value || sel.options[0]?.value;
    const m = modelsCache[selected];
    const defaultRequiresThermal = (m && m.channel_count) ? (
      Number(m.channel_count) === 4 || (Number(m.channel_count) === 3 && (!!(m && m.thermal_only) || !!(m && m.thermal_used)))
    ) : false;
    const chkTest = document.getElementById('chkUseThermalTest');
    // only override the checkbox if the user hasn't manually toggled it
    if (chkTest && !userToggledThermalTest){
      try { chkTest.checked = !!defaultRequiresThermal; } catch(_){}
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

  if(activeTab === 'tab-train'){
    if(selTrain && selTrain.value) return selTrain.value;
    if(selGlobal && selGlobal.value) return selGlobal.value;
    return 'detectron';
  }

  if(activeTab === 'tab-test'){
    if(selTest && selTest.value) return selTest.value;
    if(selGlobal && selGlobal.value) return selGlobal.value;
    return 'detectron';
  }

  // default fallback (prefer train selector if present)
  if(selTrain && selTrain.value) return selTrain.value;
  if(selTest && selTest.value) return selTest.value;
  if(selGlobal && selGlobal.value) return selGlobal.value;
  return 'detectron';
}
function getYoloOptions(){
  return {
    family: $("#selYoloFamily") ? $("#selYoloFamily").value : 'v8',
    seg: !!($("#chkYoloSeg") && $("#chkYoloSeg").checked)
  };
}

// ---------- tabs ----------
function setupTabs(){
  $$(".tabs button").forEach(btn=>{
    btn.addEventListener("click", ()=>{
      $$(".tabs button").forEach(b=>b.classList.remove("active"));
      btn.classList.add("active");
      const id = btn.dataset.tab;
      $$(".tabPanel").forEach(p=>p.classList.remove("active"));
      $(`#${id}`).classList.add("active");
      if(id === "tab-map" && MAP){ setTimeout(()=>MAP.invalidateSize(), 30); }
      if(id === "tab-logs"){
        const pane = $("#logStream");
        pane.scrollTop = pane.scrollHeight;
      }
    });
  });
}
function switchToTab(tabId){
  $$(".tabs button").forEach(b=>b.classList.toggle("active", b.dataset.tab === tabId));
  $$(".tabPanel").forEach(p=>p.classList.toggle("active", p.id === tabId));
  if(tabId === "tab-map" && MAP){ setTimeout(()=>MAP.invalidateSize(), 30); }
}

// ---------- datasets/models/sessions ----------
async function loadDatasets(){
  const res = await fetch(api.datasets);
  const js = await res.json();
  if(js.ok){ populateFolders(js.datasets); }
}
// Load models, optionally filtered by backend (e.g., ?backend=yolo)
// targetSel - optional selector string for which <select> to populate (defaults to '#selModelFolder')
async function loadModels(backend, targetSel = '#selModelFolder'){
  try{
    let url = api.models;
    if(backend) url = `${api.models}?backend=${encodeURIComponent(backend)}`;
    const res = await fetch(url);
    const js = await res.json();
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
async function loadSessions(selectLatest=true){
  const res = await fetch(api.sessions);
  const js = await res.json();
  if(!js.ok) return;
  const sel1 = $("#selResults");
  const sel2 = $("#selMapSession");
  sel1.innerHTML = ""; sel2.innerHTML = "";
  js.sessions.forEach(s=>{
    const nm = s.name.split("/").pop();
    for(const sel of [sel1, sel2]){
      const o = document.createElement("option");
      o.value = s.name;
      o.textContent = nm;
      sel.appendChild(o);
    }
  });
  if(selectLatest && js.sessions.length){
    const latest = js.sessions[0].name;
    sel1.value = latest; sel2.value = latest;
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
function openUploadModal(){ $("#uploadModal").classList.remove("hidden"); }
function closeUploadModal(){ $("#uploadModal").classList.add("hidden"); }
function resetUploadProgress(){
  $("#testUploadBar").style.width = "0%";
  setText("#testUploadText","Uploading… 0%");
  setHidden($("#testUploadProgress"), true);
}

async function startUpload(){
  const input = $("#filesTest");
  if(!input.files || !input.files.length){
    warn("test","Choose at least one file (.zip or images).");
    return;
  }
  clearAlerts("test");
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
          closeUploadModal();
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
  $("#trainMiniLog").textContent = "";
  const useThermal = $("#chkUseThermalTrain").checked;
  const iters = parseInt($("#inpIters").value || "500", 10);
  const lr = parseFloat($("#inpLR").value || "0.002");
  const batch = parseInt($("#inpBatch").value || "4", 10);
  const modelName = (document.getElementById("inpModelName")?.value || "").trim() || makeStamp();
  const modelType = $("#selModelType").value;

  setHidden($("#spinTrain"), false);
  setText("#trainStatus","Submitting training job…");

  const fd = new FormData();
  fd.append("use_thermal", useThermal ? "true" : "false");
  fd.append("max_iter", String(iters));
  fd.append("base_lr", String(lr));
  fd.append("ims_per_batch", String(batch));
  fd.append("model_name", String(modelName));
  fd.append("model_type", modelType);
  const backend = getSelectedBackend();
  fd.append("backend", backend);
  // include requested channel count for training so backend can prepare appropriate inputs
  const channelCount = (document.getElementById('selChannelCountTrain') || { value: '3' }).value;
  fd.append('channel_count', String(channelCount));
  if(backend === 'yolo'){
    const yo = getYoloOptions();
    fd.append('yolo_family', yo.family);
    fd.append('yolo_seg', yo.seg ? 'true' : 'false');
  }


  const res = await fetch(api.train, { method:"POST", body:fd });
  const js = await res.json();

  if(js.ok){
    ok("train","Training started.");
    setText("#trainStatus","Training started…");
  }else{
    err("train","Failed to start training.");
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
  clearAlerts("test"); wireAlertClose();
  $("#testMiniLog").textContent = "";
  const ds = getSelectedDataset();
  if(!ds){
    warn("test","Please select a dataset.");
    return;
  }
  const model = getSelectedModel();
  // Decide channels based solely on the selected model's metadata.
  // The frontend will send the model's expected channel_count to the server so the backend
  // can decide whether to decode/use the thermal band. If model metadata is missing,
  // default to 3 channels (RGB) and therefore no thermal decoding.
  const selectedModelName = model;
  const mmeta = selectedModelName ? modelsCache[selectedModelName] : null;
  let modelChannelCount = (mmeta && mmeta.channel_count) ? Number(mmeta.channel_count) : 3;
  // Map legacy '1' markers to 3 for the UI (we no longer use 1-channel models).
  if (modelChannelCount === 1) modelChannelCount = 3;
  // use_thermal true when model expects 4 channels (RGB+thermal) or when a
  // 3-channel model was prepared as thermal-only (thermal_only flag). Do not
  // treat '1' specially anymore.
  const useThermal = (modelChannelCount === 4 || (modelChannelCount === 3 && (!!(mmeta && mmeta.thermal_only) || !!(mmeta && mmeta.thermal_used))));
  const resultName = (document.getElementById("inpResultName")?.value || "").trim() || makeStamp();
  const testThreshold = (document.getElementById("testThreshold")?.value);

  setHidden($("#spinTest"), false);
  setText("#testStatus","Running inference…");

  const fd = new FormData();
  fd.append("dataset", ds);
  if(model) fd.append("model", model);
  fd.append("use_thermal", useThermal ? "true":"false");
  // inform backend of the expected model channel count so it can prepare inputs correctly
  // inform backend of the expected model channel count so it can prepare inputs correctly
  fd.append('channel_count', String(modelChannelCount));
  // Note: frontend no longer offers the "extract thermal as RGB" option; backend will
  // only attempt decoding when the selected model declares RGB+thermal input.
  fd.append("result_name", resultName);
  fd.append("test_threshold", testThreshold);
  const backend = getSelectedBackend();
  fd.append('backend', backend);
  // Attach optional camera references (Agisoft references.txt / CSV / TSV) when provided
  const camInput = document.getElementById('fileCameraRefs');
  if (camInput && camInput.files && camInput.files.length > 0){
    const f = camInput.files[0];
    try{
      // basic extension check
      const ext = (f.name || '').split('.').pop().toLowerCase();
      if (!['txt','csv','tsv'].includes(ext)){
        if (!confirm('Selected camera references file does not have .txt/.csv/.tsv extension. Continue anyway?')){
          setHidden($("#spinTest"), true);
          return;
        }
      }

      // quick content sniff: read first chunk and check for XML start or delimiters
      const sample = (await f.text()).slice(0, 4096);
      const sTrim = sample.trimLeft();
      if (sTrim.startsWith('<')){
        alert('The uploaded file looks like XML. The backend expects Agisoft reference tables (references.txt / CSV / TSV). Please upload the table export instead.');
        setHidden($("#spinTest"), true);
        return;
      }

      const header = sample.split(/\r?\n/)[0] || '';
      const looksLikeCsv = header.includes(',') || header.includes('\t') || header.split(/\s+/).length > 1;
      if (!looksLikeCsv){
        if (!confirm('Could not detect CSV/TSV-like headers in the uploaded camera references file. Continue anyway?')){
          setHidden($("#spinTest"), true);
          return;
        }
      }

      fd.append('cameras', f);
    }catch(e){
      console.warn('Failed to attach cameras file', e);
      try{ fd.append('cameras', f); }catch(_){ }
    }
  }
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

    if(!js.ok) throw new Error("test failed");

    // If backend reports which channel configuration it actually used, show it briefly
    try{
      if (js.used_channel_count != null){
        const uc = Number(js.used_channel_count);
        // If server indicates thermal_only (new behavior), show that explicitly.
        const thermalOnly = !!js.thermal_only;
        let msg = `Run used: ${uc}ch`;
        if (thermalOnly) msg += ' (thermal-only)';
        else if (uc === 3) msg = 'Run used: 3ch (RGB only)';
        else if (uc === 4) msg = 'Run used: 4ch (RGB + thermal)';
        ok("test", msg);
      }
    }catch(_){ }

    // console.log(js)
    currentSession = js.session;
    // Add a more explicit status line with total predictions
    ok("test", "Testing completed.");
    setText("#testStatus", `Inference complete. ${totalPreds} predictions.`);

    // load into map & results
    await applySessionToMap(currentSession);
    renderResultsGrid(js.manifest && js.manifest.length ? js.manifest : pairThumbs(js.assets));
    loadResultsInfo(currentSession);
    
    await loadSessions(true);
    $("#selResults").value = currentSession;
    $("#selMapSession").value = currentSession;
    switchToTab("tab-results");
  }catch(ex){
    if(ex.name === "AbortError"){
      warn("test","Test canceled.");
    }else{
      err("test", String(ex));
    }
  }finally{
    setHidden($("#spinTest"), true);
    testAbort = null;
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
  // console.log(js)
  renderResultsGrid(js.manifest && js.manifest.length ? js.manifest : pairThumbs(js.assets));
  loadResultsInfo(currentSession);
}

function pairThumbs(assets){
  const mapThumb = new Map((assets.thumbs||[]).map(u=>[u.split("/").pop(), u]));
  const out = [];
  for(const ov of (assets.overlays||[])){
    const fn = ov.split("/").pop();
    out.push({ file: fn, overlay: ov, thumb: mapThumb.get(fn)||ov });
  }
  return out;
}
function renderResultsGrid(manifest){
  const grid = $("#resultsGrid");
  grid.innerHTML = "";
  // console.log(manifest)
  // console.log(manifest.length)
  
  if(!manifest || !manifest.length){
    grid.innerHTML = `<div class="muted">No overlays generated.</div>`;
    // return;
  }
  // console.log(manifest)
  manifest.forEach((item, idx)=>{
    // console.log("inside loop")
    const div = document.createElement("div");
    div.className = "thumb";
    div.innerHTML = `
      <img src="${item.thumb}" alt="${item.file}">
      <div class="meta" title="${item.file}">${item.file}</div>
    `;
    div.addEventListener("click", ()=>{
      _openLightboxWithGallery(manifest, idx);
    });
    grid.appendChild(div);
  });

}

// ---------- map ----------
function initMap(){
  const street = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", { maxZoom: 22, attribution: "&copy; OpenStreetMap" });
  const sat = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", { maxZoom: 22, attribution: "&copy; Esri" });

  MAP = L.map("map", { layers: [street] });
  baseLayers = { "Street": street, "Satellite": sat };
  L.control.layers(baseLayers, {}, { position: "topleft" }).addTo(MAP);
  MAP.setView([0,0], 2);

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




function rebuildCategoryEditors(prop){
  const rec  = overlayRegistry["Anomalies"];
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
  const useNone = (!prop || prop === CATEGORY_NONE || values.length === 0 || values.length > 10);
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
      if (rec.layer?.setStyle) rec.layer.setStyle(f => styleForAnomalyFeature(f, rec.style || {}));
      renderLegend();
    };
    [iStroke, iW, iSO, iFill, iFO].forEach(inp => inp.addEventListener("input", apply));

    // commit
    rec.categorical = { prop: CATEGORY_NONE, classes, values:[ALL_KEY] };
    if (rec.layer?.setStyle) rec.layer.setStyle(f => styleForAnomalyFeature(f, rec.style || {}));
    renderLegend();
    return;
  }

  // Real categories (<=10)
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
      if (rec.layer?.setStyle) rec.layer.setStyle(f => styleForAnomalyFeature(f, rec.style || {}));
      renderLegend();
    };
    [iStroke, iW, iSO, iFill, iFO].forEach(inp => inp.addEventListener("input", apply));
  });

  rec.categorical = { prop, classes, values };
  if (rec.layer?.setStyle) rec.layer.setStyle(f => styleForAnomalyFeature(f, rec.style || {}));
  renderLegend();
}






// async function applySessionToMap(sessionName){
//   // always get summary first
//   const res = await fetch(`/api/session_summary?session=${encodeURIComponent(sessionName)}`, { cache: 'no-store' });
//   if(!res.ok){ console.warn('session_summary failed'); return; }
//   const sum = await res.json();

//   // Build safe fallbacks (in case backend omitted fields)
//   const sessRoot = `/media/sessions/${encodeURIComponent(sessionName)}/`;
//   const anomaliesUrl =
//     sum.anomalies_geojson || sum.geojson_url || sum.geojson || (sessRoot + "anomalies.geojson");
//   const imagesUrl =
//     sum.images_geojson || sum.images || sum.images_gj || (sessRoot + "images.geojson");

//   // 1) anomalies (polygons)
//   if (anomaliesUrl) {
//     try {
//       await loadGeoJSON(anomaliesUrl);
//     } catch (e) {
//       console.warn("anomalies_geojson fetch failed:", e);
//     }
//   }

//   // 2) images (points -> markers list)
//   if (imagesUrl) {
//     try {
//       const gj = await (await fetch(imagesUrl, { cache: 'no-store' })).json();
//       installImageMarkers(gj);   // fills imageMarkers[] + adds to imageMarkersLayer
//     } catch (e) {
//       console.warn("images_geojson fetch failed:", e);
//       clearImageMarkers();
//     }
//   } else {
//     clearImageMarkers();
//   }

//   // 3) update sidebar after layers installed
//   refreshLayersPanel();
// }


function propsTable(props = {}) {
  const rows = Object.entries(props).map(([k, v]) => {
    const val = (v == null) ? "" : (typeof v === "string" ? v : JSON.stringify(v));
    return `<tr><th>${escapeHtml(k)}</th><td>${escapeHtml(val)}</td></tr>`;
  }).join("");
  return `<table class="propTable">${rows}</table>`;
}

function featurePopupHTML(f) {
  const props = f?.properties || {};
  // optional preview if overlay/thumb URL exists
  const previewURL = props.overlay || props.thumb || props.url;
  const title = props.image || props.file || props.name || "Feature";
  return `
    <div class="popupWrap">
      <div class="popupTitle"><b>${escapeHtml(title)}</b></div>
      ${previewURL ? `<div class="popupPreview" style="margin:.5rem 0">
        <img src="${previewURL}" style="max-width:240px;max-height:180px;border:1px solid var(--border);border-radius:6px;">
      </div>` : ``}
      ${propsTable(props)}
    </div>
  `;
}



// async function loadGeoJSON(url){
//   const res = await fetch(url);
//   const gj = await res.json();

//   const st = overlayRegistry["Anomalies"]?.style || {
//     color: "#ff5722", weight: 1, opacity: 1,
//     fillColor: "#ff5722", fillOpacity: 0.25
//   };

//   geojsonLayer = L.geoJSON(gj, {
//     style: () => st,
//     pointToLayer: (f, latlng) => {
//       if(f.properties && f.properties.type === "image"){
//         return L.marker(latlng).bindPopup(`<div><b>${escapeHtml(f.properties.name||"image")}</b><br>${f.properties.url?`<a href="${f.properties.url}" target="_blank">open image</a>`:""}</div>`);
//       }
//       return L.circleMarker(latlng, { radius: 4, color: "#3388ff", fillColor:"#3388ff", fillOpacity:0.8 });
//     },
//     onEachFeature: (feature, layer) => {
//       try { layer.bindPopup(featurePopupHTML(feature)); } catch(_) {}
//     }
//   }).addTo(MAP);

//   overlayRegistry["Anomalies"] = { layer: geojsonLayer, type: "geojson", style: st };
//   refreshLayersPanel();
//   renderLegend();
//   try{ MAP.fitBounds(geojsonLayer.getBounds(), {padding:[20,20]}); }catch(_){}
// }


let anomaliesProp = 'class_name';  // current property to color by

async function loadGeoJSON(url){
  const res = await fetch(url);
  const gj = await res.json();

  const base = overlayRegistry["Anomalies"]?.style || {
    color: "#ff5722", weight: 1, opacity: 1,
    fillColor: "#ff5722", fillOpacity: 0.25
  };

  // remove previous Anomalies layer (if any) so we don't end up with duplicates
  try{
    const prev = overlayRegistry["Anomalies"];
    if (prev && prev.layer){ try { MAP.removeLayer(prev.layer); } catch(_){} }
  }catch(_){ }

  const layer = L.geoJSON(gj, {
    style: (f)=> styleForAnomalyFeature(f, base),
    pointToLayer: (f, latlng) => L.circleMarker(latlng, { radius: 4, color: base.color, fillColor: base.fillColor, fillOpacity: 0.8 }),
    onEachFeature: (feature, layer) => { try { layer.bindPopup(featurePopupHTML(feature)); } catch(_) {} }
  }).addTo(MAP);

  overlayRegistry["Anomalies"] = { layer, type: "geojson", style: base, data: gj, categorical: overlayRegistry["Anomalies"]?.categorical || null };
  renderLegend();
  try{ MAP.fitBounds(layer.getBounds(), {padding:[20,20]}); }catch(_){}
}


function styleForAnomalyFeature(f, fallback){
  const rec = overlayRegistry["Anomalies"];
  const cat = rec?.categorical;
  if (!cat) return fallback || rec?.style || { color:"#ff5722", weight:1, opacity:1, fillColor:"#ff5722", fillOpacity:0.25 };

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
    return fallback || rec?.style || { color:"#ff5722", weight:1, opacity:1, fillColor:"#ff5722", fillOpacity:0.25 };
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
  return fallback || rec?.style || { color:"#ff5722", weight:1, opacity:1, fillColor:"#ff5722", fillOpacity:0.25 };
}



function applyCategoricalStyling(prop='class_name'){
  const rec = overlayRegistry["Anomalies"];
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

  const rec = overlayRegistry["Anomalies"];
  if (!rec || !rec.layer) return;

  const title = document.createElement('div');
  title.className = 'legendHeader';
  const by = rec.categorical?.prop && rec.categorical.prop !== '__none__'
    ? ` — <span class="dim">by <b>${escapeHtml(rec.categorical.prop)}</b></span>` : '';
  title.innerHTML = `<div class="legendTitle">Anomalies${by}</div>`;
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
    const name = f.properties?.name || f.properties?.file || `image ${i+1}`;
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


async function applySessionToMap(sessionName){
  // 0) clear any previous GeoTIFF tiles
  removeTifTiles();

  // 1) session summary (urls for geojsons)
  const res = await fetch(`/api/session_summary?session=${encodeURIComponent(sessionName)}`, { cache: 'no-store' });
  if (!res.ok) { console.warn('session_summary failed'); return; }
  const sum = await res.json();
  // cache the session summary so loadImagesCatalog can prefer rotated_images when available
  lastLoadedSessionSummary = sum || null;

  const sessRoot   = `/media/sessions/${encodeURIComponent(sessionName)}/`;
  const anomaliesUrl = sum.anomalies_geojson || sum.geojson_url || sum.geojson || (sessRoot + 'anomalies.geojson');
  const imagesUrl    = sum.images_geojson    || sum.images      || sum.images_gj || (sessRoot + 'images.geojson');

  // 2) anomalies polygons (load regardless)
  if (anomaliesUrl){
    try { await loadGeoJSON(anomaliesUrl); }
    catch(e){ console.warn('anomalies fetch failed:', e); }
  }

  // 3) Try ORIGINAL GeoTIFF tiles
  let tiles = null;
  try{
    const r = await fetch(`/api/session_tiles?session=${encodeURIComponent(sessionName)}`, { cache:'no-store' });
    if (r.ok) tiles = await r.json();
  }catch(e){ console.warn('session_tiles failed:', e); }

  const hasTifTiles = !!(tiles?.ok && Array.isArray(tiles.layers) && tiles.layers.length);

  if (hasTifTiles){
    const b = createTifTileGroup(tiles.layers);
    TIF_TILE_GROUP  = b.group;
    TIF_TILE_LAYERS = b.layers;

    // show controller row inside Images list (replaces normal images there)
    installTilesIntoImagesList(sessionName, tiles.layers);

    // fit to raster on first load
    if (b.firstBounds){
      try{ MAP.fitBounds(b.firstBounds, { padding:[20,20] }); }catch(_){}
    }
  } else {
    // Fallback: point markers loaded from images.geojson
    await loadImagesCatalog(sessionName, imagesUrl);
  }

  refreshLayersPanel();
}

function createTifTileGroup(layerDefs){
  const group = L.layerGroup();
  const layers = [];
  let firstBounds = null;

  layerDefs.forEach((Ldef, i) => {
    const lyr = L.tileLayer(Ldef.template, {
      minZoom: Ldef.minzoom ?? 0,
      maxZoom: Ldef.maxzoom ?? 22
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

function installTilesIntoImagesList(sessionName, layerDefs){
  const list = document.getElementById('imagesList');
  if (!list) return;
  list.innerHTML = "";

  const label = layerDefs.length === 1 ? layerDefs[0].name : `Orthophoto (GeoTIFF)`;

  const row = document.createElement('li');
  row.innerHTML = `
    <label class="chk">
      <input type="checkbox" id="chkTifTiles" checked>
      <span>${escapeHtml(label)}</span>
    </label>
    <div style="margin-left:auto;display:flex;gap:.75rem;align-items:center;">
      <span class="dim">Opacity</span>
      <input id="tifOpacity" type="range" min="0" max="1" step="0.05" value="1" style="width:120px">
      <button class="iconDots more" title="Zoom">🔍</button>
    </div>
  `;

  const chk   = row.querySelector('#chkTifTiles');
  const rng   = row.querySelector('#tifOpacity');
  const zoomB = row.querySelector('.more');

  chk.addEventListener('change', ()=>{
    if (!TIF_TILE_GROUP) return;
    if (chk.checked) TIF_TILE_GROUP.addTo(MAP);
    else { try{ MAP.removeLayer(TIF_TILE_GROUP); }catch(_){} }
  });

  rng.addEventListener('input', ()=>{
    const v = Math.max(0, Math.min(1, parseFloat(rng.value || '1')));
    (TIF_TILE_LAYERS || []).forEach(l => { try { l.setOpacity(v); } catch {} });
  });

  zoomB.addEventListener('click', ()=>{
    // zoom to first layer bounds (we stored this in applySessionToMap)
    const bounds = computeLayerBounds(TIF_TILE_GROUP);
    if (bounds && bounds.isValid && bounds.isValid()) {
      MAP.fitBounds(bounds.pad(0.2));
    }
  });

  list.appendChild(row);
}



function installTileLayers(layers){
  // clear any previous raster tiles
  for (const tl of tileLayers){ try{ MAP.removeLayer(tl); }catch(_){} }
  tileLayers = [];

  // use a single registry key "Orthophoto" (group of all tile layers)
  const group = L.layerGroup();
  let firstBounds = null;

  layers.forEach((Ldef, i)=>{
    const lyr = L.tileLayer(Ldef.template, {
      minZoom: Ldef.minzoom ?? 0,
      maxZoom: Ldef.maxzoom ?? 22
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
  overlayRegistry["Orthophoto"] = { layer: group, type: "raster", style: { opacity: 1 } };

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
          .bindPopup(`<div class="mini"><b>Image:</b> ${escapeHtml(feat?.properties?.image || '')}</div>`)
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
    const feats = Array.isArray(modifiedGJ?.features) ? modifiedGJ.features : [];
    for (const f of feats){
      if (f?.geometry?.type !== 'Point') continue;

      const [lng, lat] = f.geometry.coordinates || [];
      if (typeof lat !== 'number' || typeof lng !== 'number') continue;

      // filename is in properties.image (per your sample)
      const file = f?.properties?.image || f?.properties?.file || f?.properties?.name;
      if (!file) continue;

      // prepared overlay PNG path (server writes PNG overlays now)
      const stem = file.replace(/\.[^.]+$/, '');
      // prefer backend-produced rotated images when the session summary indicates
      // camera references or explicit rotated_images assets. Mark the record so
      // the overlay code can skip client-side transforms for rotated assets.
      let url = `/media/sessions/${encodeURIComponent(sessionName)}/overlays/${encodeURIComponent(stem)}.png`;
      let isRotated = false;
      try{
        const sum = lastLoadedSessionSummary || {};
        const assets = sum.assets || {};
        if (Array.isArray(assets.rotated_images) && assets.rotated_images.length){
          url = `/media/sessions/${encodeURIComponent(sessionName)}/rotated_images/${encodeURIComponent(stem)}.png`;
          isRotated = true;
        } else if (sum.camera_meta){
          // If camera_meta exists (user uploaded references.txt) we prefer rotated_images
          url = `/media/sessions/${encodeURIComponent(sessionName)}/rotated_images/${encodeURIComponent(stem)}.png`;
          isRotated = true;
        }
      }catch(_){ /* ignore */ }

      // If the backend provided true footprint corners, use them (and rotation)
      let bounds = null;
      let storedRotation = 0;
      const corners = f?.properties?.corners;
      if (Array.isArray(corners) && corners.length >= 4){
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
          storedRotation = Number(f?.properties?.rotation || 0);
        }catch(_){ bounds = null; }
      }

      // If we didn't get corners, but backend wrote width/height in meters, build bbox
      if (!bounds && f?.properties?.width_m && f?.properties?.height_m){
        try{
          const halfW = Number(f.properties.width_m) / 2.0;
          const halfH = Number(f.properties.height_m) / 2.0;
          // convert meters -> degrees at this latitude
          const top = lat + (halfH / 111320);
          const bottom = lat - (halfH / 111320);
          const left = lng - (halfW / (111320 * Math.cos(lat * Math.PI / 180)));
          const right = lng + (halfW / (111320 * Math.cos(lat * Math.PI / 180)));
          bounds = L.latLngBounds(L.latLng(bottom, left), L.latLng(top, right));
          storedRotation = Number(f?.properties?.rotation || 0);
        }catch(_){ bounds = null; }
      }

      // Fallback: small footprint around point if we don’t have true bounds
      if (!bounds){
        const { dLat, dLon } = metersToDeg(lat, DEFAULT_IMAGE_HALF_SIZE_M);
        const sw = L.latLng(lat - dLat, lng - dLon);
        const ne = L.latLng(lat + dLat, lng + dLon);
        bounds = L.latLngBounds(sw, ne);
      }

      imageCatalog.push({ id: file, name: file, url, bounds, on: false, rotation: storedRotation, corners: Array.isArray(corners) ? corners : null, isRotated });
    }

    renderImagesList();
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
    const btn = e.target.closest('.openImg');
    if (!btn) return;
    const id = btn.getAttribute('data-id');
    const rec = imageCatalog.find(x => x.id === id);
    if (rec?.url) window.open(rec.url, '_blank');
  });
}

const btnShowAll = document.getElementById('btnShowAllImages');
const btnHideAll = document.getElementById('btnHideAllImages');
if (btnShowAll) btnShowAll.addEventListener('click', ()=> setAllImageOverlays(true));
if (btnHideAll) btnHideAll.addEventListener('click', ()=> setAllImageOverlays(false));



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
        <span>${escapeHtml(key)}</span>
      </label>
      <button class="iconDots layerMenuBtn" data-key="${escapeHtml(key)}" title="Layer menu">⋯</button>
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


// ---------- user GeoJSON upload ----------
$("#fileGeoJSON")?.addEventListener("change", async (e)=>{
  const f = e.target.files[0];
  if(!f) return;
  try{
    const text = await f.text();
    const gj = JSON.parse(text);
    const st = { color: "#ffc107", weight: 2, opacity:1, fillColor:"#ffc107", fillOpacity:0.15 };
    const layer = L.geoJSON(gj, { style: st }).addTo(MAP);
    overlayRegistry[`User: ${f.name}`] = { layer, type: "geojson", style: st };
    refreshLayersPanel();
    try{ MAP.fitBounds(layer.getBounds(), {padding:[20,20]}); }catch(_){}
  }catch(_){
    alert("Invalid GeoJSON");
  }finally{
    e.target.value = "";
  }
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

// show selected filename for camera references input
document.getElementById('fileCameraRefs')?.addEventListener('change', (e)=>{
  try{
    const inp = e.target;
    const f = inp.files && inp.files[0];
    const label = inp.closest('.fileDrop');
    const span = label ? label.querySelector('span') : null;
    if (span){
      span.textContent = f ? f.name : 'Choose references.txt / CSV / TSV…';
    }
  }catch(_){ }
});

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

    const isTestLine = (line.includes("[test]") || line.includes("UI:INFO:test") || line.includes("UI:OK:test") || line.includes("UI:ERR:test"));
    if(isTestLine){ appendMiniLog("#testMiniLog", line); }

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
  loadModels(getSelectedBackend(), '#selModelFolder');
    }
    if(line.includes("UI:ERR:train:")){
      setHidden($("#spinTrain"), true);
      err("train", line.split("UI:ERR:train:").pop().trim());
      wireAlertClose();
      setText("#trainStatus","Training failed.");
    }

    if(line.includes("UI:OK:test: Test complete")){
      setText("#testStatus","Inference complete.");
      setHidden($("#spinTest"), true);
      ok("test","Testing completed.");
      wireAlertClose();
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

function _applyLightboxScale(){
  const img = document.getElementById('lightboxImg');
  if(!img) return;
  // ensure img displays transforms correctly
  img.style.transform = `scale(${_lightboxScale})`;
  img.style.transition = 'transform 120ms ease-out';
  img.style.display = 'block';
  img.style.maxWidth = 'none';
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

  const prev = document.getElementById("imgPrev");
  const next = document.getElementById("imgNext");
  if (prev) prev.disabled = (_gIdx === 0);
  if (next) next.disabled = (_gIdx === _gallery.length - 1);
}

function _openLightboxWithGallery(items, startIdx){
  _gallery = items.map(it => ({ src: it.overlay, file: it.file }));
  _gIdx = Math.max(0, Math.min(startIdx || 0, _gallery.length - 1));
  document.getElementById("lightbox").classList.remove("hidden");
  _lightboxOpen = true;
  _setLightbox(_gIdx);
}

function _closeLightbox(){
  document.getElementById("lightbox").classList.add("hidden");
  _lightboxOpen = false;
}

function _nextImg(){ if (_gIdx < _gallery.length - 1) _setLightbox(_gIdx + 1); }
function _prevImg(){ if (_gIdx > 0) _setLightbox(_gIdx - 1); }


// ---------- UI wiring ----------
function setupUI(){
  setupTabs();

  $("#btnRefreshFolders").addEventListener("click", loadDatasets);
  $("#btnRefreshModels").addEventListener("click", ()=> loadModels(getSelectedBackend(), '#selModelFolder'));
  $("#btnOpenUploadModal").addEventListener("click", openUploadModal);
  $("#btnCloseUploadModal").addEventListener("click", ()=>{ closeUploadModal(); resetUploadProgress(); });
  $("#btnCancelUpload").addEventListener("click", ()=>{ closeUploadModal(); resetUploadProgress(); });
  $("#btnStartUpload").addEventListener("click", startUpload);

  $("#btnTrain").addEventListener("click", startTraining);
  $("#btnCancelTrain").addEventListener("click", cancelTraining);

  $("#btnTest").addEventListener("click", runTest);
  $("#btnCancelTest").addEventListener("click", cancelTest);

  $("#lnkToLogsFromTest").addEventListener("click", (e)=>{ e.preventDefault(); switchToTab("tab-logs"); });
  $("#lnkToLogsFromTrain").addEventListener("click", (e)=>{ e.preventDefault(); switchToTab("tab-logs"); });

  $("#btnLogsConnect").addEventListener("click", connectLogs);
  $("#btnLogsClear").addEventListener("click", ()=>{ $("#logStream").textContent=""; });

  $("#btnRefreshSessions").addEventListener("click", async ()=>{
    await loadSessions(true);
    await showResultsForSelected();
  });
  $("#selResults").addEventListener("change", showResultsForSelected);

  $("#btnRefreshMapSessions").addEventListener("click", async ()=>{
    await loadSessions(true);
    await refreshMapSessionSelected();
  });
  $("#selMapSession").addEventListener("change", refreshMapSessionSelected);

  // initial hidden spinners (also hidden attribute in HTML prevents flash)
  setHidden($("#spinTrain"), true);
  setHidden($("#spinTest"), true);

  wireAlertClose();

  // Backend selector wiring: show YOLO options when YOLO is selected
  const selBackendGlobal = $("#selBackend");
  const selBackendTrain = $("#selBackendTrain");
  const selBackendTest  = $("#selBackendTest");

  // Train-side: show/hide YOLO-specific train options and the Detectron model-type selector
  function _updateYoloUIForTrain(){
    const b = (selBackendTrain && selBackendTrain.value) ? selBackendTrain.value : (selBackendGlobal && selBackendGlobal.value) || 'detectron';
    const show = (b === 'yolo');
    const elOpts = $("#yoloOptions");
    const elSeg = $("#yoloSegOption");
    const elSize = $("#yoloSizeOption");
    if(elOpts) elOpts.style.display = show ? 'block' : 'none';
    // The segmentation checkbox is meaningful for Detectron (Mask R-CNN).
    // Show it only when Detectron is selected; hide it for YOLO.
    if(elSeg)  elSeg.style.display = (b === 'detectron') ? 'block' : 'none';
    if(elSize) elSize.style.display = show ? 'block' : 'none';
    // Hide Detectron-only model type selector when YOLO selected
    const modelTypeRow = document.getElementById('selModelType') ? document.getElementById('selModelType').closest('.row') : null;
    if(modelTypeRow) modelTypeRow.style.display = show ? 'none' : 'block';
  }
  if(selBackendGlobal) selBackendGlobal.addEventListener('change', _updateYoloUIForTrain);
  if(selBackendTrain) selBackendTrain.addEventListener('change', ()=>{ _updateYoloUIForTrain(); /* no-op for models list on train */ });
  _updateYoloUIForTrain();

  // Show/hide the channel count controls when the user toggles Use thermal band on Train tab
  const chkUseThermalTrain = document.getElementById('chkUseThermalTrain');
  const trainBandBlock = document.getElementById('trainBandControls');
  if (chkUseThermalTrain && trainBandBlock){
    const toggle = ()=>{
      const show = !!chkUseThermalTrain.checked;
      trainBandBlock.style.display = show ? 'block' : 'none';
    };
    // initialize
    toggle();
    chkUseThermalTrain.addEventListener('change', ()=>{ userToggledThermalTrain = true; toggle(); });
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
        Number(m.channel_count) === 4 || (Number(m.channel_count) === 3 && (!!(m && m.thermal_only) || !!(m && m.thermal_used)))
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
  }

  // keyboard: ← - Esc (also allow A/D)
  document.addEventListener("keydown", (e)=>{
    if (!_lightboxOpen) return;
    const k = e.key;
    if (k === "ArrowRight" || k === "d" || k === "D") { _nextImg(); e.preventDefault(); }
    else if (k === "ArrowLeft" || k === "a" || k === "A") { _prevImg(); e.preventDefault(); }
    else if (k === "Escape") { _closeLightbox(); e.preventDefault(); }
  });

}

document.addEventListener("DOMContentLoaded", async ()=>{
  setupUI();
  initMap();
  connectLogs();
  await Promise.all([loadDatasets(), loadModels(getSelectedBackend(), '#selModelFolder'), loadSessions(true)]);
  if($("#selResults").value){ await showResultsForSelected(); }
  if($("#selMapSession").value){ await refreshMapSessionSelected(); }
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
      return `<li><span class="key">${k}</span><span class="val">${val}</span></li>`;
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
  predictions_title.innerHTML = "Predictions: "+ metrics.total_detections;
}
