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
let geojsonLayer = null;
let imageMarkersLayer = null;
let tileLayers = [];
let currentSession = null;
let styleTarget = null;
let layerMenuState = { name: null, info: null };
let testAbort = null;

// ---------- helpers ----------
const $ = sel => document.querySelector(sel);
const $$ = sel => Array.from(document.querySelectorAll(sel));
function setHidden(el, hidden=true){ if(!el) return; hidden ? el.style.display = 'none' : el.style.display = 'block' ; }
function setText(sel, txt){ const el=$(sel); if(el) el.textContent = txt; }
function escapeHtml(s){ return (s||"").replace(/[&<>"']/g, m=>({ "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;" }[m])); }
function closeBtn(){ return `<button class="iconBtn alertClose" aria-label="Close">×</button>`; }

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
  const sel = $("#selModelFolder");
  sel.innerHTML = "";
  list.forEach(m => {
    const o = document.createElement("option");
    o.value = m.name;
    o.textContent = `${m.name} — ${m.input_mode}`;
    sel.appendChild(o);
  });
}

function getSelectedDataset(){ return $("#selTestFolder").value || null; }
function getSelectedModel(){ return $("#selModelFolder").value || null; }

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
async function loadModels(){
  const res = await fetch(api.models);
  const js = await res.json();
  if(js.ok){ populateModels(js.models); }
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

  setHidden($("#spinTrain"), false);
  setText("#trainStatus","Submitting training job…");

  const fd = new FormData();
  fd.append("use_thermal", useThermal ? "true" : "false");
  fd.append("max_iter", String(iters));
  fd.append("base_lr", String(lr));
  fd.append("ims_per_batch", String(batch));
  fd.append("model_name", String(modelName));


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
  const useThermal = $("#chkUseThermalTest").checked;
  const resultName = (document.getElementById("inpResultName")?.value || "").trim() || makeStamp();

  setHidden($("#spinTest"), false);
  setText("#testStatus","Running inference…");

  const fd = new FormData();
  fd.append("dataset", ds);
  if(model) fd.append("model", model);
  fd.append("use_thermal", useThermal ? "true":"false");
  fd.append("result_name", resultName);

  try{
    testAbort = new AbortController();
    const res = await fetch(api.testRun, { method:"POST", body: fd, signal: testAbort.signal });
    const js = await res.json();
    const totalPreds = Array.isArray(js.manifest)
      ? js.manifest.reduce((sum, it) => sum + (it.n || 0), 0)
      : 0;

    if(!js.ok) throw new Error("test failed");

    currentSession = js.session;
    ok("test", "Testing completed.");
    setText("#testStatus", `Inference complete. ${totalPreds} predictions.`);

    // load into map & results
    await applySessionToMap(currentSession);
    renderResultsGrid(js.manifest);
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
  if(!session) return;
  currentSession = session;
  const res = await fetch(`${api.sessionSummary}?session=${encodeURIComponent(session)}`);
  const js = await res.json();
  if(!js.ok) return;
  renderResultsGrid(js.manifest && js.manifest.length ? js.manifest : pairThumbs(js.assets));
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
  if(!manifest || !manifest.length){
    grid.innerHTML = `<div class="muted">No overlays generated.</div>`;
    return;
  }
  manifest.forEach((item, idx)=>{
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

  imageMarkersLayer = L.layerGroup().addTo(MAP);
  overlayRegistry["Image markers"] = { layer: imageMarkersLayer, type: "markers" };
  refreshLayersPanel();
  renderLegend();
}

async function applySessionToMap(session){
  // clear old layers
  if(geojsonLayer){ geojsonLayer.remove(); geojsonLayer = null; }
  imageMarkersLayer.clearLayers();
  tileLayers.forEach(l=>MAP.removeLayer(l));
  tileLayers = [];
  delete overlayRegistry["Anomalies"];
  delete overlayRegistry["Thermal Tiles"];

  // summary
  const sumRes = await fetch(`${api.sessionSummary}?session=${encodeURIComponent(session)}`);
  const sum = await sumRes.json();
  if(sum.geojson_url){
    await loadGeoJSON(sum.geojson_url);
  }
  // add image markers again (from geojson we can't reconstruct popups) – fetch quick
  // session manifest not needed; markers come separately from /api/test_run, so just skip here

  // TIF tiles
  const tRes = await fetch(`${api.sessionTiles}?session=${encodeURIComponent(session)}`);
  const tj = await tRes.json();
  if(tj.ok && tj.layers && tj.layers.length){
    const group = L.layerGroup();
    tj.layers.forEach((t, idx)=>{
      const tl = L.tileLayer(t.template, { minZoom: t.minzoom, maxZoom: t.maxzoom, opacity: 0.6, crossOrigin: true });
      tl.addTo(group);
      tileLayers.push(tl);
    });
    group.addTo(MAP);
    overlayRegistry["Thermal Tiles"] = { layer: L.layerGroup(tileLayers), type: "raster", style: { opacity: 0.6 } };
    refreshLayersPanel();
  }else{
    appendMiniLog("#testMiniLog","[map] Raster tiler unavailable or no TIFs in session.");
  }
}

async function loadGeoJSON(url){
  const res = await fetch(url);
  const gj = await res.json();

  const st = overlayRegistry["Anomalies"]?.style || {
    color: "#ff5722", weight: 1, opacity: 1,
    fillColor: "#ff5722", fillOpacity: 0.25
  };

  geojsonLayer = L.geoJSON(gj, {
    style: () => st,
    pointToLayer: (f, latlng) => {
      if(f.properties && f.properties.type === "image"){
        return L.marker(latlng).bindPopup(`<div><b>${escapeHtml(f.properties.name||"image")}</b><br>${f.properties.url?`<a href="${f.properties.url}" target="_blank">open image</a>`:""}</div>`);
      }
      return L.circleMarker(latlng, { radius: 4, color: "#3388ff", fillColor:"#3388ff", fillOpacity:0.8 });
    }
  }).addTo(MAP);

  overlayRegistry["Anomalies"] = { layer: geojsonLayer, type: "geojson", style: st };
  refreshLayersPanel();
  renderLegend();
  try{ MAP.fitBounds(geojsonLayer.getBounds(), {padding:[20,20]}); }catch(_){}
}

function renderLegend(){
  const st = overlayRegistry["Anomalies"]?.style || { color:"#ff5722", fillColor:"#ff5722" };
  const el = $("#legend");
  el.innerHTML = `
    <div class="legendItem"><span class="swatch" style="background:${st.fillColor}"></span> Anomaly (polygon)</div>
    <div class="legendItem"><span class="swatch" style="background:#3388ff"></span> Image marker</div>
  `;
}

// ---------- layers panel + ⋮ menu ----------
function refreshLayersPanel(){
  const ul = $("#layersList");
  ul.innerHTML = "";

  Object.keys(overlayRegistry).forEach(name=>{
    const info = overlayRegistry[name];
    if(!info || !info.layer) return;
    const li = document.createElement("li");
    li.innerHTML = `
      <label class="chk">
        <input type="checkbox" ${MAP.hasLayer(info.layer) ? "checked":""} />
        <span>${name}</span>
      </label>
      <button class="iconDots" title="Layer actions">⋮</button>
    `;
    const chk = li.querySelector("input");
    chk.addEventListener("change", ()=>{
      if(chk.checked) info.layer.addTo(MAP);
      else MAP.removeLayer(info.layer);
    });
    const dots = li.querySelector(".iconDots");
    dots.addEventListener("click", (e)=>{
      e.stopPropagation();
      openLayerMenu(name, info, e.clientX, e.clientY);
    });
    ul.appendChild(li);
  });
}

function openLayerMenu(name, info, x, y){
  const menu = $("#layerMenu");
  layerMenuState = { name, info };
  menu.style.left = `${x}px`;
  menu.style.top = `${y}px`;
  menu.classList.remove("hidden");
}
window.addEventListener("click", ()=> $("#layerMenu").classList.add("hidden"));
$("#layerMenu [data-act='zoom']").addEventListener("click", ()=>{
  const { info } = layerMenuState;
  if(!info) return;
  try{
    const b = info.layer.getBounds ? info.layer.getBounds() : null;
    if(b && b.isValid()) MAP.fitBounds(b, {padding:[20,20]});
  }catch(_){}
  $("#layerMenu").classList.add("hidden");
});
$("#layerMenu [data-act='style']").addEventListener("click", ()=>{
  const { name, info } = layerMenuState;
  if(!info) return;
  styleTarget = { name, info };
  const st = info.style || { color:"#ff5722", opacity:1, weight:1, fillColor:"#ff5722", fillOpacity:0.25 };
  $("#stColor").value = toHex(st.color);
  $("#stWidth").value = st.weight ?? 1;
  $("#stOpacity").value = st.opacity ?? 1;
  $("#fiColor").value = toHex(st.fillColor || st.color || "#ff5722");
  $("#fiOpacity").value = st.fillOpacity ?? 0.25;
  $("#styleModal").classList.remove("hidden");
  $("#layerMenu").classList.add("hidden");
});
function toHex(c){
  if(!c) return "#ff5722";
  const ctx = document.createElement("canvas").getContext("2d");
  ctx.fillStyle = c;
  return ctx.fillStyle;
}
$("#btnCloseStyle").addEventListener("click", ()=> $("#styleModal").classList.add("hidden"));
$("#btnCancelStyle").addEventListener("click", ()=> $("#styleModal").classList.add("hidden"));
$("#btnApplyStyle").addEventListener("click", ()=>{
  if(!styleTarget) return;
  const st = {
    color: $("#stColor").value,
    weight: parseFloat($("#stWidth").value || "1"),
    opacity: parseFloat($("#stOpacity").value || "1"),
    fillColor: $("#fiColor").value,
    fillOpacity: parseFloat($("#fiOpacity").value || "0.25")
  };
  styleTarget.info.style = st;
  if(styleTarget.info.type === "geojson"){
    styleTarget.info.layer.setStyle(st);
    renderLegend();
  }else if(styleTarget.info.type === "raster"){
    const op = Math.max(0, Math.min(1, st.opacity));
    // tile layers share the same opacity
    for(const l of tileLayers) l.setOpacity(op);
  }
  $("#styleModal").classList.add("hidden");
  styleTarget = null;
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
      loadModels();
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
  $("#btnRefreshModels").addEventListener("click", loadModels);
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

  // lightbox
  $("#btnCloseLightbox").addEventListener("click", _closeLightbox);
  document.getElementById("imgNext")?.addEventListener("click", _nextImg);
  document.getElementById("imgPrev")?.addEventListener("click", _prevImg);

  // keyboard: ← → Esc (also allow A/D)
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
  await Promise.all([loadDatasets(), loadModels(), loadSessions(true)]);
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
