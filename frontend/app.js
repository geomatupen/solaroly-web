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
  sessionTiles: "/api/session_tiles",
  colmapState: "/api/colmap/state",
  colmapCameras: "/api/colmap/cameras",
  colmapStart: "/api/colmap/start",
  colmapFinish: "/api/colmap/finish"
};

let MAP, baseLayers, overlayRegistry = {};
let imagesLayerGroup = null;           // holds all image markers/overlays
let imageMarkers = new Map();          // id -> L.Marker or L.ImageOverlay
let geojsonLayer = null;
let imageMarkersLayer = null;
let tileLayers = [];
let currentSession = null;
let styleTarget = null;
let layerMenuState = { name: null, info: null };
let testAbort = null;
let currentManifest = null;            // store current manifest for filtering
let mapDetectionFilterActive = false;
let anomaliesFilterActive = false;

const colmapStates = {};
let colmapPollHandle = null;
let colmapPollDataset = null;
let colmapCamerasPage = 0;
let colmapCamerasHasMore = false;
let colmapCamerasList = [];  // accumulated cameras list
let OPTIMIZE_MAP = null;
let optimizeMapLayer = null;       // optimized markers
let optimizeMapOrigLayer = null;   // original EXIF/gimbal markers
let optimizeMapFocusLayer = null;
let optimizeMapLastBoundsSig = null;
let optimizeMapLastDataset = null;
let optimizeMapFootprintLayer = null;
let optimizeShowFootprints = false;
let optimizeCameraVisibility = {};  // { filename: true/false }

// runtime caches & UI flags
let modelsCache = {};                // name -> model metadata returned by /api/models
let datasetsCache = [];              // cached dataset list with colmap_ready flag
let userToggledThermalTrain = false; // whether user manually toggled the train thermal checkbox
let userToggledThermalTest = false;  // whether user manually toggled the test thermal checkbox

const COLMAP_PARAM_FIELDS = {
  matcher: { id: 'selColmapMatcher', type: 'value' },
  camera_model: { id: 'selColmapCameraModel', type: 'value' },
  seq_overlap: { id: 'inpColmapSeqOverlap', type: 'value' },
  min_triangulation_angle: { id: 'inpColmapTriangulation', type: 'value' },
  min_model_size: { id: 'inpColmapMinSize', type: 'value' },
  min_num_matches: { id: 'inpColmapMinMatches', type: 'value' },
  max_image_size: { id: 'inpColmapMaxImage', type: 'value' },
  init_min_num_inliers: { id: 'inpColmapInitInliers', type: 'value' },
  abs_pose_min_num_inliers: { id: 'inpColmapAbsInliers', type: 'value' },
  max_model_overlap: { id: 'inpColmapMaxOverlap', type: 'value' },
  max_num_models: { id: 'inpColmapMaxModels', type: 'value' },
  peak_threshold: { id: 'inpColmapPeakThresh', type: 'value' },
  edge_threshold: { id: 'inpColmapEdgeThresh', type: 'value' },
  max_num_features: { id: 'inpColmapMaxFeatures', type: 'value' },
  num_threads: { id: 'inpColmapThreads', type: 'value' },
  exhaustive_block_size: { id: 'inpColmapExhBlock', type: 'value' },
  ba_refine_focal_length: { id: 'chkColmapBaFocal', type: 'checkbox' },
  ba_refine_principal_point: { id: 'chkColmapBaPpoint', type: 'checkbox' },
  ba_refine_extra_params: { id: 'chkColmapBaExtra', type: 'checkbox' },
  use_gpu: { id: 'chkColmapUseGpu', type: 'checkbox' },
  max_reproj_error: { id: 'inpColmapMaxReprojError', type: 'value' },
  ba_global_max_iterations: { id: 'inpColmapBaGlobalIter', type: 'value' },
  ba_local_max_iterations: { id: 'inpColmapBaLocalIter', type: 'value' }
};
const colmapParamDefaults = {};
const colmapSavedParams = new Map();
let colmapPrefillState = { dataset: null, hash: null };

// catalog & runtime overlays for photos
let imageCatalog = [];              // [{ id, name, url, bounds, on }]
let imageOverlays = new Map();      // id -> L.ImageOverlay
let imagesOpacity = 0.85;           // global opacity for image overlays

// last loaded images.geojson and session info (used to reapply overrides)
let lastLoadedImagesGJ = null;
let lastLoadedSessionName = null;
let lastLoadedImagesUrl = null;
let lastLoadedSessionSummary = null;
let rotatedImagesLookup = null;     // cached map of basename -> rotated URL

// camera position overrides loaded from an uploaded WebODM camera-positions JSON
// keyed by normalized basename (no extension), value: { lon, lat, alt }
let cameraPositionOverrides = {};

// fallback sizing if images.geojson has only a Point
const DEFAULT_IMAGE_HALF_SIZE_M = 25; // ~25 m half-width/half-height (adjust if needed)
const CAMERA_ZOOM_ICON = '<svg viewBox="0 0 20 20" width="14" height="14" xmlns="http://www.w3.org/2000/svg" aria-hidden="true" focusable="false"><circle cx="10" cy="10" r="4" fill="none" stroke="currentColor" stroke-width="1.5"/><path d="M10 2v3M10 15v3M2 10h3M15 10h3" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>';

// --- TIF raster globals ---
let TIF_TILE_GROUP = null;   // Leaflet layerGroup that holds the ZXY tile layers
let TIF_TILE_LAYERS = [];    // underlying L.tileLayer instances
let TIF_TILE_BOUNDS = null;  // LatLngBounds for the orthophoto tiles

// Pending map bounds to apply when map tab becomes visible
let pendingMapBounds = null;

function removeTifTiles(){
  if (TIF_TILE_GROUP){
    try { MAP.removeLayer(TIF_TILE_GROUP); } catch(_){ }
  }
  for (const layer of TIF_TILE_LAYERS){
    try { MAP.removeLayer(layer); } catch(_){ }
    try { layer.remove?.(); } catch(_){ }
  }
  TIF_TILE_GROUP = null;
  TIF_TILE_LAYERS = [];
  TIF_TILE_BOUNDS = null;
  const list = document.getElementById('imagesList');
  if (list && list.querySelector('#chkTifTiles')){
    list.innerHTML = '';
  }
}

// ---------- helpers ----------
const $ = sel => document.querySelector(sel);
const $$ = sel => Array.from(document.querySelectorAll(sel));
function setHidden(el, hidden=true){ if(!el) return; hidden ? el.style.display = 'none' : el.style.display = 'block'; }
function setText(sel, txt){ const el=$(sel); if(el) el.textContent = txt; }
function escapeHtml(s){ return (s||"").replace(/[&<>"']/g, m=>({ "&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;" }[m])); }
function closeBtn(){ return `<button class="iconBtn alertClose" aria-label="Close">×</button>`; }

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
      ov = L.imageOverlay(rec.url, rec.bounds, { opacity: imagesOpacity, interactive: false });
      imageOverlays.set(id, ov);
    } else {
      try { ov.setOpacity(imagesOpacity); } catch(_){ }
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

  if (anomaliesFilterActive){
    applyAnomaliesFilter();
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
  if (anomaliesFilterActive){
    applyAnomaliesFilter();
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

  const showOnlyDetections = document.getElementById('chkMapShowOnlyDetections')?.checked || false;
  const filtered = showOnlyDetections 
    ? imageCatalog.filter(rec => rec.n && rec.n > 0)
    : imageCatalog;

  ul.innerHTML = filtered.map(rec => `
    <li>
      <label class="chk">
        <input type="checkbox" class="imgToggle" data-id="${escapeHtml(rec.id)}" ${rec.on ? 'checked' : ''}>
        <span>${escapeHtml(rec.name)}${rec.n ? ` <span style="color:#0a84ff; font-weight:bold;">(${rec.n})</span>` : ''}</span>
      </label>
      <button class="iconDots imgMenu" data-id="${escapeHtml(rec.id)}" title="Options">⋮</button>
    </li>
  `).join('');
}

function applyMapDetectionFilter(){
  const chk = document.getElementById('chkMapShowOnlyDetections');
  const on = chk?.checked || false;
  mapDetectionFilterActive = on;

  if (on){
    // Only turn OFF non-detection images; keep current states for detected images
    for (const rec of imageCatalog){
      if (!(rec.n && rec.n > 0) && rec.on){
        toggleImageOverlay(rec.id, false);
      }
    }
  }

  renderImagesList();
}

function applyAnomaliesFilter(){
  const chk = document.getElementById('chkMapFilterAnomalies');
  const on = chk?.checked || false;
  anomaliesFilterActive = on;

  const rec = overlayRegistry["Anomalies"];
  if (!rec || !rec.data) return;

  const base = rec.style || {
    color: "#ff5722", weight: 1, opacity: 1,
    fillColor: "#ff5722", fillOpacity: 0.25
  };

  // Build list of active image stems
  const activeStems = new Set(
    imageCatalog
      .filter(r => r.on)
      .map(r => String(r.id).replace(/\.[^.]+$/, ''))
  );

  const full = rec.data;
  const filtered = on ? {
    ...full,
    features: (full.features || []).filter(f => {
      const img = f?.properties?.image || f?.properties?.file || f?.properties?.name;
      if (!img) return false;
      const stem = String(img).replace(/\.[^.]+$/, '');
      return activeStems.has(stem);
    })
  } : full;

  // Remove existing layer
  try{ if (rec.layer){ MAP.removeLayer(rec.layer); } }catch(_){ }

  const layer = L.geoJSON(filtered, {
    style: (f)=> styleForAnomalyFeature(f, base),
    pointToLayer: (f, latlng) => L.circleMarker(latlng, { radius: 4, color: base.color, fillColor: base.fillColor, fillOpacity: 0.8 }),
    onEachFeature: (feature, layer) => { try { layer.bindPopup(featurePopupHTML(feature)); } catch(_) {} }
  }).addTo(MAP);

  overlayRegistry["Anomalies"] = { ...rec, layer };
  renderLegend();
}

function updateMapDetectionFilterVisibility(hasTifTiles){
  const wrap = document.getElementById('mapDetectionsFilter');
  if (!wrap) return;
  // Hide filters for orthophoto/tiles pipeline
  wrap.style.display = hasTifTiles ? 'none' : 'flex';
}

function updateImageListButtonsVisibility(hasTifTiles){
  const btnShowAll = document.getElementById('btnShowAllImages');
  const btnHideAll = document.getElementById('btnHideAllImages');
  // Hide "Show all" and "Hide all" buttons for orthophoto/tiles pipeline
  if (btnShowAll) btnShowAll.style.display = hasTifTiles ? 'none' : 'block';
  if (btnHideAll) btnHideAll.style.display = hasTifTiles ? 'none' : 'block';
}


// ---------- Accurate locations / COLMAP ----------

// Common heading conversion utility: DJI heading (0=N, +CW) to radians
// For map rendering: negate=true for standard geo counterclockwise rotation
// For canvas rendering: negate=false for visual clockwise rotation
function headingToRadians(headingDeg, negate = true) {
  const rad = Number.isFinite(headingDeg) ? (negate ? -headingDeg : headingDeg) * Math.PI / 180 : 0;
  return rad;
}

function cacheColmapState(dataset, state){
  if(!dataset) return;
  colmapStates[dataset] = state;
}

function getColmapState(dataset){
  if(!dataset) return null;
  return colmapStates[dataset] || null;
}

function setColmapStatus(msg = "", tone = null){
  const el = document.getElementById('colmapStatusLine');
  if(!el) return;
  el.textContent = msg || "";
  el.classList.remove('ok','warn','err');
  if(tone && ['ok','warn','err'].includes(tone)){
    el.classList.add(tone);
  }
}

function ensureOptimizeMap(){
  const host = document.getElementById('optimizeMap');
  if(!host) return null;
  if(!OPTIMIZE_MAP){
    const street = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", { maxZoom: 22, attribution: "&copy; OpenStreetMap" });
    OPTIMIZE_MAP = L.map(host, { attributionControl:false, zoomControl:false, dragging:true, scrollWheelZoom:true, doubleClickZoom:true });
    street.addTo(OPTIMIZE_MAP);
    optimizeMapLayer = L.layerGroup().addTo(OPTIMIZE_MAP);
    optimizeMapOrigLayer = L.layerGroup().addTo(OPTIMIZE_MAP);
    optimizeMapFocusLayer = L.layerGroup().addTo(OPTIMIZE_MAP);
    optimizeMapFootprintLayer = L.layerGroup().addTo(OPTIMIZE_MAP);
    OPTIMIZE_MAP.setView([0,0],2);
  }else{
    if(!optimizeMapLayer){
      optimizeMapLayer = L.layerGroup().addTo(OPTIMIZE_MAP);
    }
    if(!optimizeMapOrigLayer){
      optimizeMapOrigLayer = L.layerGroup().addTo(OPTIMIZE_MAP);
    }
    if(!optimizeMapFocusLayer){
      optimizeMapFocusLayer = L.layerGroup().addTo(OPTIMIZE_MAP);
    }
    if(!optimizeMapFootprintLayer){
      optimizeMapFootprintLayer = L.layerGroup().addTo(OPTIMIZE_MAP);
    }
  }
  return OPTIMIZE_MAP;
}

function invalidateOptimizeMap(){
  const map = ensureOptimizeMap();
  if(map){ setTimeout(()=> map.invalidateSize(), 60); }
}

function renderOptimizeMap(state, cams = [], readyCount = 0){
  const dataset = document.getElementById('selOptimizeDataset')?.value || '';
  const summaryEl = document.getElementById('colmapMapSummary');
  const map = ensureOptimizeMap();
  if(!map || !optimizeMapLayer){
    if(summaryEl){ summaryEl.textContent = dataset ? 'Map unavailable' : 'No dataset'; }
    return;
  }

  ensureOptimizeControls();

  optimizeMapLayer.clearLayers();
  if(optimizeMapOrigLayer){ optimizeMapOrigLayer.clearLayers(); }
  if(optimizeMapFootprintLayer){ optimizeMapFootprintLayer.clearLayers(); }
  if(!dataset){
    if(summaryEl) summaryEl.textContent = 'No dataset';
    return;
  }
  if(!state || !cams.length){
    if(summaryEl) summaryEl.textContent = state ? 'No cameras' : 'Not optimized';
    return;
  }

  const bounds = [];
  let lociCount = 0;
  cams.forEach(cam => {
    const title = escapeHtml(cam.file || cam.name || 'image');
    const optLat = Number(cam?.optimized?.lat);
    const optLon = Number(cam?.optimized?.lon);
    const origLat = Number(cam?.lat);
    const origLon = Number(cam?.lon);
    const filename = cam.file || cam.name;
    const isVisible = optimizeCameraVisibility[filename] === true;

    const ready = cam.calibrated || (cam.status && String(cam.status).toLowerCase() === 'calibrated') || (cam.optimized && cam.optimized.lat != null);
    let fitLat = null, fitLon = null;

    if(!Number.isNaN(origLat) && !Number.isNaN(origLon) && optimizeMapOrigLayer){
      const m = L.circleMarker([origLat, origLon], {
        radius: 4,
        color: '#f4b400',
        weight: 1.0,
        fillColor: '#f4b400',
        fillOpacity: 0.55
      }).bindPopup(`<div><strong>${title}</strong><br>Original (EXIF/gimbal)</div>`);
      m.addTo(optimizeMapOrigLayer);
      fitLat = origLat; fitLon = origLon;
      lociCount += 1;
      bounds.push([origLat, origLon]);
    }

    if(!Number.isNaN(optLat) && !Number.isNaN(optLon)){
      const marker = L.circleMarker([optLat, optLon], {
        radius: ready ? 6 : 5,
        color: '#18b76a',
        weight: 1.5,
        fillColor: '#18b76a',
        fillOpacity: 0.9
      });
      marker.bindPopup(`<div><strong>${title}</strong><br>${ready ? 'Optimized' : 'Pending'}</div>`);
      marker.addTo(optimizeMapLayer);
      lociCount += 1;
      bounds.push([optLat, optLon]);
      fitLat = optLat; fitLon = optLon;

      if(!Number.isNaN(origLat) && !Number.isNaN(origLon)){
        // draw a connector line for quick visual diff
        const seg = L.polyline([[origLat, origLon], [optLat, optLon]], { color:'#7c3aed', weight:1.5, opacity:0.8, dashArray:'4,4' });
        seg.addTo(optimizeMapLayer);
      }

      if(optimizeMapFootprintLayer && isVisible){
        drawFootprint(optLat, optLon, Number(cam?.optimized?.rotation), Number(cam?.optimized?.w_px ?? cam?.w ?? 0), Number(cam?.optimized?.h_px ?? cam?.h ?? 0), Number(cam?.optimized?.meters_per_pixel ?? cam?.meters_per_pixel ?? 0.05), '#0ea5e9', optimizeMapFootprintLayer);
      }
    }

    if(optimizeMapFootprintLayer && isVisible && !Number.isNaN(origLat) && !Number.isNaN(origLon)){
      let origRot = cam?.rotation;
      if(!Number.isFinite(origRot)){
        origRot = cam?.rotation_gimbal;
      }
      if(!Number.isFinite(origRot)){
        origRot = cam?.rotation_aircraft;
      }
      drawFootprint(origLat, origLon, Number(origRot ?? 0), Number(cam?.w ?? cam?.w_px ?? 0), Number(cam?.h ?? cam?.h_px ?? 0), Number(cam?.meters_per_pixel ?? 0.05), '#f4b400', optimizeMapFootprintLayer, true);
    }

    if(fitLat !== null && fitLon !== null){
      bounds.push([fitLat, fitLon]);
    }
  });

  if(summaryEl){
    if(!lociCount){
      summaryEl.textContent = 'No GPS data';
    }else{
      summaryEl.textContent = `${readyCount}/${cams.length} optimized`;
    }
  }

  // Only refit when dataset changes or bounds signature changes to avoid jumpy view
  if(bounds.length){
    const sig = `${dataset}|${bounds.length}|${bounds.map(b=>b.join(',')).join(';')}`;
    if(dataset !== optimizeMapLastDataset || sig !== optimizeMapLastBoundsSig){
      try{
        map.fitBounds(L.latLngBounds(bounds).pad(0.1));
      }catch(_){
        const bb = L.latLngBounds(bounds);
        map.setView(bb.getCenter(), Math.min(map.getZoom(), 18));
      }
      optimizeMapLastBoundsSig = sig;
      optimizeMapLastDataset = dataset;
    }
  }
  setTimeout(()=> map.invalidateSize(), 80);
}

function drawFootprint(lat, lon, rotDeg, wPx, hPx, mpp, color, layer, dashed=false){
  if(!Number.isFinite(lat) || !Number.isFinite(lon) || !Number.isFinite(wPx) || !Number.isFinite(hPx) || wPx<=0 || hPx<=0){
    return;
  }
  const halfWm = (wPx * mpp) / 2.0;
  const halfHm = (hPx * mpp) / 2.0;
  // Align with backend footprint math: use camera heading directly (CW positive) and adjust lat sign below.
  const rad = headingToRadians(rotDeg, false);
  const ca = Math.cos(rad), sa = Math.sin(rad);
  const scaleLat = 111320;
  const scaleLon = 111320 * Math.cos(lat * Math.PI/180);
  const corners = [
    [-halfWm, -halfHm],
    [ halfWm, -halfHm],
    [ halfWm,  halfHm],
    [-halfWm,  halfHm]
  ].map(([x,y])=>{
    const rx = x*ca - y*sa;
    const ry = x*sa + y*ca;
    // subtract ry to mirror backend geojson corner construction (pixel y is down).
    return [lat - (ry/scaleLat), lon + (rx/scaleLon)];
  });
  const poly = L.polygon(corners, { color, weight: 1.5, opacity: 0.9, fillColor: color, fillOpacity: 0.12, dashArray: dashed ? '4,4' : null });
  poly.addTo(layer);
}

function showCameraDetail(filename, cams){
  const cam = cams.find(c => (c.file || c.name) === filename);
  if(!cam) return;
  const title = escapeHtml(cam.file || cam.name || 'image');
  // Try to get original rotation from multiple sources in priority order
  let origRot = cam?.rotation;
  if(!Number.isFinite(origRot)){
    origRot = cam?.rotation_gimbal;
  }
  if(!Number.isFinite(origRot)){
    origRot = cam?.rotation_aircraft;
  }
  origRot = Number(origRot ?? 0);
  const optRot = Number(cam?.optimized?.rotation ?? 0);
  const origLat = Number(cam?.lat);
  const origLon = Number(cam?.lon);
  const optLat = Number(cam?.optimized?.lat);
  const optLon = Number(cam?.optimized?.lon);
  const w = Number(cam?.w ?? cam?.w_px ?? 0);
  const h = Number(cam?.h ?? cam?.h_px ?? 0);
  const mpp = Number(cam?.meters_per_pixel ?? 0.05);
  const modal = document.createElement('div');
  modal.style.position = 'fixed';
  modal.style.top = '0';
  modal.style.left = '0';
  modal.style.width = '100%';
  modal.style.height = '100%';
  modal.style.backgroundColor = 'rgba(0,0,0,0.5)';
  modal.style.zIndex = '2000';
  modal.style.display = 'flex';
  modal.style.alignItems = 'center';
  modal.style.justifyContent = 'center';
  modal.innerHTML = `
    <div style="background:white;border-radius:8px;padding:20px;max-width:600px;max-height:80vh;overflow:auto;box-shadow:0 4px 20px rgba(0,0,0,0.2)">
      <h3>${title}</h3>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;margin:20px 0">
        <div>
          <h4 style="margin-top:0">Original (EXIF/Gimbal)</h4>
          <p><strong>Rotation:</strong> ${origRot.toFixed(1)}°</p>
          <p><strong>Location:</strong> ${Number.isFinite(origLat) && Number.isFinite(origLon) ? `${origLat.toFixed(5)}, ${origLon.toFixed(5)}` : 'N/A'}</p>
          <p><strong>Size:</strong> ${w}×${h} px (${(w*mpp).toFixed(1)}×${(h*mpp).toFixed(1)} m)</p>
          <div id="origFootprint" style="width:100%;height:150px;border:1px solid #ccc;border-radius:4px;background:#f9f9f9;margin-top:10px"></div>
        </div>
        <div>
          <h4 style="margin-top:0">Optimized (COLMAP)</h4>
          <p><strong>Rotation:</strong> ${optRot.toFixed(1)}° <span style="color:${Math.abs(optRot-origRot)>2?'red':'green'}">(${optRot-origRot>0?'+':''}${(optRot-origRot).toFixed(1)}°)</span></p>
          <p><strong>Location:</strong> ${Number.isFinite(optLat) && Number.isFinite(optLon) ? `${optLat.toFixed(5)}, ${optLon.toFixed(5)}` : 'N/A'}</p>
          <p><strong>Size:</strong> ${w}×${h} px (${(w*mpp).toFixed(1)}×${(h*mpp).toFixed(1)} m)</p>
          <div id="optFootprint" style="width:100%;height:150px;border:1px solid #ccc;border-radius:4px;background:#f9f9f9;margin-top:10px"></div>
        </div>
      </div>
      <button style="width:100%;padding:10px;margin-top:20px;background:#0ea5e9;color:white;border:none;border-radius:4px;cursor:pointer;font-size:14px" onclick="this.closest('[style*=fixed]').remove()">Close</button>
    </div>
  `;
  document.body.appendChild(modal);
  setTimeout(()=>{
    drawDetailFootprint('origFootprint', origLat, origLon, origRot, w, h, mpp, '#f4b400');
    drawDetailFootprint('optFootprint', optLat, optLon, optRot, w, h, mpp, '#18b76a');
  }, 50);
}

function drawDetailFootprint(elemId, lat, lon, rotDeg, wPx, hPx, mpp, color){
  const el = document.getElementById(elemId);
  if(!el) return;
  if(!Number.isFinite(lat) || !Number.isFinite(lon) || !Number.isFinite(wPx) || !Number.isFinite(hPx) || wPx<=0 || hPx<=0) return;
  const canvas = document.createElement('canvas');
  canvas.width = 300;
  canvas.height = 150;
  el.innerHTML = '';
  el.appendChild(canvas);
  const ctx = canvas.getContext('2d');
  const cw = canvas.width, ch = canvas.height;
  const cx = cw/2, cy = ch/2;
  const rad = headingToRadians(rotDeg, false);
  const scale = Math.min((cw*0.8)/(Math.max(wPx, hPx)*mpp), (ch*0.8)/(Math.max(wPx, hPx)*mpp), 30);
  ctx.fillStyle = '#f0f0f0';
  ctx.fillRect(0,0,cw,ch);
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.globalAlpha = 0.3;
  ctx.save();
  ctx.translate(cx, cy);
  ctx.rotate(rad);
  const w = (wPx * mpp * scale);
  const h = (hPx * mpp * scale);
  ctx.fillRect(-w/2, -h/2, w, h);
  ctx.restore();
  ctx.globalAlpha = 1.0;
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.strokeRect(cx-w/2, cy-h/2, w, h);
  ctx.fillStyle = color;
  ctx.font = '12px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText(`${rotDeg.toFixed(0)}°`, cx, ch-5);
}

function ensureOptimizeControls(){
  const host = document.getElementById('optimizeMap');
  if(!host) return;
  let btnFp = document.getElementById('btnOptShowFootprints');
  if(!btnFp){
    btnFp = document.createElement('button');
    btnFp.id = 'btnOptShowFootprints';
    btnFp.type = 'button';
    btnFp.textContent = 'Show footprints';
    btnFp.className = 'pill pill-muted';
    btnFp.style.position = 'absolute';
    btnFp.style.right = '12px';
    btnFp.style.top = '12px';
    btnFp.style.zIndex = 1100;
    btnFp.style.cursor = 'pointer';
    btnFp.style.backgroundColor = 'rgba(14,165,233,0.92)';
    btnFp.style.color = '#fff';
    btnFp.style.boxShadow = '0 2px 8px rgba(0,0,0,0.25)';
    btnFp.style.border = 'none';
    btnFp.style.padding = '8px 12px';
    btnFp.style.borderRadius = '16px';
    host.style.position = host.style.position || 'relative';
    host.appendChild(btnFp);
  }
  btnFp.onclick = ()=>{
    optimizeShowFootprints = !optimizeShowFootprints;
    btnFp.textContent = optimizeShowFootprints ? 'Hide footprints' : 'Show footprints';
    btnFp.style.backgroundColor = optimizeShowFootprints ? 'rgba(14,165,233,0.98)' : 'rgba(14,165,233,0.72)';
    const ds = document.getElementById('selOptimizeDataset')?.value;
    const state = getColmapState(ds);
    if(state){
      const cams = Array.isArray(state.cameras) ? state.cameras : Object.values(state.cameras || {});
      // When global toggles, flip all camera visibility to match the global state
      cams.forEach(cam => {
        const name = cam.file || cam.name;
        if(!name) return;
        optimizeCameraVisibility[name] = optimizeShowFootprints;
      });
      renderOptimizeMap(state, cams, cams.filter(c=>c.calibrated || (c.optimized && c.optimized.lat != null)).length);
      renderColmapCameras(state);
    }
  };
}

function focusOptimizeCamera(lat, lon){
  if(!Number.isFinite(lat) || !Number.isFinite(lon)) return;
  const map = ensureOptimizeMap();
  if(!map || !optimizeMapFocusLayer) return;
  const target = [lat, lon];
  const nextZoom = Math.max(map.getZoom() || 0, 18);
  try{
    map.flyTo(target, nextZoom, { duration: 0.7 });
  }catch(_){
    map.setView(target, nextZoom);
  }
  optimizeMapFocusLayer.clearLayers();
  const pulse = L.circleMarker(target, {
    radius: 10,
    color: '#68c3ff',
    weight: 2,
    fillColor: '#68c3ff',
    fillOpacity: 0.25
  });
  pulse.addTo(optimizeMapFocusLayer);
  setTimeout(() => {
    if(optimizeMapFocusLayer){
      try{ optimizeMapFocusLayer.removeLayer(pulse); }catch(_){ }
    }
  }, 1800);
}

function updateAccurateUI(){
  const ds = getSelectedDataset();
  const state = getColmapState(ds);
  const badge = document.getElementById('accurateStatusBadge');
  const hint = document.getElementById('accurateHint');
  const chk = document.getElementById('chkAccurateLocations');
  const radColmap = document.getElementById('radAccurateColmap');
  const radOptical = document.getElementById('radAccurateOptical');
  const modeRow = document.getElementById('accurateModeRow');
  const optRow = document.getElementById('useOptimizationFromRow');
  const optSel = document.getElementById('selUseOptimizationFrom');
  const btn = document.getElementById('btnTest');
  const ready = !!(state && state.ready);
  const jobStatus = state?.job?.status;
  const wantsAccurate = !!chk?.checked;
  
  // Show/hide mode selection
  if(modeRow){
    setHidden(modeRow, !wantsAccurate);
  }
  
  // Clear radios when unchecked
  if(!wantsAccurate){
    if(radColmap) radColmap.checked = false;
    if(radOptical) radOptical.checked = false;
  }
  
  const modeColmap = wantsAccurate && !!radColmap?.checked;
  const modeOptical = wantsAccurate && !!radOptical?.checked;
  const modeSelected = modeColmap || modeOptical;
  const opticalProject = (optSel && optSel.value) ? optSel.value : "";
  const readyProjects = (datasetsCache || []).filter(d => d && d.colmap_ready);
  
  // Show/hide optimization dropdown
  if(optRow){
    setHidden(optRow, !modeOptical);
  }
  
  // Populate optimization dropdown when optical mode selected
  if(modeOptical && optSel){
    const prevVal = optSel.value;
    optSel.innerHTML = readyProjects.length
      ? '<option value="">-- Select optimized dataset --</option>'
      : '<option value="">-- No optimized datasets --</option>';
    readyProjects.forEach(d => {
      const o = document.createElement('option');
      o.value = d.name;
      o.textContent = d.name;
      optSel.appendChild(o);
    });
    if(prevVal && readyProjects.some(d => d.name === prevVal)){
      optSel.value = prevVal;
    }
  }
  
  // Disable optical radio if no projects
  if(radOptical){
    radOptical.disabled = readyProjects.length === 0;
    if(radOptical.disabled && radOptical.checked && radColmap){
      radColmap.checked = true;
      radOptical.checked = false;
    }
  }

  if(badge){
    let cls = 'pill pill-muted';
    let text = 'Select dataset';
    if(!ds){
      text = 'No dataset';
    }else if(wantsAccurate && !modeSelected){
      cls = 'pill pill-warn'; text = 'Select mode';
    }else if(modeOptical){
      if(opticalProject){
        cls = 'pill pill-ready'; text = 'Optical poses';
      }else{
        cls = 'pill pill-warn'; text = 'Select project';
      }
    }else if(ready){
      cls = 'pill pill-ready'; text = 'Ready';
    }else if(jobStatus === 'running' || jobStatus === 'queued'){
      cls = 'pill pill-warn'; text = 'Optimizing…';
    }else if(jobStatus === 'awaiting_finish'){
      cls = 'pill pill-warn'; text = 'Awaiting finish';
    }else if(state){
      cls = 'pill pill-warn'; text = 'Not optimized';
    }
    badge.className = cls;
    badge.textContent = text;
  }

  if(hint){
    if(!ds){
      hint.textContent = 'Select a dataset';
    }else if(wantsAccurate && !modeSelected){
      hint.textContent = 'Choose COLMAP or optical sync';
    }else if(modeOptical){
      hint.textContent = opticalProject ? 'Using optical project' : 'Select optimized project';
    }else if(ready){
      hint.textContent = 'Using aligned poses';
    }else if(jobStatus === 'awaiting_finish'){
      hint.textContent = 'Open Optimize tab and finish job';
    }else if(jobStatus === 'running' || jobStatus === 'queued'){
      hint.textContent = 'Optimization running…';
    }else{
      hint.textContent = 'Requires optimized poses';
    }
  }

  if(chk && btn){
    const disable = wantsAccurate && (
      !modeSelected ||
      (modeColmap && !ready) ||
      (modeOptical && !opticalProject)
    );
    btn.disabled = disable;
    btn.classList.toggle('disabled', disable);
  }

  // Update mosaic checkbox visibility: show only if camera_meta exists (rotation will happen)
  const mosaicControls = document.getElementById('mosaicControls');
  if(mosaicControls){
    const cameraMetaExists = wantsAccurate && modeSelected && (
      modeOptical ? !!opticalProject : (state?.camera_meta && Object.keys(state.camera_meta).length > 0)
    );
    mosaicControls.hidden = !cameraMetaExists;
  }
}

function describeMetaPath(path){
  if(!path) return '–';
  const parts = String(path).split(/[\\/]/).filter(Boolean);
  if(parts.length >= 2) return `${parts[parts.length-2]}/${parts[parts.length-1]}`;
  return parts.pop() || path;
}

async function loadColmapCameras(dataset, page = 0){
  if(!dataset) return;
  try {
    const res = await fetch(`${api.colmapCameras}?dataset=${encodeURIComponent(dataset)}&page=${page}&limit=50`);
    const js = await res.json();
    if(!res.ok || !js.ok) throw new Error(js.detail || 'Failed to load cameras');
    
    if(page === 0){
      colmapCamerasList = [];  // reset on first page
    }
    colmapCamerasList.push(...js.cameras);
    colmapCamerasPage = page;
    colmapCamerasHasMore = js.has_more;
    
    // Re-render list with newly loaded paginated data
    const st = getColmapState(dataset);
    if(st){
      renderColmapCameras(st);
    }
    
    return { cameras: colmapCamerasList, total: js.total, hasMore: js.has_more };
  } catch(err){
    console.error('Error loading cameras:', err);
    return null;
  }
}

function renderColmapCameras(state){
  const wrap = document.getElementById('colmapCameraList');
  const summaryEl = document.getElementById('colmapCameraSummary');
  if(!wrap) return;
  const selectedDataset = document.getElementById('selOptimizeDataset')?.value || '';
  if(!selectedDataset){
    wrap.innerHTML = `<div class="muted tiny">Select a dataset to preview cameras.</div>`;
    if(summaryEl) summaryEl.textContent = 'No dataset';
    renderOptimizeMap(null, []);
    return;
  }
  if(!state){
    wrap.innerHTML = `<div class="muted tiny">No optimization data yet.</div>`;
    if(summaryEl) summaryEl.textContent = 'Not optimized';
    renderOptimizeMap(null, []);
    return;
  }
  
  // Use ALL cameras from state for the map (not paginated)
  const allCams = Array.isArray(state.cameras) ? state.cameras : [];
  
  // Use paginated list for the sidebar display
  const displayCams = colmapCamerasList.length > 0 ? colmapCamerasList : allCams;
  
  if(!allCams.length){
    wrap.innerHTML = `<div class="muted tiny">No images detected for this dataset.</div>`;
    if(summaryEl) summaryEl.textContent = 'No cameras';
    renderOptimizeMap(state, []);
    return;
  }
  const readyCount = allCams.filter(cam => cam.calibrated || String(cam.status || '').toLowerCase() === 'calibrated').length;
  if(summaryEl){
    summaryEl.textContent = `${readyCount}/${allCams.length} calibrated`;
  }
  // Render map with ALL cameras
  renderOptimizeMap(state, allCams, readyCount);
  
  // Render list with paginated cameras
  const rows = displayCams.map(cam => {
    const rawName = cam.file || cam.name || 'image';
    const name = escapeHtml(rawName);
    const ready = cam.calibrated || (cam.status && String(cam.status).toLowerCase() === 'calibrated') || (cam.optimized && cam.optimized.lat != null);
    let tagHtml = '';
    if(ready){
      tagHtml = '<span class="tag tag-ready">ready</span>';
    }else if(cam.status){
      tagHtml = `<span class="tag tag-muted">${escapeHtml(String(cam.status))}</span>`;
    }else if(cam.has_gps === false){
      tagHtml = '<span class="tag tag-warn">no gps</span>';
    }
    const latVal = cam?.optimized?.lat ?? cam.lat;
    const lonVal = cam?.optimized?.lon ?? cam.lon;
    const altVal = cam?.optimized?.alt ?? cam.alt;
    const headingVal = cam?.optimized?.rotation ?? cam.rotation;
    const latNum = Number(latVal);
    const lonNum = Number(lonVal);
    const altNum = Number(altVal);
    const headingNum = Number(headingVal);
    const hasLatLon = Number.isFinite(latNum) && Number.isFinite(lonNum);
    const parts = [];
    if(hasLatLon){
      parts.push(`${latNum.toFixed(5)}, ${lonNum.toFixed(5)}`);
    }
    if(Number.isFinite(headingNum)){
      parts.push(`${headingNum.toFixed(1)} deg`);
    }
    if(Number.isFinite(altNum)){
      parts.push(`${altNum.toFixed(1)} m`);
    }
    const meta = parts.length ? escapeHtml(parts.join(' | ')) : null;
    const metaHtml = meta || '<span class="muted">No GPS</span>';
    const zoomTitle = hasLatLon ? `Zoom to ${rawName}` : 'No GPS data';
    const zoomAttrs = hasLatLon ? `data-lat="${latNum}" data-lon="${lonNum}"` : 'disabled';
    const zoomBtn = `<button class="camZoomBtn" type="button" ${zoomAttrs} title="${escapeHtml(zoomTitle)}" aria-label="${escapeHtml(zoomTitle)}">${CAMERA_ZOOM_ICON}</button>`;
    const isVis = optimizeCameraVisibility[rawName] === true;
    const eyeTitle = isVis ? 'Hide on map' : 'Show on map';
    const eyeBtn = `<button class="camEyeBtn iconBtn" data-filename="${escapeHtml(rawName)}" title="${eyeTitle}" style="width:24px;padding:0;font-size:16px">${isVis ? '👁️' : '👁‍🗨'}</button>`;
    const detailBtn = `<button class="camDetailBtn iconBtn" data-filename="${escapeHtml(rawName)}" title="Show before/after" style="width:24px;padding:0;font-size:14px">→</button>`;
    return `<div class="cameraRow"><div class="cameraRowHead"><div class="cameraRowHeadMain"><strong>${name}</strong> ${tagHtml}</div><div style="display:flex;gap:4px">${eyeBtn}${detailBtn}${zoomBtn}</div></div><div class="camMeta">${metaHtml}</div></div>`;
  }).join('');
  
  wrap.innerHTML = rows;
  
  // Add "Load more" button if there are more cameras to display
  if(colmapCamerasHasMore){
    wrap.insertAdjacentHTML('beforeend', `<button id="btnLoadMoreCameras" class="secondary tiny" style="width:100%; margin-top:8px;">Load more cameras (${colmapCamerasList.length} loaded)</button>`);
    document.getElementById('btnLoadMoreCameras')?.addEventListener('click', async () => {
      await loadColmapCameras(selectedDataset, colmapCamerasPage + 1);
      renderColmapCameras(state);
    });
  }

  // Attach event listeners for eye toggle and detail buttons
  wrap.querySelectorAll('.camEyeBtn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      const filename = btn.getAttribute('data-filename');
      const current = optimizeCameraVisibility[filename] === true;
      optimizeCameraVisibility[filename] = !current;
      const ds = document.getElementById('selOptimizeDataset')?.value;
      const st = getColmapState(ds);
      if(st){
        // Re-render map with ALL cameras
        const allCamsList = Array.isArray(st.cameras) ? st.cameras : [];
        renderOptimizeMap(st, allCamsList, allCamsList.filter(c=>c.calibrated || (c.optimized && c.optimized.lat != null)).length);
        renderColmapCameras(st);
      }
    });
  });
  wrap.querySelectorAll('.camDetailBtn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      const filename = btn.getAttribute('data-filename');
      // Use all cameras for detail lookup, not just paginated list
      showCameraDetail(filename, allCams);
    });
  });
  wrap.querySelectorAll('.camZoomBtn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      const lat = Number(btn.getAttribute('data-lat'));
      const lon = Number(btn.getAttribute('data-lon'));
      if(Number.isFinite(lat) && Number.isFinite(lon)){
        focusOptimizeCamera(lat, lon);
      }
    });
  });
}

function updateColmapProgress(job){
  const wrap = document.getElementById('colmapProgressWrap');
  const bar = document.getElementById('colmapProgressBar');
  const label = document.getElementById('colmapProgressText');
  if(!wrap || !bar || !label){
    return;
  }
  if(!job){
    wrap.hidden = true;
    bar.style.width = '0%';
    label.textContent = 'Idle';
    return;
  }
  wrap.hidden = false;
  const pct = Math.max(0, Math.min(100, Math.round(((job.progress ?? 0) * 100))));
  bar.style.width = `${pct}%`;
  const detail = job.progress_detail;
  let suffix = '';
  if(detail && detail.total){
    suffix = ` — ${detail.done}/${detail.total}`;
  }
  const step = job.current_step || job.status || 'running';
  label.textContent = `${step.replace(/_/g,' ')}${suffix}`;
}

function updateOptimizePanel(state){
  const dataset = document.getElementById('selOptimizeDataset')?.value;
  const pill = document.getElementById('optimizeReadyPill');
  if(pill){
    let cls = 'pill pill-muted';
    let text = dataset ? 'Not ready' : 'Select dataset';
    if(state?.ready){
      cls = 'pill pill-ready'; text = 'Ready';
    }
    pill.className = cls;
    pill.textContent = text;
  }

  setText('#colmapJobStatus', state?.job?.status || '–');
  setText('#colmapJobStep', state?.job?.current_step || '–');
  const jobIdEl = document.getElementById('colmapJobId');
  if(jobIdEl){ jobIdEl.textContent = state?.job?.id ? `Job ${state.job.id}` : '—'; }
  const metaEl = document.getElementById('colmapMetaLink');
  if(metaEl){ metaEl.textContent = state?.meta_path ? describeMetaPath(state.meta_path) : '—'; }
  const logsEl = document.getElementById('colmapLogs');
  if(logsEl){
    logsEl.textContent = (state?.logs || []).join('\n');
    logsEl.scrollTop = logsEl.scrollHeight;
  }
  renderColmapCameras(state);
  updateColmapProgress(state?.job || null);

  const finishBtn = document.getElementById('btnFinishColmap');
  if(finishBtn){ 
    // Enable if job is awaiting_finish, OR if meta file exists and not yet marked ready (job lost on reload)
    const canFinish = (state?.job?.status === 'awaiting_finish') || (state?.meta_path && !state?.ready);
    finishBtn.disabled = !canFinish;
  }
  const startBtn = document.getElementById('btnStartColmap');
  if(startBtn){
    const blocked = state?.job && ['running','queued'].includes(state.job.status);
    startBtn.disabled = !!blocked;
  }
}

function applyColmapState(dataset, state){
  cacheColmapState(dataset, state);
  const paramSnapshot = state?.job?.params ?? state?.saved_params ?? {};
  rememberColmapParams(dataset, paramSnapshot);
  if(dataset === getSelectedDataset()){
    updateAccurateUI();
  }
  const optSel = document.getElementById('selOptimizeDataset');
  if(optSel && optSel.value === dataset){
    maybeApplySavedColmapParams(dataset);
    updateOptimizePanel(state);
    // Load paginated cameras in background (doesn't block rendering)
    loadColmapCameras(dataset, 0).catch(err => console.error('Failed to load cameras:', err));
  }
}

function clearColmapPoll(){
  if(colmapPollHandle){
    clearInterval(colmapPollHandle);
    colmapPollHandle = null;
    colmapPollDataset = null;
  }
}

function scheduleColmapPoll(dataset, state){
  const status = state?.job?.status;
  const shouldPoll = !!(status && ['running','queued','awaiting_finish'].includes(status));
  if(!shouldPoll){
    if(colmapPollDataset === dataset){
      clearColmapPoll();
    }
    return;
  }
  if(colmapPollDataset === dataset && colmapPollHandle){
    return;
  }
  clearColmapPoll();
  colmapPollDataset = dataset;
  colmapPollHandle = setInterval(()=> refreshColmapState({ dataset, silent: true }), 5000);
}

async function refreshColmapState({ dataset, silent } = {}){
  const selOpt = document.getElementById('selOptimizeDataset');
  const ds = dataset || selOpt?.value || getSelectedDataset();
  if(!ds) return null;
  try{
    const res = await fetch(`${api.colmapState}?dataset=${encodeURIComponent(ds)}`);
    const js = await res.json();
    if(!res.ok || !js.ok) throw new Error(js.detail || 'Failed to load COLMAP state');
    applyColmapState(ds, js.state);
    scheduleColmapPoll(ds, js.state);
    if(!silent){
      if(js.state?.job?.status === 'awaiting_finish'){
        setColmapStatus('COLMAP finished. Click "Mark finished" to accept.', 'warn');
      }else if(js.state?.job?.status === 'running'){
        setColmapStatus('COLMAP optimization running…');
      }else{
        setColmapStatus('');
      }
    }
    return js.state;
  }catch(err){
    if(!silent){
      console.warn('colmap state', err);
      setColmapStatus(err.message || 'Failed to fetch COLMAP state', 'warn');
    }
    return null;
  }
}

function buildColmapParams(){
  const params = {};
  const matcher = document.getElementById('selColmapMatcher')?.value;
  if(matcher) params.matcher = matcher;
  const cameraModel = document.getElementById('selColmapCameraModel')?.value;
  if(cameraModel) params.camera_model = cameraModel;
  const overlapVal = document.getElementById('inpColmapSeqOverlap')?.value;
  if(overlapVal !== undefined && overlapVal !== ''){
    const val = parseInt(overlapVal, 10);
    if(!Number.isNaN(val)) params.seq_overlap = val;
  }
  const tri = document.getElementById('inpColmapTriangulation')?.value;
  if(tri !== undefined && tri !== ''){
    const val = Number(tri);
    if(!Number.isNaN(val)) params.min_triangulation_angle = val;
  }
  const minSizeVal = document.getElementById('inpColmapMinSize')?.value;
  if(minSizeVal !== undefined && minSizeVal !== ''){
    const val = parseInt(minSizeVal, 10);
    if(!Number.isNaN(val)) params.min_model_size = val;
  }
  const minMatchesVal = document.getElementById('inpColmapMinMatches')?.value;
  if(minMatchesVal !== undefined && minMatchesVal !== ''){
    const val = parseInt(minMatchesVal, 10);
    if(!Number.isNaN(val)) params.min_num_matches = val;
  }
  const maxImageVal = document.getElementById('inpColmapMaxImage')?.value;
  if(maxImageVal !== undefined && maxImageVal !== ''){
    const val = parseInt(maxImageVal, 10);
    if(!Number.isNaN(val)) params.max_image_size = val;
  }
  const peakVal = document.getElementById('inpColmapPeakThresh')?.value;
  if(peakVal !== undefined && peakVal !== ''){
    const val = Number(peakVal);
    if(!Number.isNaN(val)) params.peak_threshold = val;
  }
  const edgeVal = document.getElementById('inpColmapEdgeThresh')?.value;
  if(edgeVal !== undefined && edgeVal !== ''){
    const val = Number(edgeVal);
    if(!Number.isNaN(val)) params.edge_threshold = val;
  }
  const maxFeatVal = document.getElementById('inpColmapMaxFeatures')?.value;
  if(maxFeatVal !== undefined && maxFeatVal !== ''){
    const val = parseInt(maxFeatVal, 10);
    if(!Number.isNaN(val)) params.max_num_features = val;
  }
  const threadsVal = document.getElementById('inpColmapThreads')?.value;
  if(threadsVal !== undefined && threadsVal !== ''){
    const val = parseInt(threadsVal, 10);
    if(!Number.isNaN(val)) params.num_threads = val;
  }
  const initInliersVal = document.getElementById('inpColmapInitInliers')?.value;
  if(initInliersVal !== undefined && initInliersVal !== ''){
    const val = parseInt(initInliersVal, 10);
    if(!Number.isNaN(val)) params.init_min_num_inliers = val;
  }
  const absInliersVal = document.getElementById('inpColmapAbsInliers')?.value;
  if(absInliersVal !== undefined && absInliersVal !== ''){
    const val = parseInt(absInliersVal, 10);
    if(!Number.isNaN(val)) params.abs_pose_min_num_inliers = val;
  }
  const maxOverlapVal = document.getElementById('inpColmapMaxOverlap')?.value;
  if(maxOverlapVal !== undefined && maxOverlapVal !== ''){
    const val = parseInt(maxOverlapVal, 10);
    if(!Number.isNaN(val)) params.max_model_overlap = val;
  }
  const maxModelsVal = document.getElementById('inpColmapMaxModels')?.value;
  if(maxModelsVal !== undefined && maxModelsVal !== ''){
    const val = parseInt(maxModelsVal, 10);
    if(!Number.isNaN(val)) params.max_num_models = val;
  }
  const exhBlockVal = document.getElementById('inpColmapExhBlock')?.value;
  if(exhBlockVal !== undefined && exhBlockVal !== ''){
    const val = parseInt(exhBlockVal, 10);
    if(!Number.isNaN(val)) params.exhaustive_block_size = val;
  }
  const baFocal = document.getElementById('chkColmapBaFocal');
  if(baFocal){
    params.ba_refine_focal_length = !!baFocal.checked;
  }
  const baPp = document.getElementById('chkColmapBaPpoint');
  if(baPp){
    params.ba_refine_principal_point = !!baPp.checked;
  }
  const baExtra = document.getElementById('chkColmapBaExtra');
  if(baExtra){
    params.ba_refine_extra_params = !!baExtra.checked;
  }
  const useGpu = document.getElementById('chkColmapUseGpu');
  if(useGpu){
    params.use_gpu = !!useGpu.checked;
  }
  const maxReprojVal = document.getElementById('inpColmapMaxReprojError')?.value;
  if(maxReprojVal !== undefined && maxReprojVal !== ''){
    const val = Number(maxReprojVal);
    if(!Number.isNaN(val)) params.max_reproj_error = val;
  }
  const baGlobalIterVal = document.getElementById('inpColmapBaGlobalIter')?.value;
  if(baGlobalIterVal !== undefined && baGlobalIterVal !== ''){
    const val = parseInt(baGlobalIterVal, 10);
    if(!Number.isNaN(val)) params.ba_global_max_iterations = val;
  }
  const baLocalIterVal = document.getElementById('inpColmapBaLocalIter')?.value;
  if(baLocalIterVal !== undefined && baLocalIterVal !== ''){
    const val = parseInt(baLocalIterVal, 10);
    if(!Number.isNaN(val)) params.ba_local_max_iterations = val;
  }
  return params;
}

function ensureColmapParamDefaults(){
  if(Object.keys(colmapParamDefaults).length){
    return;
  }
  Object.entries(COLMAP_PARAM_FIELDS).forEach(([key, conf]) => {
    const el = document.getElementById(conf.id);
    if(!el) return;
    if(conf.type === 'checkbox'){
      colmapParamDefaults[key] = !!el.checked;
    }else{
      colmapParamDefaults[key] = el.value ?? '';
    }
  });
}

function normalizeColmapParams(raw){
  const out = {};
  if(!raw || typeof raw !== 'object'){
    return out;
  }
  Object.keys(COLMAP_PARAM_FIELDS).forEach(key => {
    if(!Object.prototype.hasOwnProperty.call(raw, key)){
      return;
    }
    const val = raw[key];
    if(val === undefined || val === null || val === ''){
      return;
    }
    if(COLMAP_PARAM_FIELDS[key].type === 'checkbox'){
      out[key] = !!val;
    }else{
      out[key] = val;
    }
  });
  return out;
}

function hashColmapParams(raw){
  const normalized = normalizeColmapParams(raw);
  const ordered = {};
  Object.keys(COLMAP_PARAM_FIELDS).forEach(key => {
    if(Object.prototype.hasOwnProperty.call(normalized, key)){
      ordered[key] = normalized[key];
    }
  });
  return JSON.stringify(ordered);
}

function rememberColmapParams(dataset, params){
  if(!dataset) return;
  colmapSavedParams.set(dataset, normalizeColmapParams(params));
}

function applyColmapParamsToForm(dataset, params){
  if(!dataset) return;
  ensureColmapParamDefaults();
  const normalized = normalizeColmapParams(params);
  Object.entries(COLMAP_PARAM_FIELDS).forEach(([key, conf]) => {
    const el = document.getElementById(conf.id);
    if(!el) return;
    const hasCustom = Object.prototype.hasOwnProperty.call(normalized, key);
    if(conf.type === 'checkbox'){
      el.checked = hasCustom ? !!normalized[key] : !!colmapParamDefaults[key];
    }else if(hasCustom){
      el.value = normalized[key];
    }else{
      const fallback = colmapParamDefaults[key];
      el.value = fallback === undefined || fallback === null ? '' : fallback;
    }
  });
  colmapPrefillState = { dataset, hash: hashColmapParams(normalized) };
}

function maybeApplySavedColmapParams(dataset){
  if(!dataset) return;
  const saved = colmapSavedParams.get(dataset) || {};
  const savedHash = hashColmapParams(saved);
  if(colmapPrefillState.dataset === dataset && colmapPrefillState.hash === savedHash){
    return;
  }
  applyColmapParamsToForm(dataset, saved);
}

async function startColmapJob(){
  const dataset = document.getElementById('selOptimizeDataset')?.value;
  if(!dataset){
    setColmapStatus('Select a dataset to optimize.', 'warn');
    return;
  }
  const params = buildColmapParams();
  rememberColmapParams(dataset, params);
  colmapPrefillState = { dataset, hash: hashColmapParams(params) };
  async function _launch(confirmReset){
    const fd = new FormData();
    fd.append('dataset', dataset);
    if(Object.keys(params).length){
      fd.append('params', JSON.stringify(params));
    }
    if(confirmReset){
      fd.append('confirm_reset', 'true');
    }
    setColmapStatus(confirmReset ? 'Resetting previous COLMAP run…' : 'Starting COLMAP…');
    const res = await fetch(api.colmapStart, { method:'POST', body: fd });
    const js = await res.json().catch(()=>({}));
    if(res.status === 412){
      // Backend indicates previous results exist; ask user and retry with confirm_reset
      const ok = window.confirm('Previous COLMAP results exist. Restarting will delete them. Continue?');
      if(ok){
        return _launch(true);
      }
      setColmapStatus('Restart cancelled.', 'warn');
      return;
    }
    if(!res.ok || !js.ok){
      throw new Error(js.detail || 'Failed to start COLMAP');
    }
    setColmapStatus('COLMAP started.', 'ok');
    applyColmapState(dataset, js.state || null);
    scheduleColmapPoll(dataset, js.state);
  }

  try{
    await _launch(false);
  }catch(err){
    console.error('colmap start', err);
    setColmapStatus(err.message || 'Failed to start COLMAP', 'err');
  }
}

async function finishColmapJob(){
  const dataset = document.getElementById('selOptimizeDataset')?.value;
  const state = getColmapState(dataset);
  if(!dataset){
    setColmapStatus('Select a dataset to finish.', 'warn');
    return;
  }
  // If backend was restarted, job may be missing from state; allow finish when meta exists and not ready yet
  let jobId = state?.job?.id;
  if(!jobId){
    if(state?.meta_path && !state?.ready){
      jobId = `resume-${dataset}`;
    } else {
      setColmapStatus('No COLMAP job ready to finish.', 'warn');
      return;
    }
  }
  const fd = new FormData();
  fd.append('dataset', dataset);
  fd.append('job_id', jobId);
  setColmapStatus('Marking dataset as ready…');
  try{
    const res = await fetch(api.colmapFinish, { method:'POST', body: fd });
    const js = await res.json();
    if(!res.ok || !js.ok) throw new Error(js.detail || 'Failed to finalize COLMAP results');
    setColmapStatus('Dataset ready for accurate locations.', 'ok');
    applyColmapState(dataset, js.state || null);
    scheduleColmapPoll(dataset, js.state);
  }catch(err){
    console.error('colmap finish', err);
    setColmapStatus(err.message || 'Failed to finalize COLMAP results', 'err');
  }
}

function syncOptimizeDatasetFromTest(){
  const ds = getSelectedDataset();
  const sel = document.getElementById('selOptimizeDataset');
  if(ds && sel){
    sel.value = ds;
    onOptimizeDatasetChange();
  }
}

function onTestDatasetChange(){
  const ds = getSelectedDataset();
  const sel = document.getElementById('selOptimizeDataset');
  if(ds && sel){
    sel.value = ds;
  }
  const cached = getColmapState(ds);
  if(cached){
    updateAccurateUI();
  }else if(ds){
    refreshColmapState({ dataset: ds, silent: true });
  }else{
    updateAccurateUI();
  }
}

function onOptimizeDatasetChange(){
  const sel = document.getElementById('selOptimizeDataset');
  const ds = sel?.value;
  if(!ds){
    updateOptimizePanel(null);
    colmapCamerasPage = 0;
    colmapCamerasList = [];
    return;
  }
  // Reset pagination for new dataset
  colmapCamerasPage = 0;
  colmapCamerasList = [];
  colmapCamerasHasMore = false;
  
  const cached = getColmapState(ds);
  if(cached){
    updateOptimizePanel(cached);
    scheduleColmapPoll(ds, cached);
  }else{
    refreshColmapState({ dataset: ds });
  }
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

  const opacityVal = rec.layer.getOpacity ? (rec.layer.getOpacity() * 100).toFixed(0) : 100;
  
  // Only show Style option for GeoJSON overlays (not TIF/raster)
  const isTifRaster = rec.type === 'tif_overlay' || rec.type === 'raster';
  const showStyle = !isTifRaster;
  const styleItem = showStyle ? '<li data-action="style">Style…</li>' : '';
  
  menu.innerHTML = `
    <ul>
      <li data-action="zoom">Zoom to layer</li>
      ${styleItem}
      <li data-action="download">Download</li>
      <li data-action="delete">Delete</li>
      <li style="border-top: 1px solid #ddd; padding-top: 8px; margin-top: 8px; padding-left: 12px; padding-right: 12px;">
        <div style="display: flex; align-items: center; gap: 8px; font-size: 12px;">
          <label style="margin: 0;">Opacity:</label>
          <input type="range" class="layerOpacitySlider" data-key="${escapeHtml(key)}" min="0" max="100" value="${opacityVal}" style="flex: 1; max-width: 80px;">
          <span class="opacityVal" style="min-width: 30px; text-align: right;">${opacityVal}%</span>
        </div>
      </li>
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
    let bounds = null;
    
    // First try to get bounds from the layer itself
    try {
      bounds = computeLayerBounds(rec.layer);
    } catch(e) { }
    
    // If no bounds computed, try child layer bounds from options
    if (!bounds && rec.layer && rec.layer.eachLayer) {
      try {
        let childBounds = null;
        rec.layer.eachLayer(l => {
          const b = l?.options?.bounds;
          if (b && typeof b.isValid === 'function' && b.isValid()) {
            childBounds = childBounds ? childBounds.extend(b) : b;
          }
        });
        if (childBounds) {
          bounds = childBounds;
        }
      } catch (e) { }
    }
    
    // If no bounds computed, try stored bounds
    if (!bounds && rec.bounds) {
      // Check if rec.bounds is a LatLngBounds object or raw array
      if (typeof rec.bounds.isValid === 'function' && rec.bounds.isValid()) {
        // Already a LatLngBounds object
        bounds = rec.bounds;
      } else if (Array.isArray(rec.bounds) && rec.bounds.length === 2) {
        // Convert raw bounds array [[lat1, lon1], [lat2, lon2]] to LatLngBounds
        try {
          bounds = L.latLngBounds(rec.bounds);
        } catch(e) { }
      }
    }
    
    if (bounds && typeof bounds.isValid === 'function' && bounds.isValid()) {
      MAP.fitBounds(bounds, { padding: [20, 20] });
    }
    closeLayerMenu();
    return;
  }

  if (action === 'style') {
    styleTarget = { name: key, info: rec };

    const stSelBlk = document.getElementById('stColorByBlock');
    const stSel    = document.getElementById('stColorBy');
    const stCat    = document.getElementById('stCatList');

    // Check if this is a GeoJSON layer with features
    if (!rec?.data || !rec.data.features || rec.data.features.length === 0){
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

    // Collect all properties from features
    const props = new Set();
    try{
      rec.data.features.forEach(f => {
        if (f?.properties) Object.keys(f.properties).forEach(k => props.add(k));
      });
    }catch(_){}

    // Filter properties: only show those with less than 20 unique values
    const CATEGORY_NONE = '__none__';
    const validProps = new Set();
    for (const p of props){
      const uniqueVals = uniqueValuesFromGJ(rec.data, p);
      if (uniqueVals.length > 0 && uniqueVals.length < 20) {
        validProps.add(p);
      }
    }

    // Helper: pick a sensible default
    function pickDefaultProp(){
      // Try common property names first
      const tryProps = ['class_name','class_id','category','type','name', ...validProps];
      for (const p of tryProps){
        if (!p || p === CATEGORY_NONE || !validProps.has(p)) continue;
        return p;
      }
      // Return first valid prop or CATEGORY_NONE
      return validProps.size > 0 ? Array.from(validProps)[0] : CATEGORY_NONE;
    }

    const currentProp = rec.categorical?.prop ?? pickDefaultProp();

    // Fill dropdown
    stSel.innerHTML = '';
    const optNone = document.createElement('option');
    optNone.value = CATEGORY_NONE; optNone.textContent = 'None';
    stSel.appendChild(optNone);
    Array.from(validProps).sort().forEach(k=>{
      const o = document.createElement('option');
      o.value = k; o.textContent = k;
      if (k === currentProp) o.selected = true;
      stSel.appendChild(o);
    });

    // Build editor for selected prop
    const choose = (val)=>{
      rebuildCategoryEditors(val, key);
    };

    choose(currentProp);
    stSel.onchange = ()=> choose(stSel.value);

    document.getElementById('styleModal').classList.remove('hidden');
    closeLayerMenu();
    return;
  }

  if (action === 'download') {
    downloadLayer(key, rec);
    closeLayerMenu();
    return;
  }

  if (action === 'delete') {
    // Show confirmation dialog
    const layerName = key || 'this overlay';
    const confirmDelete = confirm(`Are you sure you want to delete "${layerName}"?`);
    
    if (!confirmDelete) {
      closeLayerMenu();
      return;
    }
    
    // Remove from map
    if (MAP.hasLayer(rec.layer)) {
      MAP.removeLayer(rec.layer);
    }
    
    // Delete from backend if not temporary
    if (rec.overlay_id && !rec.temporary) {
      fetch('/api/delete_overlay', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ overlay_id: rec.overlay_id })
      }).then(response => {
        if (response.ok) {
          console.log(`✓ Deleted overlay "${layerName}" from backend`);
        }
      }).catch(err => console.error('Failed to delete overlay from backend:', err));
    }
    
    delete overlayRegistry[key];
    refreshLayersPanel();
    closeLayerMenu();
    return;
  }

});

function downloadLayer(key, rec){
  if (!rec) return;
  
  let dataStr, filename, fileType, url;
  
  // Handle GeoJSON layers
  if (rec.data && rec.data.features) {
    dataStr = JSON.stringify(rec.data, null, 2);
    filename = `${key}.geojson`;
    fileType = 'application/json';
    
    // Create download link
    const blob = new Blob([dataStr], { type: fileType });
    url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  } else if (rec.type === 'tif_overlay' || rec.type === 'raster') {
    // For TIF overlays, request download from backend
    // Use the overlay_id if available, otherwise use the key
    const downloadUrl = rec.overlay_id 
      ? `/api/download_overlay?overlay_id=${encodeURIComponent(rec.overlay_id)}`
      : null;
    
    if (downloadUrl) {
      const link = document.createElement('a');
      link.href = downloadUrl;
      link.download = `${key}.tif`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  }
}

function openOrthoMenu(btn, sessionName, layerDefs){
  const orthoMenuEl = document.getElementById('orthoMenu');
  if (orthoMenuEl) orthoMenuEl.remove();
  
  const menu = document.createElement('div');
  menu.id = 'orthoMenu';
  menu.className = 'imageMenu';
  
  menu.innerHTML = `
    <ul>
      <li data-action="zoom">Zoom to layer</li>
      <li data-action="download">Download</li>
    </ul>
  `;
  
  document.body.appendChild(menu);
  
  // Position the menu
  const r = btn.getBoundingClientRect();
  const vw = document.documentElement.clientWidth;
  const pad = 8;
  const mw = menu.offsetWidth || 150;
  const mh = menu.offsetHeight || 80;
  
  let left = Math.round(r.left);
  let top = Math.round(r.bottom + 4);
  
  if (left + mw > vw - pad) {
    left = Math.round(r.right - mw);
  }
  
  if (top + mh > window.innerHeight - pad) {
    top = Math.round(r.top - mh - 4);
  }
  
  menu.style.left = Math.max(pad, left) + 'px';
  menu.style.top = Math.max(pad, top) + 'px';
  
  // Handle menu clicks
  menu.addEventListener('click', (e) => {
    const action = e.target.closest('li')?.dataset.action;
    if (!action) return;
    
    if (action === 'zoom') {
      if (TIF_TILE_BOUNDS && TIF_TILE_BOUNDS.isValid && TIF_TILE_BOUNDS.isValid()) {
        MAP.fitBounds(TIF_TILE_BOUNDS.pad(0.2));
      }
    } else if (action === 'download') {
      // Download ortho - request stitch from backend if multiple tiles
      const downloadUrl = `/api/download_ortho?session=${encodeURIComponent(sessionName)}`;
      const link = document.createElement('a');
      link.href = downloadUrl;
      link.download = `${sessionName}_ortho.tif`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
    
    menu.remove();
  });
  
  // Close on outside click
  const closeMenu = (e) => {
    if (!menu.contains(e.target) && !btn.contains(e.target)) {
      menu.remove();
      document.removeEventListener('click', closeMenu);
    }
  };
  setTimeout(() => document.addEventListener('click', closeMenu), 0);
}

function openImageMenu(btn){
  const imageMenuEl = document.getElementById('imageMenu');
  if (imageMenuEl) imageMenuEl.remove();
  
  const id = btn.getAttribute('data-id');
  const rec = imageCatalog.find(x => x.id === id);
  if (!rec) return;
  
  const menu = document.createElement('div');
  menu.id = 'imageMenu';
  menu.className = 'imageMenu';
  menu.dataset.id = id;
  
  menu.innerHTML = `
    <ul>
      <li data-action="zoom">Zoom to layer</li>
      <li data-action="download">Download</li>
    </ul>
  `;
  
  document.body.appendChild(menu);
  
  // Position the menu
  const r = btn.getBoundingClientRect();
  const vw = document.documentElement.clientWidth;
  const pad = 8;
  const mw = menu.offsetWidth || 150;
  const mh = menu.offsetHeight || 80;
  
  let left = Math.round(r.left);
  let top = Math.round(r.bottom + 4);
  
  if (left + mw > vw - pad) {
    left = Math.round(r.right - mw);
  }
  
  if (top + mh > window.innerHeight - pad) {
    top = Math.round(r.top - mh - 4);
  }
  
  menu.style.left = Math.max(pad, left) + 'px';
  menu.style.top = Math.max(pad, top) + 'px';
  
  // Handle menu clicks
  menu.addEventListener('click', (e) => {
    const action = e.target.closest('li')?.dataset.action;
    if (!action) return;
    
    if (action === 'zoom') {
      if (rec.bounds && rec.bounds.isValid && rec.bounds.isValid()) {
        MAP.fitBounds(rec.bounds.pad(0.2));
      } else {
        const ov = imageOverlays.get(rec.id);
        if (ov && ov.getBounds) {
          const b = ov.getBounds();
          if (b.isValid()) MAP.fitBounds(b.pad(0.2));
        }
      }
    } else if (action === 'download') {
      // Download the image
      if (rec.url) {
        const link = document.createElement('a');
        link.href = rec.url;
        link.download = rec.name || rec.id;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
      }
    }
    
    imageMenuEl?.remove();
  });
  
  // Close on outside click
  const closeMenu = (e) => {
    if (!menu.contains(e.target) && !btn.contains(e.target)) {
      menu.remove();
      document.removeEventListener('click', closeMenu);
    }
  };
  setTimeout(() => document.addEventListener('click', closeMenu), 0);
}

// Layer opacity slider in menu
document.addEventListener('input', (e)=>{
  const slider = e.target.closest('.layerOpacitySlider');
  if (!slider) return;
  
  const key = slider.dataset.key;
  const rec = overlayRegistry[key];
  if (!rec || !rec.layer) return;
  
  const opacity = slider.value / 100;
  // Update the layer opacity
  if (rec.layer.setOpacity) {
    rec.layer.setOpacity(opacity);
  } else if (rec.layer.setStyle) {
    rec.layer.setStyle({ opacity, fillOpacity: opacity });
  } else if (rec.layer.eachLayer) {
    let count = 0;
    rec.layer.eachLayer(l => {
      if (l.setOpacity) {
        l.setOpacity(opacity);
        count += 1;
      } else if (l.setStyle) {
        l.setStyle({ opacity, fillOpacity: opacity });
        count += 1;
      }
    });
  } else {
  }
  
  // Update the display value in the menu
  const opacityVal = slider.closest('li').querySelector('.opacityVal');
  if (opacityVal) opacityVal.textContent = slider.value + '%';
});

document.addEventListener('click', (e)=>{
  const btn = e.target.closest('.camZoomBtn');
  if(!btn || btn.disabled) return;
  const lat = Number(btn.dataset.lat);
  const lon = Number(btn.dataset.lon);
  if(!Number.isFinite(lat) || !Number.isFinite(lon)) return;
  e.preventDefault();
  focusOptimizeCamera(lat, lon);
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
  const selTest = $("#selTestFolder");
  const prevTest = selTest?.value || null;
  if(selTest){
    selTest.innerHTML = "";
    list.forEach(d => {
      const o = document.createElement("option");
      o.value = d.name;
      o.textContent = `${d.name} (${d.count})`;
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
      o.textContent = d.name;
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
      if(id === "tab-map" && MAP){ 
        setTimeout(()=>{
          MAP.invalidateSize();
          if(pendingMapBounds){
            MAP.fitBounds(pendingMapBounds.bounds, pendingMapBounds.options);
            pendingMapBounds = null;
          }
        }, 30);
      }
      if(id === "tab-optimize"){ invalidateOptimizeMap(); }
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
}

// ---------- datasets/models/sessions ----------
async function loadDatasets(){
  const res = await fetch(api.datasets);
  const js = await res.json();
  if(js.ok){
    datasetsCache = js.datasets || [];
    populateFolders(js.datasets);
    onTestDatasetChange();
    onOptimizeDatasetChange();
  }
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
  const backend = getSelectedBackend();
  fd.append("backend", backend);
  if(backend === 'yolo'){
    const yo = getYoloOptions();
    fd.append('yolo_family', yo.family);

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
  const useThermal = (modelChannelCount === 3 && (!!(mmeta && mmeta.thermal_only) || !!(mmeta && mmeta.thermal_used)));
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
  fd.append("clear_existing", clearExisting ? "true" : "false");
  fd.append("test_threshold", testThreshold);
  if(accurateMode === "colmap") fd.append("accurate_locations", "true");
  if(accurateMode === "optical" && optimizationProject) fd.append("optimization_project", optimizationProject);
  const wantsMosaic = document.getElementById("chkMosaicImages")?.checked;
  if(wantsMosaic) fd.append("mosaic_enabled", "true");
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

    if(!js.ok) throw new Error("test failed");

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
  const base = String(name).split("/").pop();
  if(!base) return null;
  return base.replace(/\.[^.]+$/, "").toLowerCase();
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

    div.innerHTML = `
      <img src="${item.thumb}" alt="${item.file}">
      <div class="meta" title="${item.file}">${item.file}</div>
      ${detectionBadge}
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

function applyMapDetectionFilter() {
  const chk = document.getElementById('chkMapShowOnlyDetections');
  const on = chk?.checked || false;
  mapDetectionFilterActive = on;

  if (on) {
    // Only turn OFF non-detection images; keep current states for detected images
    for (const rec of imageCatalog) {
      if (!(rec.n && rec.n > 0) && rec.on) {
        toggleImageOverlay(rec.id, false);
      }
    }
  }

  renderImagesList();
}

// ---------- map ----------
function initMap(){
  const street = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", { maxZoom: 22, attribution: "&copy; OpenStreetMap" });
  const sat = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", { maxZoom: 22, attribution: "&copy; Esri" });

  MAP = L.map("map", { layers: [street] });
  baseLayers = { "Street": street, "Satellite": sat };
  L.control.layers(baseLayers, {}, { position: "topleft" }).addTo(MAP);
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




function rebuildCategoryEditors(prop, layerKey = "Anomalies"){
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
  const rec = overlayRegistry["Anomalies"];
  const cat = rec?.categorical;
  return styleForCategoricalFeature(f, fallback, cat);
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


async function applySessionToMap(sessionName){
  // 0) clear any previous GeoTIFF tiles
  removeTifTiles();

  // 1) session summary (urls for geojsons)
  const res = await fetch(`/api/session_summary?session=${encodeURIComponent(sessionName)}`, { cache: 'no-store' });
  if (!res.ok) { console.warn('session_summary failed'); return; }
  const sum = await res.json();
  // cache the session summary so loadImagesCatalog can prefer rotated_images when available
  lastLoadedSessionSummary = sum || null;
  rotatedImagesLookup = null;

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
    updateMapDetectionFilterVisibility(true);
    updateImageListButtonsVisibility(true);
    const chkMapFilterAnomalies = document.getElementById('chkMapFilterAnomalies');
    if (chkMapFilterAnomalies) chkMapFilterAnomalies.checked = false;
    anomaliesFilterActive = false;
    const b = createTifTileGroup(tiles.layers);
    TIF_TILE_GROUP  = b.group;
    TIF_TILE_LAYERS = b.layers;
    TIF_TILE_BOUNDS = b.firstBounds;
    TIF_TILE_BOUNDS = b.firstBounds;

    // show controller row inside Images list (replaces normal images there)
    installTilesIntoImagesList(sessionName, tiles.layers);

    // fit to raster on first load
    if (b.firstBounds){
      try{ MAP.fitBounds(b.firstBounds, { padding:[20,20] }); }catch(_){}
    }
  } else {
    updateMapDetectionFilterVisibility(false);
    updateImageListButtonsVisibility(false);
    // Fallback: point markers loaded from images.geojson
    await loadImagesCatalog(sessionName, imagesUrl);
    // Apply anomalies filter if enabled
    applyAnomaliesFilter();
    
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
    const feats = Array.isArray(modifiedGJ?.features) ? modifiedGJ.features : [];
    for (const f of feats){
      if (f?.geometry?.type !== 'Point') continue;

      const [lng, lat] = f.geometry.coordinates || [];
      if (typeof lat !== 'number' || typeof lng !== 'number') continue;

      // filename is in properties.image (per your sample)
      const file = f?.properties?.image || f?.properties?.file || f?.properties?.name;
      if (!file) continue;

      // Use overlay PNG path (contains annotations/predictions)
      // Do not use rotated_images (those are for inference only)
      const stem = file.replace(/\.[^.]+$/, '');
      const defaultOverlay = `/media/sessions/${encodeURIComponent(sessionName)}/overlays/${encodeURIComponent(stem)}.png`;
      const url = defaultOverlay;

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

      imageCatalog.push({ id: file, name: file, url, bounds, on: false, rotation: storedRotation, corners: Array.isArray(corners) ? corners : null, n: detectionCount });
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

  const btnRefreshFolders = $("#btnRefreshFolders");
  if(btnRefreshFolders) btnRefreshFolders.addEventListener("click", loadDatasets);
  const selTestFolder = document.getElementById('selTestFolder');
  if(selTestFolder){ selTestFolder.addEventListener('change', onTestDatasetChange); }
  const btnRefreshModels = $("#btnRefreshModels");
  if(btnRefreshModels) btnRefreshModels.addEventListener("click", ()=> loadModels(getSelectedBackend(), '#selModelFolder'));
  const btnOpenUpload = $("#btnOpenUploadModal");
  if(btnOpenUpload) btnOpenUpload.addEventListener("click", openUploadModal);
  const btnCloseUpload = $("#btnCloseUploadModal");
  if(btnCloseUpload) btnCloseUpload.addEventListener("click", ()=>{ closeUploadModal(); resetUploadProgress(); });
  const btnCancelUpload = $("#btnCancelUpload");
  if(btnCancelUpload) btnCancelUpload.addEventListener("click", ()=>{ closeUploadModal(); resetUploadProgress(); });
  const btnStartUpload = $("#btnStartUpload");
  if(btnStartUpload) btnStartUpload.addEventListener("click", startUpload);

  const btnTrain = $("#btnTrain");
  if(btnTrain) btnTrain.addEventListener("click", startTraining);
  const btnCancelTrain = $("#btnCancelTrain");
  if(btnCancelTrain) btnCancelTrain.addEventListener("click", cancelTraining);

  const btnTest = $("#btnTest");
  if(btnTest) btnTest.addEventListener("click", runTest);
  const btnCancelTest = $("#btnCancelTest");
  if(btnCancelTest) btnCancelTest.addEventListener("click", cancelTest);
  
  // Detection filter toggles
  const chkShowOnlyDetections = document.getElementById('chkShowOnlyDetections');
  if(chkShowOnlyDetections){ chkShowOnlyDetections.addEventListener('change', _applyDetectionFilter); }
  const chkMapShowOnlyDetections = document.getElementById('chkMapShowOnlyDetections');
  if(chkMapShowOnlyDetections){ chkMapShowOnlyDetections.addEventListener('change', applyMapDetectionFilter); }
  const chkMapFilterAnomalies = document.getElementById('chkMapFilterAnomalies');
  if(chkMapFilterAnomalies){ chkMapFilterAnomalies.addEventListener('change', applyAnomaliesFilter); }
  
  const chkAccurate = document.getElementById('chkAccurateLocations');
  if(chkAccurate){ chkAccurate.addEventListener('change', updateAccurateUI); }
  document.getElementById('radAccurateColmap')?.addEventListener('change', updateAccurateUI);
  document.getElementById('radAccurateOptical')?.addEventListener('change', updateAccurateUI);
  document.getElementById('selUseOptimizationFrom')?.addEventListener('change', updateAccurateUI);
  const btnGoOptimize = document.getElementById('btnGoOptimize');
  if(btnGoOptimize){
    btnGoOptimize.addEventListener('click', (e)=>{
      e.preventDefault();
      syncOptimizeDatasetFromTest();
      switchToTab('tab-optimize');
      refreshColmapState();
    });
  }

  const colmapInfoModal = document.getElementById('colmapInfoModal');
  const openColmapInfo = ()=> colmapInfoModal?.classList.remove('hidden');
  const closeColmapInfo = ()=> colmapInfoModal?.classList.add('hidden');
  document.getElementById('btnColmapInfo')?.addEventListener('click', openColmapInfo);
  document.getElementById('btnCloseColmapInfo')?.addEventListener('click', closeColmapInfo);
  document.getElementById('btnColmapInfoDone')?.addEventListener('click', closeColmapInfo);
  const openColmapConfig = ()=> document.getElementById('colmapConfigModal')?.classList.remove('hidden');
  const closeColmapConfig = ()=> document.getElementById('colmapConfigModal')?.classList.add('hidden');
  document.getElementById('btnColmapConfig')?.addEventListener('click', openColmapConfig);
  document.getElementById('btnCloseColmapConfig')?.addEventListener('click', closeColmapConfig);
  document.getElementById('btnColmapConfigDone')?.addEventListener('click', closeColmapConfig);

  const lnkLogsTest = $("#lnkToLogsFromTest");
  if(lnkLogsTest) lnkLogsTest.addEventListener("click", (e)=>{ e.preventDefault(); switchToTab("tab-logs"); });
  const lnkLogsTrain = $("#lnkToLogsFromTrain");
  if(lnkLogsTrain) lnkLogsTrain.addEventListener("click", (e)=>{ e.preventDefault(); switchToTab("tab-logs"); });

  const btnLogsConnect = $("#btnLogsConnect");
  if(btnLogsConnect) btnLogsConnect.addEventListener("click", connectLogs);
  const btnLogsClear = $("#btnLogsClear");
  if(btnLogsClear) btnLogsClear.addEventListener("click", ()=>{ const ls=$("#logStream"); if(ls) ls.textContent=""; });

  const btnRefreshSessions = $("#btnRefreshSessions");
  if(btnRefreshSessions) btnRefreshSessions.addEventListener("click", async ()=>{
    await loadSessions(true);
    await showResultsForSelected();
  });
  const selResults = $("#selResults");
  if(selResults) selResults.addEventListener("change", showResultsForSelected);

  const btnRefreshMapSessions = $("#btnRefreshMapSessions");
  if(btnRefreshMapSessions) btnRefreshMapSessions.addEventListener("click", async ()=>{
    await loadSessions(true);
    await refreshMapSessionSelected();
  });
  const selMapSession = $("#selMapSession");
  if(selMapSession) selMapSession.addEventListener("change", refreshMapSessionSelected);

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
    const elSize = $("#yoloSizeOption");
    if(elOpts) elOpts.style.display = show ? 'block' : 'none';
    if(elSize) elSize.style.display = show ? 'block' : 'none';
  }
  if(selBackendGlobal) selBackendGlobal.addEventListener('change', _updateYoloUIForTrain);
  if(selBackendTrain) selBackendTrain.addEventListener('change', ()=>{ _updateYoloUIForTrain(); /* no-op for models list on train */ });
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

  const selOptimize = document.getElementById('selOptimizeDataset');
  if(selOptimize){ selOptimize.addEventListener('change', onOptimizeDatasetChange); }
  document.getElementById('btnSyncOptimizeDataset')?.addEventListener('click', (e)=>{ e.preventDefault(); syncOptimizeDatasetFromTest(); });
  document.getElementById('btnRefreshColmap')?.addEventListener('click', ()=> refreshColmapState());
  document.getElementById('btnStartColmap')?.addEventListener('click', startColmapJob);
  document.getElementById('btnFinishColmap')?.addEventListener('click', finishColmapJob);

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

  updateAccurateUI();
  updateOptimizePanel(getColmapState(document.getElementById('selOptimizeDataset')?.value) || null);
  renderLegend();
}

document.addEventListener("DOMContentLoaded", async ()=>{
  setupUI();
  initMap();
  if (typeof initMapOverlayUI === 'function') {
    initMapOverlayUI();
  }
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
