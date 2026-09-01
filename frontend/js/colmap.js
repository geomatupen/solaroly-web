/* COLMAP and optimize-tab logic */

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

// ---------- Accurate locations / COLMAP ----------

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
  const btn = document.getElementById('btnRunTest');
  const mosaicControls = document.getElementById('mosaicControls');
  const colmapEnabled = !!(window.featureFlags && window.featureFlags.colmap);
  const testIsRunning = typeof testAbort !== 'undefined' && testAbort !== null;

  if(!colmapEnabled){
    if(chk){ chk.checked = false; chk.disabled = true; }
    if(radColmap){ radColmap.checked = false; radColmap.disabled = true; }
    if(radOptical){ radOptical.checked = false; radOptical.disabled = true; }
    if(modeRow) setHidden(modeRow, true);
    if(optRow) setHidden(optRow, true);
    if(mosaicControls) mosaicControls.hidden = false;
    if(badge){ badge.className = 'pill pill-muted'; badge.textContent = 'Disabled'; }
    if(hint){ hint.textContent = 'Accurate poses disabled on this server.'; }
    if(btn){
      btn.disabled = testIsRunning;
      btn.classList.toggle('disabled', testIsRunning);
    }
    return;
  }else{
    if(chk) chk.disabled = false;
    if(radColmap) radColmap.disabled = false;
    if(radOptical) radOptical.disabled = false;
  }
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
    const disable = testIsRunning || (wantsAccurate && (
      !modeSelected ||
      (modeColmap && !ready) ||
      (modeOptical && !opticalProject)
    ));
    btn.disabled = disable;
    btn.classList.toggle('disabled', disable);
  }

  // Approximate EXIF/GPS mosaicing is independent of the optional COLMAP feature.
  if(mosaicControls){
    mosaicControls.hidden = false;
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
  if(!(window.featureFlags && window.featureFlags.colmap)){
    clearColmapPoll();
    return;
  }
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
  if(!(window.featureFlags && window.featureFlags.colmap)){
    return null;
  }
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
  if(!(window.featureFlags && window.featureFlags.colmap)){
    setColmapStatus('COLMAP is disabled on this server.', 'warn');
    return;
  }
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
  if(!(window.featureFlags && window.featureFlags.colmap)){
    setColmapStatus('COLMAP is disabled on this server.', 'warn');
    return;
  }
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
  const selectedDataset = Array.isArray(datasetsCache)
    ? datasetsCache.find(item => item && item.name === ds)
    : null;
  const isOrthophoto = selectedDataset?.input_type === 'tif';
  window.TestImageAlignment?.syncAvailability?.(selectedDataset?.input_type || null);
  const lensCheckbox = document.getElementById('chkUndistortThermal');
  const lensHint = document.getElementById('lensCorrectionHint');
  if(lensCheckbox){
    lensCheckbox.disabled = isOrthophoto;
    lensCheckbox.title = isOrthophoto
      ? 'Lens correction is skipped for orthophotos.'
      : '';
  }
  if(lensHint){
    lensHint.textContent = isOrthophoto
      ? 'Skipped for orthophotos because they are already geometrically corrected products.'
      : 'Optional runtime calibration from repeated straight structures; stops safely if correction cannot be proven.';
  }
  const mosaicCheckbox = document.getElementById('chkMosaicImages');
  const mosaicHint = document.getElementById('mosaicHint');
  if(mosaicCheckbox){
    mosaicCheckbox.disabled = isOrthophoto;
    mosaicCheckbox.title = isOrthophoto
      ? 'Approximate mosaicing is unavailable for orthophoto input.'
      : '';
    if(isOrthophoto && mosaicCheckbox.checked){
      mosaicCheckbox.checked = false;
      mosaicCheckbox.dispatchEvent(new Event('change'));
    }
  }
  if(mosaicHint){
    mosaicHint.textContent = isOrthophoto
      ? 'Unavailable for orthophotos because they are already mosaicked geospatial products.'
      : 'Available for folders of geotagged individual images.';
  }
  const sel = document.getElementById('selOptimizeDataset');
  if(ds && sel){
    sel.value = ds;
  }
  if(!(window.featureFlags && window.featureFlags.colmap)){
    updateAccurateUI();
    return;
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
  if(!(window.featureFlags && window.featureFlags.colmap)){
    updateOptimizePanel(null);
    return;
  }
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

function initOptimizeTabControls(){
  const chkAccurate = document.getElementById('chkAccurateLocations');
  if(chkAccurate){ chkAccurate.addEventListener('change', updateAccurateUI); }
  document.getElementById('radAccurateColmap')?.addEventListener('change', updateAccurateUI);
  document.getElementById('radAccurateOptical')?.addEventListener('change', updateAccurateUI);
  document.getElementById('selUseOptimizationFrom')?.addEventListener('change', updateAccurateUI);

  const btnGoOptimize = document.getElementById('btnGoOptimize');
  if(btnGoOptimize){
    btnGoOptimize.addEventListener('click', (e)=>{
      e.preventDefault();
      if(!(window.featureFlags && window.featureFlags.colmap)){
        warn('test', 'COLMAP optimization is disabled on this server.');
        return;
      }
      syncOptimizeDatasetFromTest();
      switchToTab('tab-optimize');
      refreshColmapState();
    });
  }

  const colmapInfoModal = document.getElementById('colmapInfoModal');
  const openColmapInfo = ()=> {
    if(colmapInfoModal){
      colmapInfoModal.classList.add('show');
      colmapInfoModal.classList.remove('hidden');
    }
  };
  const closeColmapInfo = ()=> {
    if(colmapInfoModal){
      colmapInfoModal.classList.remove('show');
      colmapInfoModal.classList.add('hidden');
    }
  };
  document.getElementById('btnColmapInfo')?.addEventListener('click', openColmapInfo);
  document.getElementById('btnCloseColmapInfo')?.addEventListener('click', closeColmapInfo);
  document.getElementById('btnColmapInfoDone')?.addEventListener('click', closeColmapInfo);

  const openColmapConfig = ()=> {
    const modal = document.getElementById('colmapConfigModal');
    if(modal){
      modal.classList.add('show');
      modal.classList.remove('hidden');
    }
  };
  const closeColmapConfig = ()=> {
    const modal = document.getElementById('colmapConfigModal');
    if(modal){
      modal.classList.remove('show');
      modal.classList.add('hidden');
    }
  };
  document.getElementById('btnColmapConfig')?.addEventListener('click', openColmapConfig);
  document.getElementById('btnCloseColmapConfig')?.addEventListener('click', closeColmapConfig);
  document.getElementById('btnColmapConfigDone')?.addEventListener('click', closeColmapConfig);

  const selOptimize = document.getElementById('selOptimizeDataset');
  if(selOptimize){ selOptimize.addEventListener('change', onOptimizeDatasetChange); }
  document.getElementById('btnSyncOptimizeDataset')?.addEventListener('click', (e)=>{ e.preventDefault(); syncOptimizeDatasetFromTest(); });
  document.getElementById('btnRefreshColmap')?.addEventListener('click', ()=> refreshColmapState());
  document.getElementById('btnStartColmap')?.addEventListener('click', startColmapJob);
  document.getElementById('btnFinishColmap')?.addEventListener('click', finishColmapJob);

  updateAccurateUI();
  const dataset = document.getElementById('selOptimizeDataset')?.value;
  updateOptimizePanel(getColmapState(dataset) || null);
}
