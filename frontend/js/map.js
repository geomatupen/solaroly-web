/* Map rendering, overlays, and layers panel logic */

let MAP, baseLayers, overlayRegistry = {};
let imagesLayerGroup = null;           // holds all image markers/overlays
let imageMarkers = new Map();          // id -> L.Marker or L.ImageOverlay
let geojsonLayer = null;
let imageMarkersLayer = null;
let tileLayers = [];
let currentSession = null;
let styleTarget = null;
let layerMenuState = { name: null, info: null };
let mapDetectionFilterActive = false;
let prevMapDetectionStates = null;
let activeGeoJsonHoverReset = null;

function addGeoJsonHoverHighlight(feature, featureLayer) {
  const geometryType = feature?.geometry?.type;
  if (!['Polygon', 'MultiPolygon'].includes(geometryType) || !featureLayer?.setStyle) return;
  let previousStyle = null;
  const resetHighlight = () => {
    if (previousStyle) featureLayer.setStyle(previousStyle);
    previousStyle = null;
    if (activeGeoJsonHoverReset === resetHighlight) activeGeoJsonHoverReset = null;
  };
  featureLayer.on('mouseover', event => {
    if (previousStyle) return;
    if (activeGeoJsonHoverReset && activeGeoJsonHoverReset !== resetHighlight) {
      activeGeoJsonHoverReset();
    }
    const options = event.target.options || {};
    previousStyle = {
      color: options.color,
      weight: options.weight,
      opacity: options.opacity,
      fillColor: options.fillColor,
      fillOpacity: options.fillOpacity,
    };
    event.target.setStyle({
      color: '#ffffff',
      weight: Number(options.weight || 1) + 2,
      opacity: 1,
      fillColor: options.fillColor || options.color,
      fillOpacity: Math.max(Number(options.fillOpacity || 0), 0.4),
    });
    activeGeoJsonHoverReset = resetHighlight;
  });
  featureLayer.on('mouseout', resetHighlight);
  featureLayer.on('click', resetHighlight);
  featureLayer.on('remove', resetHighlight);
}

window.addGeoJsonHoverHighlight = addGeoJsonHoverHighlight;

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
    const located = Array.isArray(lastLoadedImagesGJ?.features)
      ? lastLoadedImagesGJ.features.filter(feature => feature?.geometry?.type === 'Point').length
      : 0;
    ul.innerHTML = located
      ? `<li class="dim">${located} image location${located === 1 ? '' : 's'} loaded. No individual image overlays are available.</li>`
      : `<li class="dim">No geolocated images in this session</li>`;
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
        <span>${escapeHtml(rec.name)}${rec.reviewOnly ? ' <small class="muted">· mosaic source</small>' : ''}${rec.n ? ` <span style="color:#0a84ff; font-weight:bold;">(${rec.n})</span>` : ''}</span>
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
    prevMapDetectionStates = new Map();
    for (const rec of imageCatalog){
      prevMapDetectionStates.set(rec.id, !!rec.on);
      const hasDetections = !!(rec.n && rec.n > 0);
      if (hasDetections && !rec.on){
        toggleImageOverlay(rec.id, true);
      } else if (!hasDetections && rec.on){
        toggleImageOverlay(rec.id, false);
      }
    }
  } else if (prevMapDetectionStates){
    for (const rec of imageCatalog){
      const desired = prevMapDetectionStates.has(rec.id) ? prevMapDetectionStates.get(rec.id) : true;
      if (rec.on !== desired){
        toggleImageOverlay(rec.id, desired);
      }
    }
    prevMapDetectionStates = null;
  }

  renderImagesList();
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
  const showLabels = Boolean(rec.data?.features?.length) && !isTifRaster;
  const labelsItem = showLabels
    ? '<li data-action="labels">Labels…</li><li class="layerLabelField" hidden><label>Label field<select><option value="">Off</option></select></label></li>'
    : '';
  
  menu.innerHTML = `
    <ul>
      <li data-action="zoom">Zoom to layer</li>
      ${styleItem}
      ${labelsItem}
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

  if (showLabels) {
    const fields = new Set();
    for (const feature of rec.data.features) {
      for (const [field, value] of Object.entries(feature?.properties || {})) {
        if (value != null && typeof value !== 'object') fields.add(field);
      }
      if (fields.size >= 40) break;
    }
    const select = menu.querySelector('.layerLabelField select');
    Array.from(fields).sort().forEach(field => {
      const option = document.createElement('option');
      option.value = field;
      option.textContent = field;
      select.appendChild(option);
    });
    select.value = rec.labelField || '';
    select.addEventListener('change', event => {
      event.stopPropagation();
      rec.labelField = select.value;
      rec.layer.eachLayer?.(layer => {
        try { layer.unbindTooltip?.(); } catch (_) {}
        const value = select.value ? layer.feature?.properties?.[select.value] : null;
        if (value == null || value === '') return;
        try {
          layer.bindTooltip(String(value), {
            className: 'mapFeatureLabel',
            direction: 'center',
            permanent: true,
            opacity: 0.94,
          });
        } catch (_) {}
      });
    });
  }

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

  if (action === 'labels') {
    const control = menu.querySelector('.layerLabelField');
    if (control) control.hidden = !control.hidden;
    return;
  }

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
 
