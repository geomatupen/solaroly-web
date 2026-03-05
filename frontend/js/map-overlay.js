// ============================================================================
// MAP OVERLAY FUNCTIONALITY - GeoJSON and TIF file uploads
// ============================================================================

// Overlay registry for user-uploaded files
let mapOverlayLayers = {};

/**
 * Reusable TIF tiling function
 * Creates tile layers from a GeoTIFF tile definition
 */
function createOverlayTileLayers(tileDefs, overlayName) {
  const group = L.layerGroup();
  const layers = [];
  let bounds = null;

  if (!Array.isArray(tileDefs) || tileDefs.length === 0) {
    console.warn('No valid tile definitions provided');
    return null;
  }

  tileDefs.forEach((tileDef, i) => {
    try {
      // Validate bounds before creating tile layer
      let layerBounds = null;
      if (tileDef.bounds && Array.isArray(tileDef.bounds) && tileDef.bounds.length === 2) {
        const sw = L.latLng(tileDef.bounds[0][0], tileDef.bounds[0][1]);
        const ne = L.latLng(tileDef.bounds[1][0], tileDef.bounds[1][1]);
        layerBounds = L.latLngBounds(sw, ne);
        
        if (!layerBounds.isValid()) {
          console.warn(`Invalid bounds for tile layer ${i}:`, tileDef.bounds);
          return;
        }
        
        bounds = bounds ? bounds.extend(layerBounds) : layerBounds;
      }
      
      const layer = L.tileLayer(tileDef.url, {
        bounds: layerBounds,
        minZoom: tileDef.minzoom ?? 0,
        maxZoom: tileDef.maxzoom ?? 22,
        tileSize: 256,
        noWrap: true,
        attribution: tileDef.attribution || '© Overlay',
        pane: 'overlayTifPane'
      });
      group.addLayer(layer);
      layers.push(layer);
    } catch(e) {
      console.error(`Failed to create tile layer ${i}:`, e);
    }
  });

  // Do NOT add to map by default - user will toggle visibility
  // group.addTo(MAP);
  
  // Register in overlay registry
  overlayRegistry[overlayName] = { 
    layer: group, 
    type: 'tif_overlay',
    bounds: bounds
  };
  
    // Auto-enable the overlay on upload
    layer.addTo(MAP);
    
  return { group, layers, bounds };
}

/**
 * Open the "Add File" modal
 */
function openAddMapFileModal() {
  const modal = document.getElementById('addMapFileModal');
  if (!modal) return;
  
  modal.classList.remove('hidden');
  
  // Reset forms and radio buttons
  document.getElementById('rdoGeoJSON').checked = false;
  document.getElementById('rdoTIF').checked = false;
  document.getElementById('geoJsonForm').style.display = 'none';
  document.getElementById('tifForm').style.display = 'none';
  document.getElementById('addFileForm').style.display = 'none';
  document.getElementById('btnUploadGeoJson').style.display = 'none';
  document.getElementById('btnUploadTIF').style.display = 'none';
  
  // Clear input values
  document.getElementById('geoJsonName').value = 'overlay_geojson';
  document.getElementById('geoJsonFile').value = '';
  document.getElementById('geoJsonTemporary').checked = false;
  document.getElementById('tifName').value = 'overlay_ortho';
  document.getElementById('tifFile').value = '';
}

function showGeoJsonForm() {
  document.getElementById('geoJsonForm').style.display = 'block';
  document.getElementById('tifForm').style.display = 'none';
  document.getElementById('addFileForm').style.display = 'block';
  document.getElementById('btnUploadGeoJson').style.display = 'block';
  document.getElementById('btnUploadTIF').style.display = 'none';
}

function showTifForm() {
  document.getElementById('geoJsonForm').style.display = 'none';
  document.getElementById('tifForm').style.display = 'block';
  document.getElementById('addFileForm').style.display = 'block';
  document.getElementById('btnUploadGeoJson').style.display = 'none';
  document.getElementById('btnUploadTIF').style.display = 'block';
}

/**
 * Handle GeoJSON upload
 */
async function uploadGeoJsonOverlay() {
  const nameInput = document.getElementById('geoJsonName');
  const fileInput = document.getElementById('geoJsonFile');
  const tempCheckbox = document.getElementById('geoJsonTemporary');
  
  const name = (nameInput.value || 'overlay_geojson').trim();
  const file = fileInput.files[0];
  const isTemporary = tempCheckbox?.checked || false;
  
  if (!file) {
    alert('Please select a GeoJSON file');
    return;
  }
  
  if (!name) {
    alert('Please enter a file name');
    return;
  }
  
  // Check if name already exists
  if (overlayRegistry[name]) {
    const changeName = confirm(`An overlay named "${name}" already exists. Do you want to replace it?`);
    if (!changeName) {
      alert('Please change the name before uploading.');
      return;
    }
  }
  
  try {
    const text = await file.text();
    const gj = JSON.parse(text);
    
    if (!gj.type || gj.type !== 'FeatureCollection') {
      alert('Invalid GeoJSON format. Must be a FeatureCollection.');
      return;
    }
    
    // Create layer from GeoJSON
    const layer = L.geoJSON(gj, {
      style: {
        color: '#ff7800',
        weight: 2,
        opacity: 0.7,
        fillColor: '#ff7800',
        fillOpacity: 0.3
      },
      pointToLayer: (feature, latlng) => {
        return L.circleMarker(latlng, {
          radius: 5,
          color: '#ff7800',
          weight: 2,
          opacity: 0.7,
          fillColor: '#ff7800',
          fillOpacity: 0.3
        });
      },
      onEachFeature: (feature, layer) => {
        if (feature.properties) {
          let popupContent = '<div class="mini">';
          for (const [key, val] of Object.entries(feature.properties)) {
            popupContent += `<b>${escapeHtml(key)}:</b> ${escapeHtml(String(val))}<br>`;
          }
          popupContent += '</div>';
          layer.bindPopup(popupContent);
        }
      }
    });
    // Do NOT add to map by default - user will toggle visibility
    // layer.addTo(MAP);
    
    // Save to backend if not temporary
    let overlay_id = null;
    if (!isTemporary) {
      try {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('name', name);
        
        const saveResp = await fetch('/api/upload_geojson_overlay', {
          method: 'POST',
          body: formData
        });
        
        const saveData = await saveResp.json();
        if (saveData.ok) {
          overlay_id = saveData.overlay_id;
          console.log(`✓ GeoJSON saved to backend: ${overlay_id}`);
        }
      } catch (err) {
        console.error('Failed to save GeoJSON to backend:', err);
        alert('Warning: Failed to save GeoJSON to backend. It will be temporary.');
      }
    }
    
    // Register in overlay registry
    const layerBounds = layer.getBounds ? layer.getBounds() : null;
    overlayRegistry[name] = { 
      layer, 
      type: 'geojson_overlay',
      style: { opacity: 0.7, color: '#ff7800' },
      data: gj,
      temporary: isTemporary,
      overlay_id: overlay_id,
      bounds: layerBounds
    };
    
    // Auto-enable the overlay on upload
    layer.addTo(MAP);
    
    refreshLayersPanel();
    
    // Fit to layer bounds if possible
    try {
      const bounds = computeLayerBounds(layer);
      if (bounds && bounds.isValid && bounds.isValid()) {
        MAP.fitBounds(bounds, { padding: [20, 20] });
      }
    } catch(e) {}
    
    // Close modal and reset
    document.getElementById('addMapFileModal').classList.add('hidden');
    fileInput.value = '';
    nameInput.value = 'overlay_geojson';
    tempCheckbox.checked = false;
    
    console.log(`✓ GeoJSON overlay "${name}" added (${isTemporary ? 'temporary' : 'persistent'})`);
  } catch (e) {
    console.error('Failed to load GeoJSON:', e);
    alert('Failed to load GeoJSON file: ' + e.message);
  }
}

/**
 * Handle TIF upload
 */
async function uploadTifOverlay() {
  const nameInput = document.getElementById('tifName');
  const fileInput = document.getElementById('tifFile');
  const progressWrap = document.getElementById('tifUploadProgress');
  const progressBar = document.getElementById('tifUploadBar');
  const progressText = document.getElementById('tifUploadText');
  
  const name = (nameInput.value || 'overlay_ortho').trim();
  const file = fileInput.files[0];
  
  if (!file) {
    alert('Please select a TIF file');
    return;
  }
  
  if (!name) {
    alert('Please enter a file name');
    return;
  }
  
  // Check if name already exists
  if (overlayRegistry[name]) {
    const changeName = confirm(`An overlay named "${name}" already exists. Do you want to replace it?`);
    if (!changeName) {
      alert('Please change the name before uploading.');
      return;
    }
  }
  
  try {
    progressWrap.style.display = 'block';
    progressBar.style.width = '0%';
    progressText.textContent = 'Uploading… 0%';
    
    const formData = new FormData();
    formData.append('file', file);
    formData.append('name', name);
    
    const xhr = new XMLHttpRequest();
    
    xhr.upload.addEventListener('progress', (e) => {
      if (e.lengthComputable) {
        const pct = Math.round((e.loaded / e.total) * 100);
        progressBar.style.width = pct + '%';
        progressText.textContent = `Uploading… ${pct}%`;
      }
    });
    
    xhr.addEventListener('load', async () => {
      if (xhr.status === 200) {
        try {
          const result = JSON.parse(xhr.responseText);
          
          if (result.ok && result.tiles && Array.isArray(result.tiles.layers)) {
            progressText.textContent = 'Processing tiles…';
            
            // Create tile layers using reusable function
            const tileData = result.tiles.layers.map(layer => ({
              url: layer.template || layer.url,
              minzoom: layer.minzoom ?? 0,
              maxzoom: layer.maxzoom ?? 22,
              bounds: layer.bounds,
              attribution: layer.name || name
            }));
            
            const overlay = createOverlayTileLayers(tileData, name);
            
            // Store overlay_id in registry for backend deletion
            if (overlay && overlayRegistry[name]) {
              overlayRegistry[name].overlay_id = result.overlay_id;
              overlayRegistry[name].temporary = false;
            }
            
            // Auto-enable the overlay on upload
            if (overlay && overlay.group) {
              overlay.group.addTo(MAP);
            }
            
            if (overlay && overlay.bounds) {
              MAP.fitBounds(overlay.bounds, { padding: [20, 20] });
            }
            
            // Close modal and reset
            document.getElementById('addMapFileModal').classList.add('hidden');
            fileInput.value = '';
            nameInput.value = 'overlay_ortho';
            progressWrap.style.display = 'none';
            
            console.log(`✓ TIF overlay "${name}" uploaded and tiled (ID: ${result.overlay_id})`);
          } else {
            throw new Error(result.error || 'Failed to generate tiles');
          }
        } catch (e) {
          throw new Error('Failed to parse response: ' + e.message);
        }
      } else {
        throw new Error(`Server error: ${xhr.status} ${xhr.statusText}`);
      }
    });
    
    xhr.addEventListener('error', () => {
      throw new Error('Network error during upload');
    });
    
    xhr.open('POST', '/api/upload_tif_overlay');
    xhr.send(formData);
    
  } catch (e) {
    console.error('TIF upload failed:', e);
    alert('Failed to upload TIF: ' + e.message);
    progressWrap.style.display = 'none';
  }
}

/**
 * Load saved overlays from backend
 */
async function loadSavedOverlays() {
  try {
    const resp = await fetch('/api/list_overlays');
    const data = await resp.json();
    
    if (!data.ok || !data.overlays) return;
    
    for (const overlay of data.overlays) {
      if (overlay.type === 'geojson') {
        // Load GeoJSON overlay
        const gjResp = await fetch(overlay.path);
        const gjData = await gjResp.json();
        
        const bounds = computeLayerBounds(gjData);
        const layer = L.geoJSON(gjData, {
          style: { color: '#2b87ff', weight: 2, fillOpacity: 0.2 }
        });
        // Do NOT add to map by default
        
        overlayRegistry[overlay.name] = {
          layer,
          type: 'geojson',
          data: gjData,
          bounds,
          temporary: false,
          overlay_id: overlay.overlay_id
        };
      } else if (overlay.type === 'tif') {
        // Load GeoTIFF overlay - need to get tile info
        const tileResp = await fetch(`/api/get_overlay_tiles?overlay_id=${overlay.overlay_id}`);
        const tileData = await tileResp.json();
        
        if (tileData.ok && tileData.tiles && tileData.tiles.layers) {
          const group = L.layerGroup();
          tileData.tiles.layers.forEach(layerDef => {
            // Validate bounds before creating tile layer
            if (!layerDef.bounds || !Array.isArray(layerDef.bounds) || layerDef.bounds.length !== 2) {
              console.warn('Invalid bounds for tile layer:', layerDef);
              return;
            }
            
            const bounds = L.latLngBounds(layerDef.bounds);
            if (!bounds.isValid()) {
              console.warn('Invalid bounds detected:', layerDef.bounds);
              return;
            }
            
            const tileLayer = L.tileLayer(layerDef.template, {
              bounds: bounds,
              minZoom: layerDef.minzoom || 0,
              maxZoom: layerDef.maxzoom || 22,
              tileSize: 256,
              noWrap: true,
              pane: 'overlayTifPane'
            });
            group.addLayer(tileLayer);
          });
          // Do NOT add to map by default - user will toggle visibility
          // group.addTo(MAP);
          
          // Convert raw bounds array to LatLngBounds object
          let storedBounds = null;
          if (tileData.tiles.bounds && Array.isArray(tileData.tiles.bounds) && tileData.tiles.bounds.length === 2) {
            try {
              storedBounds = L.latLngBounds(tileData.tiles.bounds);
            } catch(e) {
              console.warn('Failed to create bounds from tile data:', e);
            }
          }
          
          overlayRegistry[overlay.name] = {
            layer: group,
            type: 'raster',
            overlay_id: overlay.overlay_id,
            bounds: storedBounds
          };
        }
      }
    }
    
    refreshLayersPanel();
  } catch (err) {
    console.error('Failed to load saved overlays:', err);
  }
}

/**
 * Initialize map overlay UI
 */
function initMapOverlayUI() {
  const btnAdd = document.getElementById('btnAddMapFile');
  if (btnAdd) {
    btnAdd.addEventListener('click', openAddMapFileModal);
  }
  
  // Auto-populate names from selected files
  const geoJsonFile = document.getElementById('geoJsonFile');
  const geoJsonName = document.getElementById('geoJsonName');
  if (geoJsonFile && geoJsonName) {
    geoJsonFile.addEventListener('change', (e) => {
      if (e.target.files[0]) {
        const fileName = e.target.files[0].name.replace(/\.[^.]+$/, ''); // Remove extension
        geoJsonName.value = fileName;
      }
    });
  }
  
  const tifFile = document.getElementById('tifFile');
  const tifName = document.getElementById('tifName');
  if (tifFile && tifName) {
    tifFile.addEventListener('change', (e) => {
      if (e.target.files[0]) {
        const fileName = e.target.files[0].name.replace(/\.[^.]+$/, ''); // Remove extension
        tifName.value = fileName;
      }
    });
  }
  
  // Load saved overlays on init
  loadSavedOverlays();
  
  // Radio button listeners
  const rdoGeoJSON = document.getElementById('rdoGeoJSON');
  const rdoTIF = document.getElementById('rdoTIF');
  if (rdoGeoJSON) {
    rdoGeoJSON.addEventListener('change', (e) => {
      if (e.target.checked) showGeoJsonForm();
    });
  }
  if (rdoTIF) {
    rdoTIF.addEventListener('change', (e) => {
      if (e.target.checked) showTifForm();
    });
  }
  
  const btnUploadGeoJson = document.getElementById('btnUploadGeoJson');
  if (btnUploadGeoJson) {
    btnUploadGeoJson.addEventListener('click', uploadGeoJsonOverlay);
  }
  
  const btnUploadTIF = document.getElementById('btnUploadTIF');
  if (btnUploadTIF) {
    btnUploadTIF.addEventListener('click', uploadTifOverlay);
  }

  // Close button handlers
  const btnCloseModal = document.getElementById('btnCloseAddFileModal');
  if (btnCloseModal) {
    btnCloseModal.addEventListener('click', () => {
      document.getElementById('addMapFileModal').classList.add('hidden');
    });
  }

  const btnCancel = document.getElementById('btnCancelAddFile');
  if (btnCancel) {
    btnCancel.addEventListener('click', () => {
      document.getElementById('addMapFileModal').classList.add('hidden');
    });
  }

  // Close modal when clicking outside
  const modal = document.getElementById('addMapFileModal');
  if (modal) {
    modal.addEventListener('click', (e) => {
      if (e.target === modal) {
        modal.classList.add('hidden');
      }
    });
  }
}
