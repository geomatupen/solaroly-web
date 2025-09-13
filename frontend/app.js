(function () {
  // ----------------- helpers -----------------
  const API_BASE = location.origin; // served by same uvicorn
  const $ = (id) => document.getElementById(id);
  const setStatus = (el, text, cls) => { el.textContent = text; el.className = "status" + (cls ? " " + cls : ""); };
  const show = (el) => el.removeAttribute("hidden");
  const hide = (el) => el.setAttribute("hidden", "hidden");
  const on = (el, ev, fn) => el.addEventListener(ev, fn);

  const logStream = $("logStream");
  const appendLog = (line) => {
    const atBottom = (logStream.scrollTop + logStream.clientHeight) >= (logStream.scrollHeight - 5);
    logStream.textContent += (logStream.textContent ? "\n" : "") + line;
    if (atBottom) logStream.scrollTop = logStream.scrollHeight;
  };

  async function postForm(url, fd) {
    const res = await fetch(url, { method: "POST", body: fd });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json().catch(() => ({}));
  }

  // ----------------- tabs -----------------
  document.querySelectorAll(".tab").forEach(btn => {
    on(btn, "click", () => {
      document.querySelectorAll(".tab").forEach(x => x.classList.remove("active"));
      document.querySelectorAll(".tabPanel").forEach(x => x.classList.remove("active"));
      btn.classList.add("active");
      const id = "tab-" + btn.dataset.tab;
      $(id).classList.add("active");
    });
  });

  // ----------------- SSE logs -----------------
  const logConn = $("logConn");

  // Use your existing alert boxes
  const trainWarn = $("trainWarn"), trainErr = $("trainErr"), trainOk = $("trainOk");
  const testWarn  = $("testWarn"),  testErr  = $("testErr"),  testOk  = $("testOk");

  function handleUiMessage(msg) {
    // Map INFO → OK banners so decode messages show as green info
    if (msg.startsWith("UI:WARN:train:")) {
      trainWarn.textContent = msg.replace("UI:WARN:train:", "").trim(); show(trainWarn);
    } else if (msg.startsWith("UI:ERR:train:")) {
      trainErr.textContent  = msg.replace("UI:ERR:train:", "").trim();  show(trainErr);
    } else if (msg.startsWith("UI:OK:train:") || msg.startsWith("UI:INFO:train:")) {
      trainOk.textContent   = msg.split("train:")[1].trim();            show(trainOk);
    } else if (msg.startsWith("UI:WARN:test:")) {
      testWarn.textContent  = msg.replace("UI:WARN:test:", "").trim();  show(testWarn);
    } else if (msg.startsWith("UI:ERR:test:")) {
      testErr.textContent   = msg.replace("UI:ERR:test:", "").trim();   show(testErr);
    } else if (msg.startsWith("UI:OK:test:") || msg.startsWith("UI:INFO:test:")) {
      testOk.textContent    = msg.split("test:")[1].trim();             show(testOk);
    }
  }

  const connectLogs = () => {
    const es = new EventSource(`${API_BASE}/api/logs`);
    es.onopen = () => { setStatus(logConn, "Connected", "ok"); appendLog("[logs] connected"); };
    es.onmessage = (e) => { if (e && e.data) { handleUiMessage(e.data); appendLog(e.data); } };
    es.onerror = () => { setStatus(logConn, "Reconnecting…", "warn"); try { es.close(); } catch(_) {} setTimeout(connectLogs, 1500); };
  };
  connectLogs();

  // ----------------- health -----------------
  const btnHealth = $("btnHealth"), healthStatus = $("healthStatus");
  on(btnHealth, "click", async () => {
    try { const r = await fetch(`${API_BASE}/api/health`); setStatus(healthStatus, r.ok ? "OK" : `HTTP ${r.status}`, r.ok ? "ok" : "warn"); }
    catch (e) { setStatus(healthStatus, "Error", "err"); appendLog(`[health] ${e.message || e}`); }
  });

  // ----------------- Train -----------------
  const spinTrain = $("spinTrain"), btnTrain = $("btnTrain"), chkUseThermalTrain = $("chkUseThermalTrain");
  const inpMaxIter = $("inpMaxIter"), inpBaseLR = $("inpBaseLR"), inpBatch = $("inpBatch");

  on(btnTrain, "click", async () => {
    // Clear banners for a fresh run
    hide(trainWarn); hide(trainErr); hide(trainOk);

    spinTrain.classList.remove("hidden");
    btnTrain.disabled = true;
    try {
      const fd = new FormData();
      fd.append("use_thermal", chkUseThermalTrain.checked ? "true" : "false");
      fd.append("max_iter", inpMaxIter.value || "9000");
      fd.append("base_lr",  inpBaseLR.value  || "0.002");
      fd.append("ims_per_batch", inpBatch.value || "4");
      const j = await postForm(`${API_BASE}/api/train`, fd);
      trainOk.textContent = "Training started. Watch logs for progress."; show(trainOk);
      appendLog(`[train] ${JSON.stringify(j)}`);
    } catch (e) {
      trainErr.textContent = e.message || String(e); show(trainErr);
      appendLog(`[train] ERROR: ${e.message || e}`);
    } finally {
      spinTrain.classList.add("hidden");
      btnTrain.disabled = false;
    }
  });

  // ----------------- Test (uploads) -----------------
  const spinTest = $("spinTest"), btnTest = $("btnTest"), chkUseThermalTest = $("chkUseThermalTest"), filesTest = $("filesTest");

  on(btnTest, "click", async () => {
    // Clear banners for a fresh run
    hide(testWarn); hide(testErr); hide(testOk);

    if (!filesTest.files || filesTest.files.length === 0) {
      testWarn.textContent = "Please choose at least one image."; show(testWarn);
      return;
    }
    spinTest.classList.remove("hidden"); btnTest.disabled = true;
    try {
      const fd = new FormData();
      fd.append("use_thermal", chkUseThermalTest.checked ? "true" : "false");
      for (const f of filesTest.files) fd.append("files", f, f.name);
      const j = await postForm(`${API_BASE}/api/test_uploads`, fd);
      testOk.textContent = "Test complete. Layers are added to the map."; show(testOk);
      appendLog(`[test] ${JSON.stringify(j)}`);

      // Load returned GeoJSON(s) into the map
      if (j && j.geojson_url) {
        await addGeoJsonLayerFromUrl(j.geojson_url, `Anomalies ${new Date().toLocaleTimeString()}`);
      }
      if (Array.isArray(j.image_points)) {
        j.image_points.forEach(pt => {
          addPointMarker(pt.lat, pt.lon, pt.name, pt.url);
        });
      }
      switchToTab("layers");
    } catch (e) {
      testErr.textContent = e.message || String(e); show(testErr);
      appendLog(`[test] ERROR: ${e.message || e}`);
    } finally {
      spinTest.classList.add("hidden"); btnTest.disabled = false;
    }
  });

  // ----------------- Map / Layers -----------------
  let map = L.map("map", { zoomControl: true }).setView([20, 0], 2);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: "&copy; OpenStreetMap",
    maxZoom: 19
  }).addTo(map);

  const layerList = $("layerList");
  const layers = []; // {id, layer, name, visible}

  function addLayerEntry(layer, name) {
    const id = Math.random().toString(36).slice(2);
    const item = document.createElement("div");
    item.className = "layerItem";
    const left = document.createElement("div");
    left.textContent = name;
    const right = document.createElement("div");
    const chk = document.createElement("input");
    chk.type = "checkbox"; chk.checked = true;
    const del = document.createElement("button"); del.className = "btn secondary"; del.textContent = "Remove";
    right.append(chk, del);
    item.append(left, right);
    layerList.prepend(item);

    chk.addEventListener("change", () => {
      if (chk.checked) { map.addLayer(layer); } else { map.removeLayer(layer); }
    });
    del.addEventListener("click", () => {
      map.removeLayer(layer);
      item.remove();
    });

    layers.push({ id, layer, name, visible: true });
  }

  async function addGeoJsonLayerFromUrl(url, name = "GeoJSON") {
    const res = await fetch(url);
    const gj = await res.json();
    const layer = L.geoJSON(gj, {
      style: f => ({ color: "#4cc38a", weight: 2, fillOpacity: 0.15 }),
      pointToLayer: (f, latlng) => L.circleMarker(latlng, { radius: 5, color: "#ffd166" }),
      onEachFeature: (f, l) => {
        const props = Object.entries(f.properties || {}).map(([k,v]) => `<div><b>${k}</b>: ${String(v)}</div>`).join("");
        l.bindPopup(`<div><b>${f.geometry?.type}</b>${props ? `<hr>${props}` : ""}</div>`);
      }
    }).addTo(map);
    addLayerEntry(layer, name);
    try { map.fitBounds(layer.getBounds(), { maxZoom: 15, padding: [20,20] }); } catch(_) {}
  }

  function addPointMarker(lat, lon, name, imgUrl) {
    if (typeof lat !== "number" || typeof lon !== "number") return;
    const m = L.marker([lat, lon]).addTo(map);
    let html = `<b>${name || "Image"}</b>`;
    if (imgUrl) html += `<div style="margin-top:6px"><img src="${imgUrl}" alt="" style="max-width:220px;border-radius:8px;border:1px solid #333"/></div>`;
    m.bindPopup(html);
    addLayerEntry(m, name || "Image");
  }

  // upload custom GeoJSON overlay
  const fileGeoJson = $("fileGeoJson");
  fileGeoJson.addEventListener("change", async () => {
    const f = fileGeoJson.files?.[0];
    if (!f) return;
    try {
      const text = await f.text();
      const gj = JSON.parse(text);
      const blobUrl = URL.createObjectURL(new Blob([JSON.stringify(gj)], {type:"application/geo+json"}));
      await addGeoJsonLayerFromUrl(blobUrl, f.name);
    } catch (e) {
      appendLog(`[layers] Failed to add GeoJSON: ${e.message || e}`);
    } finally {
      fileGeoJson.value = "";
    }
  });

  function switchToTab(name){
    document.querySelectorAll(".tab").forEach(x => x.classList.remove("active"));
    document.querySelectorAll(".tabPanel").forEach(x => x.classList.remove("active"));
    document.querySelector(`.tab[data-tab="${name}"]`).classList.add("active");
    $(`tab-${name}`).classList.add("active");
  }

  // logs clear
  on($("btnClearLogs"), "click", () => { logStream.textContent = ""; });

  // drag & drop for test
  const fileDrop = document.querySelector('#tab-test .fileDrop');
  fileDrop.addEventListener('dragover', e => { e.preventDefault(); fileDrop.classList.add('hover'); });
  fileDrop.addEventListener('dragleave', () => fileDrop.classList.remove('hover'));
  fileDrop.addEventListener('drop', e => { e.preventDefault(); fileDrop.classList.remove('hover'); filesTest.files = e.dataTransfer.files; });
})();
