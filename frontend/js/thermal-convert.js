(() => {
  let activeJobId = null;
  let pollTimer = null;
  let lastScan = null;

  const byId = id => document.getElementById(id);
  const terminalStates = new Set(['completed', 'failed', 'cancelled']);
  const conversionUrl = () => window.api?.thermalConvert || '/api/thermal-convert';

  function showError(message = ''){
    const box = byId('thermalConvertError');
    if(!box) return;
    box.textContent = message;
    box.hidden = !message;
  }

  function setRunning(running){
    const start = byId('btnStartThermalConvert');
    const scan = byId('btnScanThermalFolder');
    const cancel = byId('btnCancelThermalConvert');
    const spinner = byId('spinThermalConvert');
    const currentInput = (byId('inpThermalInputDir')?.value || '').trim();
    const currentType = byId('selThermalConversionType')?.value || 'radiometric';
    const includeRadiometric = Boolean(byId('chkThermalIncludeRadiometric')?.checked);
    const canConvert = Boolean(
      lastScan && lastScan.inputValue === currentInput &&
      lastScan.conversionType === currentType &&
      lastScan.includeRadiometric === includeRadiometric && lastScan.supported > 0
    );
    if(start){
      start.hidden = !canConvert;
      start.disabled = running || !canConvert;
    }
    if(scan) scan.disabled = running;
    if(cancel) cancel.disabled = !running;
    if(spinner) spinner.hidden = !running;
    [
      'selThermalConversionType', 'inpThermalInputDir', 'inpThermalOutputDir', 'selThermalOutputFormat',
      'inpThermalJpegQuality', 'chkThermalOverwrite', 'chkThermalIncludeRadiometric'
    ].forEach(id => {
      const control = byId(id);
      if(control) control.disabled = running;
    });
  }

  function resetScan(){
    lastScan = null;
    const result = byId('thermalScanResult');
    if(result){
      result.hidden = true;
      result.textContent = '';
      result.className = 'thermalScanResult';
    }
    const progress = byId('thermalConvertProgress');
    const stats = byId('thermalConvertStats');
    const status = byId('thermalConvertStatus');
    const bar = byId('thermalConvertBar');
    if(progress) progress.hidden = true;
    if(stats) stats.hidden = true;
    if(status) status.textContent = '';
    if(bar) bar.style.width = '0%';
    ['thermalStatConverted', 'thermalStatSkipped', 'thermalStatFailed'].forEach(id => {
      const value = byId(id);
      if(value) value.textContent = '0';
    });
    setRunning(false);
  }

  function renderScan(scan){
    const result = byId('thermalScanResult');
    if(!result) return;
    const supported = Number(scan.supported || 0);
    const unsupported = Number(scan.unsupported || 0);
    const excludedRadiometric = Number(scan.excluded_radiometric || 0);
    const ignored = Number(scan.ignored_images || 0);
    const cameras = Array.isArray(scan.cameras) ? scan.cameras : [];
    const fileTypes = scan.file_types || {};
    const unsupportedSamples = Array.isArray(scan.unsupported_samples) ? scan.unsupported_samples : [];
    const isStandard = scan.conversion_type === 'standard';
    const lines = [];
    if(supported > 0){
      if(isStandard){
        lines.push(`${supported} readable JPG/PNG image${supported === 1 ? '' : 's'} found.`);
        const types = Object.entries(fileTypes).map(([name, count]) => `${name} (${count})`);
        if(types.length) lines.push(`Detected: ${types.join(', ')}.`);
      }else{
        lines.push(`${supported} supported radiometric JPEG${supported === 1 ? '' : 's'} found.`);
      }
      if(!isStandard && cameras.length){
        lines.push(`Detected: ${cameras.map(item => `${item.model} (${item.count})`).join(', ')}.`);
      }
      if(unsupported > 0){
        const label = isStandard ? 'unreadable JPG/PNG file' : 'unsupported JPG/JPEG file';
        lines.push(`${unsupported} ${label}${unsupported === 1 ? '' : 's'} will be skipped; only supported files will be converted.`);
        if(!isStandard){
          const models = [...new Set(unsupportedSamples.map(item => item.camera_model || 'Unknown camera'))];
          if(models.length) lines.push(`Unsupported detected: ${models.join(', ')}.`);
        }
      }
      if(isStandard && excludedRadiometric > 0){
        lines.push(`${excludedRadiometric} radiometric JPEG${excludedRadiometric === 1 ? '' : 's'} skipped; use Radiometric thermal JPEG mode for those files.`);
      }
      if(ignored > 0){
        const ignoredLabel = isStandard ? 'other-format image' : 'non-JPEG image';
        lines.push(`${ignored} ${ignoredLabel}${ignored === 1 ? '' : 's'} ignored.`);
      }
      result.classList.add(unsupported > 0 || excludedRadiometric > 0 ? 'warn' : 'ok');
    }else{
      lines.push(isStandard
        ? 'No eligible standard JPG, JPEG, or PNG images were found directly in this folder. Conversion cannot start.'
        : 'No supported radiometric JPG/JPEG images were found directly in this folder. Conversion cannot start.');
      if(unsupported > 0){
        const label = isStandard ? 'unreadable JPG/PNG file' : 'unsupported JPG/JPEG file';
        lines.push(`${unsupported} ${label}${unsupported === 1 ? '' : 's'} found.`);
        if(!isStandard){
          const models = [...new Set(unsupportedSamples.map(item => item.camera_model || 'Unknown camera'))];
          if(models.length) lines.push(`Unsupported detected: ${models.join(', ')}.`);
        }
      }
      if(isStandard && excludedRadiometric > 0){
        lines.push(`${excludedRadiometric} radiometric JPEG${excludedRadiometric === 1 ? '' : 's'} skipped; select Radiometric thermal JPEG mode to convert them.`);
      }
      if(ignored > 0){
        const ignoredLabel = isStandard ? 'other-format image' : 'non-JPEG image';
        lines.push(`${ignored} ${ignoredLabel}${ignored === 1 ? '' : 's'} ignored.`);
      }
      result.classList.add('err');
    }
    result.textContent = lines.join(' ');
    result.hidden = false;
  }

  async function scanInputFolder(){
    showError();
    const input = (byId('inpThermalInputDir')?.value || '').trim();
    const conversionType = byId('selThermalConversionType')?.value || 'radiometric';
    const includeRadiometric = Boolean(byId('chkThermalIncludeRadiometric')?.checked);
    if(!input){
      showError('Enter an input folder path to scan.');
      resetScan();
      return false;
    }
    const button = byId('btnScanThermalFolder');
    if(button){
      button.disabled = true;
      button.textContent = 'Scanning…';
    }
    try{
      const body = new FormData();
      body.append('input_dir', input);
      body.append('conversion_type', conversionType);
      body.append('include_radiometric', includeRadiometric ? 'true' : 'false');
      const response = await fetch(`${conversionUrl()}/scan`, {method:'POST', body});
      const result = await response.json().catch(()=>({}));
      if(!response.ok || !result.ok) throw new Error(result.detail || 'Could not scan the input folder.');
      lastScan = {...result.scan, inputValue: input, conversionType, includeRadiometric};
      renderScan(lastScan);
      setRunning(false);
      return Number(lastScan.supported || 0) > 0;
    }catch(error){
      resetScan();
      showError(error.message || 'Could not scan the input folder.');
      return false;
    }finally{
      if(button){
        button.disabled = false;
        button.textContent = 'Scan input folder';
      }
    }
  }

  function renderJob(job){
    const total = Number(job.total || 0);
    const completed = Number(job.completed || 0);
    const percent = total ? Math.round((completed / total) * 100) : 0;
    const progress = byId('thermalConvertProgress');
    const stats = byId('thermalConvertStats');
    const bar = byId('thermalConvertBar');
    const text = byId('thermalConvertProgressText');
    const status = byId('thermalConvertStatus');
    if(progress) progress.hidden = false;
    if(stats) stats.hidden = false;
    if(bar) bar.style.width = `${percent}%`;
    if(text){
      const current = job.current_file ? ` · ${job.current_file}` : '';
      text.textContent = `${completed} / ${total} (${percent}%)${current}`;
    }
    if(byId('thermalStatConverted')) byId('thermalStatConverted').textContent = job.converted || 0;
    if(byId('thermalStatSkipped')) byId('thermalStatSkipped').textContent = job.skipped || 0;
    if(byId('thermalStatFailed')) byId('thermalStatFailed').textContent = job.failed || 0;
    if(status) status.textContent = `Status: ${job.status}`;

    if(terminalStates.has(job.status)){
      setRunning(false);
      if(job.status === 'completed' && bar) bar.style.width = '100%';
      if(job.status === 'completed' && total === 0){
        showError('No JPG or JPEG files were found directly inside the selected input folder.');
      }else if(job.first_error){
        showError(`First error: ${job.first_error}`);
      }
    }
  }

  async function pollJob(){
    if(!activeJobId) return;
    try{
      const response = await fetch(`${conversionUrl()}/${encodeURIComponent(activeJobId)}`);
      const result = await response.json().catch(()=>({}));
      if(!response.ok || !result.ok) throw new Error(result.detail || 'Could not read conversion status.');
      renderJob(result.job);
      if(!terminalStates.has(result.job.status)){
        pollTimer = window.setTimeout(pollJob, 700);
      }else{
        activeJobId = null;
      }
    }catch(error){
      setRunning(false);
      showError(error.message || 'Could not read conversion status.');
    }
  }

  async function startConversion(){
    showError();
    const input = (byId('inpThermalInputDir')?.value || '').trim();
    const output = (byId('inpThermalOutputDir')?.value || '').trim();
    const conversionType = byId('selThermalConversionType')?.value || 'radiometric';
    const includeRadiometric = Boolean(byId('chkThermalIncludeRadiometric')?.checked);
    if(!input || !output){
      showError('Enter both the input folder and output folder paths.');
      return;
    }
    if(output === input){
      showError('Input and output folders must be different.');
      return;
    }
    if(
      !lastScan || lastScan.inputValue !== input ||
      lastScan.conversionType !== conversionType ||
      lastScan.includeRadiometric !== includeRadiometric
    ){
      const canConvert = await scanInputFolder();
      if(!canConvert) return;
    }
    if(Number(lastScan.supported || 0) < 1) return;

    const body = new FormData();
    body.append('input_dir', input);
    body.append('output_dir', output);
    body.append('conversion_type', conversionType);
    body.append('include_radiometric', includeRadiometric ? 'true' : 'false');
    body.append('output_format', byId('selThermalOutputFormat')?.value || 'jpg');
    body.append('quality', byId('inpThermalJpegQuality')?.value || '100');
    body.append('overwrite', byId('chkThermalOverwrite')?.checked ? 'true' : 'false');
    setRunning(true);
    const progress = byId('thermalConvertProgress');
    const stats = byId('thermalConvertStats');
    if(progress) progress.hidden = false;
    if(stats) stats.hidden = false;
    try{
      const response = await fetch(`${conversionUrl()}/start`, {method:'POST', body});
      const result = await response.json().catch(()=>({}));
      if(!response.ok || !result.ok) throw new Error(result.detail || 'Could not start conversion.');
      activeJobId = result.job.id;
      renderJob(result.job);
      window.clearTimeout(pollTimer);
      pollJob();
    }catch(error){
      setRunning(false);
      showError(error.message || 'Could not start conversion.');
    }
  }

  async function cancelConversion(){
    if(!activeJobId) return;
    try{
      await fetch(`${conversionUrl()}/${encodeURIComponent(activeJobId)}/cancel`, {method:'POST'});
      const status = byId('thermalConvertStatus');
      if(status) status.textContent = 'Cancellation requested…';
    }catch(error){
      showError(error.message || 'Could not cancel conversion.');
    }
  }

  document.addEventListener('DOMContentLoaded', () => {
    const modal = byId('thermalConvertModal');
    const openModal = () => {
      modal?.classList.remove('hidden');
      modal?.classList.add('show');
    };
    const closeModal = () => {
      modal?.classList.remove('show');
      modal?.classList.add('hidden');
    };
    byId('btnOpenThermalConvert')?.addEventListener('click', () => {
      openModal();
    });
    byId('btnCloseThermalConvert')?.addEventListener('click', () => {
      closeModal();
    });
    modal?.addEventListener('click', event => {
      if(event.target === modal) closeModal();
    });
    document.addEventListener('keydown', event => {
      if(event.key === 'Escape') closeModal();
    });
    byId('btnStartThermalConvert')?.addEventListener('click', startConversion);
    byId('btnScanThermalFolder')?.addEventListener('click', scanInputFolder);
    byId('btnCancelThermalConvert')?.addEventListener('click', cancelConversion);
    byId('inpThermalInputDir')?.addEventListener('input', () => {
      showError();
      resetScan();
    });
    const updateConversionType = () => {
      showError();
      const standard = byId('selThermalConversionType')?.value === 'standard';
      const typeHelp = byId('thermalConversionTypeHelp');
      const inputHelp = byId('thermalInputHelp');
      const includeRow = byId('thermalIncludeRadiometricRow');
      const includeCheckbox = byId('chkThermalIncludeRadiometric');
      if(typeHelp) typeHelp.textContent = standard
        ? 'Converts the visible pixels of ordinary JPG, JPEG, and PNG images to grayscale.'
        : 'Extracts the radiometric sensor data embedded in supported DJI and FLIR JPEGs.';
      if(inputHelp) inputHelp.textContent = standard
        ? 'Folder containing standard JPG, JPEG, or PNG images.'
        : 'Folder containing the original radiometric JPEGs.';
      if(includeRow) includeRow.hidden = !standard;
      if(!standard && includeCheckbox) includeCheckbox.checked = false;
      resetScan();
    };
    byId('selThermalConversionType')?.addEventListener('change', updateConversionType);
    byId('chkThermalIncludeRadiometric')?.addEventListener('change', resetScan);
    const updateFormatOptions = () => {
      const qualityRow = byId('thermalQualityRow');
      const isJpeg = byId('selThermalOutputFormat')?.value === 'jpg';
      if(qualityRow) qualityRow.style.display = isJpeg ? 'flex' : 'none';
      if(isJpeg && byId('inpThermalJpegQuality')) byId('inpThermalJpegQuality').value = '100';
    };
    byId('selThermalOutputFormat')?.addEventListener('change', updateFormatOptions);
    updateFormatOptions();
    updateConversionType();
    setRunning(false);
  });
})();
