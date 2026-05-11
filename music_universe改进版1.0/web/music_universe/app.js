async function loadSiteData() {
  if (window.SITE_DATA && typeof window.SITE_DATA === "object") {
    return window.SITE_DATA;
  }
  const response = await fetch("./music_universe_data_embedded.json", { cache: "no-store" });
  if (!response.ok) {
    throw new Error(
      `无法加载数据（HTTP ${response.status}）。请确认已先加载 music_universe_data_embedded.js，或与 index.html 同目录下提供 music_universe_data_embedded.json。`
    );
  }
  return response.json();
}

(async function(){
  const errorBanner = document.getElementById('errorBanner');
  function showError(msg){
    const tip = document.getElementById('protocolTip');
    if (tip) tip.hidden = true;
    errorBanner.style.display = 'block';
    errorBanner.textContent = msg;
  }

  if (!window.THREE || !window.THREE.OrbitControls) {
    showError('Failed to load Three.js or OrbitControls. Please refresh or open with a local server.');
    return;
  }
  try {
    const SITE_DATA = await loadSiteData();

    if (errorBanner) {
      errorBanner.style.display = 'none';
      errorBanner.textContent = '';
    }
    const protocolTipEl = document.getElementById('protocolTip');
    if (protocolTipEl && location.protocol !== 'file:') {
      protocolTipEl.hidden = true;
    }

  const spotifyItems = SITE_DATA.spotifyItems || [];
  const billboardItems = SITE_DATA.billboardItems || [];
  const genres = SITE_DATA.genres || [];
  const stats = SITE_DATA.stats || {};
  const billboardStats = SITE_DATA.billboardStats || {};
  const allItems = [...spotifyItems, ...billboardItems];

  const modeCopy = {
    universe: {
      label: 'Universe',
      text: 'Danceability, energy, and valence form the main coordinate system.'
    },
    galaxyCluster: {
      label: 'Galaxy Cluster',
      text: 'A PCA-like projection groups tracks with similar audio fingerprints.'
    },
    coverWall: {
      label: 'Cover Wall',
      text: 'Tracks are arranged as a poster wall for fast scanning and comparison.'
    },
    genreRings: {
      label: 'Genre Rings',
      text: 'Each genre becomes an orbit, making the catalog structure visible.'
    },
    billboardDecades: {
      label: 'Billboard Decades',
      text: 'Billboard #1 songs are staged by decade, weeks at #1, and audio mood.'
    }
  };

  function toFiniteNumber(value, fallback = 0){
    const n = Number(value);
    return Number.isFinite(n) ? n : fallback;
  }

  function cleanText(value, fallback = '-'){
    if (value === undefined || value === null) return fallback;
    const text = String(value).trim();
    if (!text || text.toLowerCase() === 'nan') return fallback;
    return text;
  }

  function normalizeSearchText(value){
    return cleanText(value, '').replace(/;/g, ' ').replace(/\s+/g, ' ').trim();
  }

  function makeSearchQuery(item){
    return normalizeSearchText(`${cleanText(item.title, '')} ${cleanText(item.artist, '')}`);
  }

  function formatNumber(value, digits = 0){
    const n = Number(value);
    if (!Number.isFinite(n)) return '-';
    return n.toLocaleString('en-US', { maximumFractionDigits: digits });
  }

  function clampPercent(value){
    const n = Number(value);
    if (!Number.isFinite(n)) return 0;
    return Math.max(0, Math.min(100, n));
  }

  function hashString(input){
    const str = String(input || "");
    let hash = 2166136261;
    for (let i = 0; i < str.length; i++){
      hash ^= str.charCodeAt(i);
      hash += (hash << 1) + (hash << 4) + (hash << 7) + (hash << 8) + (hash << 24);
    }
    return hash >>> 0;
  }

  function dot(a, b){
    let sum = 0;
    for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
    return sum;
  }

  function normalizeVector(vec){
    let norm = 0;
    for (let i = 0; i < vec.length; i++) norm += vec[i] * vec[i];
    norm = Math.sqrt(norm);
    if (norm < 1e-10) return vec.map(() => 0);
    return vec.map(v => v / norm);
  }

  function matVecMultiply(mat, vec){
    const out = new Array(mat.length).fill(0);
    for (let i = 0; i < mat.length; i++){
      let sum = 0;
      for (let j = 0; j < vec.length; j++) sum += mat[i][j] * vec[j];
      out[i] = sum;
    }
    return out;
  }

  function computeGalaxyProjection(items){
    if (!items.length) return [];

    const featureRows = items.map(item => [
      toFiniteNumber(item.danceability, 0),
      toFiniteNumber(item.energy, 0),
      toFiniteNumber(item.valence, 0),
      toFiniteNumber(item.tempo, 0) / 220,
      toFiniteNumber(item.popularity, 0) / 100
    ]);
    const dims = featureRows[0].length;
    const means = new Array(dims).fill(0);
    const stds = new Array(dims).fill(0);

    for (const row of featureRows){
      for (let j = 0; j < dims; j++) means[j] += row[j];
    }
    for (let j = 0; j < dims; j++) means[j] /= featureRows.length;

    for (const row of featureRows){
      for (let j = 0; j < dims; j++){
        const d = row[j] - means[j];
        stds[j] += d * d;
      }
    }
    for (let j = 0; j < dims; j++){
      stds[j] = Math.sqrt(stds[j] / Math.max(1, featureRows.length - 1));
      if (stds[j] < 1e-6) stds[j] = 1;
    }

    const normalizedRows = featureRows.map(row => row.map((v, j) => (v - means[j]) / stds[j]));
    const cov = Array.from({length: dims}, () => new Array(dims).fill(0));
    const denom = Math.max(1, normalizedRows.length - 1);

    for (const row of normalizedRows){
      for (let i = 0; i < dims; i++){
        for (let j = i; j < dims; j++){
          cov[i][j] += row[i] * row[j];
        }
      }
    }
    for (let i = 0; i < dims; i++){
      for (let j = i; j < dims; j++){
        cov[i][j] /= denom;
        cov[j][i] = cov[i][j];
      }
    }

    function leadingEigenVector(existingBasis){
      let vec = new Array(dims).fill(0);
      vec[existingBasis.length % dims] = 1;
      vec = normalizeVector(vec.map((v, i) => v + (i + 1) * 0.013));

      for (let iter = 0; iter < 42; iter++){
        let next = matVecMultiply(cov, vec);
        for (const base of existingBasis){
          const proj = dot(next, base);
          for (let k = 0; k < dims; k++) next[k] -= proj * base[k];
        }
        vec = normalizeVector(next);
      }
      return vec;
    }

    const basis = [];
    for (let i = 0; i < 3; i++){
      basis.push(leadingEigenVector(basis));
    }

    const projected = normalizedRows.map(row => basis.map(axis => dot(row, axis)));
    const halfSpan = [48, 26, 48];
    const axisRanges = [];

    for (let axis = 0; axis < 3; axis++){
      let min = Infinity;
      let max = -Infinity;
      for (let i = 0; i < projected.length; i++){
        const v = projected[i][axis];
        if (v < min) min = v;
        if (v > max) max = v;
      }
      axisRanges.push({
        center: (max + min) * 0.5,
        range: Math.max(1e-6, (max - min) * 0.5)
      });
    }

    return projected.map((coords, idx) => {
      const out = [0, 0, 0];
      for (let axis = 0; axis < 3; axis++){
        const info = axisRanges[axis];
        out[axis] = ((coords[axis] - info.center) / info.range) * halfSpan[axis];
      }

      const h = hashString(items[idx].id || idx);
      const jitterX = ((h & 255) / 255 - 0.5) * 2.4;
      const jitterY = (((h >>> 8) & 255) / 255 - 0.5) * 1.7;
      const jitterZ = (((h >>> 16) & 255) / 255 - 0.5) * 2.4;
      return [out[0] + jitterX, out[1] + jitterY, out[2] + jitterZ];
    });
  }

  const hasPrecomputedGalaxy = spotifyItems.every(
    item => item.modes && Array.isArray(item.modes.galaxyCluster) && item.modes.galaxyCluster.length === 3
  );
  if (!hasPrecomputedGalaxy){
    const galaxyCoords = computeGalaxyProjection(spotifyItems);
    for (let i = 0; i < spotifyItems.length; i++){
      const item = spotifyItems[i];
      if (!item.modes) item.modes = {};
      item.modes.galaxyCluster = galaxyCoords[i] || item.modes.universe || [0, 0, 0];
    }
  }

  const sceneContainer = document.getElementById('sceneContainer');
  const scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0x050816, 0.0048);

  const camera = new THREE.PerspectiveCamera(58, sceneContainer.clientWidth / sceneContainer.clientHeight, 0.1, 2000);
  camera.position.set(0, 18, 125);

  const renderer = new THREE.WebGLRenderer({antialias:true, alpha:true, powerPreference:'high-performance'});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2.5));
  renderer.setSize(sceneContainer.clientWidth, sceneContainer.clientHeight);
  renderer.outputEncoding = THREE.sRGBEncoding;
  sceneContainer.appendChild(renderer.domElement);

  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.06;
  controls.minDistance = 35;
  controls.maxDistance = 260;
  controls.target.set(0, 0, 0);

  const ambient = new THREE.AmbientLight(0xffffff, 0.7);
  scene.add(ambient);
  const light1 = new THREE.PointLight(0x8ea2ff, 1.8, 600);
  light1.position.set(45, 50, 35);
  scene.add(light1);
  const light2 = new THREE.PointLight(0x67e8f9, 1.2, 500);
  light2.position.set(-60, -20, -40);
  scene.add(light2);

  function createStarLayer(count, spreadX, spreadY, spreadZ, color, size, opacity){
    const geo = new THREE.BufferGeometry();
    const pos = new Float32Array(count * 3);
    for(let i=0;i<count;i++){
      pos[i*3] = (Math.random() - 0.5) * spreadX;
      pos[i*3+1] = (Math.random() - 0.5) * spreadY;
      pos[i*3+2] = (Math.random() - 0.5) * spreadZ;
    }
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    const mat = new THREE.PointsMaterial({
      color,
      size,
      transparent:true,
      opacity,
      depthWrite:false
    });
    return new THREE.Points(geo, mat);
  }

  const stars = createStarLayer(3000, 1450, 1100, 1450, 0xbdd6ff, 1.12, 0.66);
  const starsNear = createStarLayer(2200, 1080, 840, 1080, 0xffffff, 1.5, 0.56);
  const starsFar = createStarLayer(4200, 2300, 1700, 2300, 0x94a3ff, 0.82, 0.36);
  const starsDust = createStarLayer(2600, 2600, 1900, 2600, 0xc8d4ff, 0.55, 0.26);
  scene.add(stars);
  scene.add(starsNear);
  scene.add(starsFar);
  scene.add(starsDust);
  const twinkleLayers = [
    { points: stars, base: 0.66, amp: 0.2, speed: 0.9 + Math.random() * 0.7, phase: Math.random() * Math.PI * 2 },
    { points: starsNear, base: 0.56, amp: 0.28, speed: 1.1 + Math.random() * 0.8, phase: Math.random() * Math.PI * 2 },
    { points: starsFar, base: 0.36, amp: 0.16, speed: 0.7 + Math.random() * 0.5, phase: Math.random() * Math.PI * 2 },
    { points: starsDust, base: 0.26, amp: 0.12, speed: 0.6 + Math.random() * 0.45, phase: Math.random() * Math.PI * 2 }
  ];

  const centerOrb = new THREE.Mesh(
    new THREE.SphereGeometry(4.5, 32, 32),
    new THREE.MeshBasicMaterial({color:0x8cb8ff, transparent:true, opacity:.24})
  );
  scene.add(centerOrb);
  const centerCore = new THREE.Mesh(
    new THREE.SphereGeometry(2.3, 32, 32),
    new THREE.MeshBasicMaterial({color:0xe7f2ff, transparent:true, opacity:.42})
  );
  scene.add(centerCore);

  const pulseOrb = new THREE.Mesh(
    new THREE.SphereGeometry(10, 32, 32),
    new THREE.MeshBasicMaterial({color:0x67e8f9, transparent:true, opacity:.1})
  );
  scene.add(pulseOrb);

  const raycaster = new THREE.Raycaster();
  const mouse = new THREE.Vector2();
  let selectedMesh = null;
  let hoveredMesh = null;
  let currentItem = null;
  let autoRotate = true;
  let currentMode = 'universe';
  let currentDataset = 'spotify';
  let genreCycleTimer = null;
  let tourTimer = null;
  let tourStep = 0;
  let audioCtx = null;
  let previewAudio = null;
  let activeAudioNodes = [];
  let previewTimer = null;
  let artworkRequestId = 0;

  // 首次用户手势时预解锁 AudioContext，减轻部分浏览器对 Web Audio 的自动播放限制（见《问题排查》3.4）。
  (function installAudioGestureUnlock(){
    const unlock = async () => {
      const Ctx = window.AudioContext || window.webkitAudioContext;
      if (!Ctx) return;
      try {
        if (!audioCtx) audioCtx = new Ctx();
        if (audioCtx.state === 'suspended') await audioCtx.resume();
      } catch (_) {}
      document.removeEventListener('pointerdown', unlock, true);
      document.removeEventListener('touchstart', unlock, true);
    };
    document.addEventListener('pointerdown', unlock, true);
    document.addEventListener('touchstart', unlock, true);
  })();

  const songTitle = document.getElementById('songTitle');
  const songSub = document.getElementById('songSub');
  const pillGenre = document.getElementById('pillGenre');
  const pillPopularity = document.getElementById('pillPopularity');
  const pillExtra = document.getElementById('pillExtra');
  const metaList = document.getElementById('metaList');
  const artBox = document.getElementById('artBox');
  const featureBars = document.getElementById('featureBars');
  const modeName = document.getElementById('modeName');
  const modeText = document.getElementById('modeText');
  const visibleCount = document.getElementById('visibleCount');
  const hoverTooltip = document.getElementById('hoverTooltip');
  const activeDataset = document.getElementById('activeDataset');
  const activeFilter = document.getElementById('activeFilter');
  const activeModeStat = document.getElementById('activeModeStat');
  const datasetBadge = document.getElementById('datasetBadge');
  const playPreviewBtn = document.getElementById('playPreviewBtn');
  const stopPreviewBtn = document.getElementById('stopPreviewBtn');
  const spotifyLink = document.getElementById('spotifyLink');
  const youtubeLink = document.getElementById('youtubeLink');
  const audioStatus = document.getElementById('audioStatus');
  const insightBadge = document.getElementById('insightBadge');
  const insightSummary = document.getElementById('insightSummary');
  const moodGrid = document.getElementById('moodGrid');
  const topTracks = document.getElementById('topTracks');
  const surpriseBtn = document.getElementById('surpriseBtn');
  const tourBtn = document.getElementById('tourBtn');

  document.getElementById('statSongs').textContent = formatNumber(stats.songs ?? allItems.length);
  document.getElementById('statGenres').textContent = formatNumber(stats.genres ?? genres.length);
  document.getElementById('statAvgPop').textContent = formatNumber(stats.avg_popularity, 2);
  document.getElementById('statTopGenre').textContent = cleanText(stats.top_genre);

  const legendBox = document.getElementById('legendBox');
  function renderLegend(showGenres){
    legendBox.innerHTML = '';
    showGenres.forEach(g => {
      const row = document.createElement('div');
      row.className = 'legendItem';
      const dot = document.createElement('span');
      dot.className = 'dot';
      dot.style.color = g.color;
      dot.style.background = g.color;
      const name = document.createElement('span');
      name.textContent = cleanText(g.name);
      row.appendChild(dot);
      row.appendChild(name);
      legendBox.appendChild(row);
    });
  }
  renderLegend(genres.map((g, i) => ({name:g, color:(spotifyItems.find(x => x.genre === g)||{}).color || '#8ea2ff'})));

  const genreFilter = document.getElementById('genreFilter');
  function fillGenreFilter(){
    genreFilter.innerHTML = '<option value="All Genres">All Genres</option>';
    genres.forEach(g => {
      const opt = document.createElement('option');
      opt.value = g;
      opt.textContent = g;
      genreFilter.appendChild(opt);
    });
    const bill = document.createElement('option');
    bill.value = 'Billboard Only';
    bill.textContent = 'Billboard Only';
    genreFilter.appendChild(bill);
  }
  fillGenreFilter();

  const baseSphere = new THREE.SphereGeometry(1.0, 26, 26);
  const meshGroup = new THREE.Group();
  scene.add(meshGroup);
  const itemMeshes = [];

  function makeMesh(item){
    const material = new THREE.MeshStandardMaterial({
      color: new THREE.Color(item.color || '#8ea2ff'),
      emissive: new THREE.Color(item.color || '#8ea2ff'),
      emissiveIntensity: item.dataset === 'billboard' ? 0.85 : 1.2,
      metalness: 0.02,
      roughness: 0.14,
      transparent: true,
      opacity: 0.96,
    });
    const mesh = new THREE.Mesh(baseSphere, material);
    const s = item.size || 1.0;
    mesh.scale.set(s, s, s);
    mesh.userData.item = item;
    const p = item.modes.universe || item.modes.billboardDecades || [0,0,0];
    mesh.position.set(p[0], p[1], p[2]);
    mesh.userData.current = new THREE.Vector3(p[0], p[1], p[2]);
    mesh.userData.target = new THREE.Vector3(p[0], p[1], p[2]);
    mesh.userData.baseScale = s;
    mesh.userData.visibleWanted = true;
    mesh.userData.dataset = item.dataset;
    itemMeshes.push(mesh);
    meshGroup.add(mesh);
  }

  allItems.forEach(makeMesh);

  function updateModeSummary(){
    const copy = modeCopy[currentMode] || modeCopy.universe;
    const selectedGenre = genreFilter.value || 'All Genres';
    const keyword = cleanText(document.getElementById('keywordInput').value, '');
    const threshold = Number(document.getElementById('popNumber').value || 0);
    const filterParts = [selectedGenre];
    if (threshold > 0) filterParts.push(`${currentDataset === 'billboard' ? 'Weeks' : 'Popularity'} >= ${threshold}`);
    if (keyword) filterParts.push(`"${keyword}"`);

    modeName.textContent = copy.label;
    modeText.textContent = copy.text;
    activeDataset.textContent = currentDataset === 'billboard' ? 'Billboard' : 'Spotify';
    activeFilter.textContent = filterParts.join(' · ');
    activeModeStat.textContent = copy.label;
    datasetBadge.textContent = currentDataset === 'billboard' ? 'Billboard' : 'Spotify';
  }

  function valuePercent(item, key){
    const value = toFiniteNumber(item[key], 0);
    if (item.dataset === 'spotify' && ['danceability', 'energy', 'valence'].includes(key)) {
      return clampPercent(value * 100);
    }
    return clampPercent(value);
  }

  function getVisibleMeshes(){
    return itemMeshes.filter(mesh => mesh.userData.visibleWanted && mesh.userData.item.dataset === currentDataset);
  }

  function getVisibleItems(){
    return getVisibleMeshes().map(mesh => mesh.userData.item);
  }

  function average(items, getter){
    if (!items.length) return 0;
    let sum = 0;
    items.forEach(item => sum += getter(item));
    return sum / items.length;
  }

  function topLabel(items, getter){
    const counts = new Map();
    items.forEach(item => {
      const label = cleanText(getter(item), '');
      if (!label) return;
      label.split(';').map(v => v.trim()).filter(Boolean).forEach(part => {
        counts.set(part, (counts.get(part) || 0) + 1);
      });
    });
    return [...counts.entries()].sort((a, b) => b[1] - a[1])[0] || ['-', 0];
  }

  function renderMoodMetric(label, value){
    const row = document.createElement('div');
    row.className = 'moodItem';
    const top = document.createElement('div');
    top.className = 'moodItemTop';
    const name = document.createElement('span');
    name.textContent = label;
    const number = document.createElement('strong');
    number.textContent = String(Math.round(value));
    const track = document.createElement('div');
    track.className = 'moodTrack';
    const fill = document.createElement('div');
    fill.className = 'moodFill';
    fill.style.width = `${clampPercent(value)}%`;
    top.appendChild(name);
    top.appendChild(number);
    track.appendChild(fill);
    row.appendChild(top);
    row.appendChild(track);
    moodGrid.appendChild(row);
  }

  function focusMesh(mesh, shouldPlay = false, previewOpts){
    if (!mesh) return;
    if (selectedMesh) selectedMesh.scale.setScalar(selectedMesh.userData.baseScale);
    selectedMesh = mesh;
    selectedMesh.scale.setScalar(selectedMesh.userData.baseScale * 1.8);
    updateSongCard(selectedMesh.userData.item);
    const worldPos = new THREE.Vector3();
    selectedMesh.getWorldPosition(worldPos);
    controls.target.lerp(worldPos, 0.35);
    if (shouldPlay) void playTrackPreview(selectedMesh.userData.item, previewOpts || {});
  }

  function renderTopTracks(items){
    topTracks.innerHTML = '';
    const metricKey = currentDataset === 'billboard' ? 'weeks' : 'popularity';
    const metricLabel = currentDataset === 'billboard' ? 'wks' : 'pop';
    const ranked = [...getVisibleMeshes()]
      .sort((a, b) => toFiniteNumber(b.userData.item[metricKey]) - toFiniteNumber(a.userData.item[metricKey]))
      .slice(0, 5);

    if (!ranked.length) {
      const empty = document.createElement('div');
      empty.className = 'notice';
      empty.textContent = '当前筛选没有可见歌曲，试着降低阈值或清空搜索。';
      topTracks.appendChild(empty);
      return;
    }

    ranked.forEach((mesh, idx) => {
      const item = mesh.userData.item;
      const row = document.createElement('button');
      row.className = 'trackRow';
      row.type = 'button';
      const rank = document.createElement('span');
      rank.className = 'trackRank';
      rank.textContent = String(idx + 1);
      const info = document.createElement('span');
      info.className = 'trackInfo';
      const title = document.createElement('strong');
      title.textContent = cleanText(item.title, 'Unknown');
      const sub = document.createElement('span');
      sub.textContent = cleanText(item.artist);
      const metric = document.createElement('span');
      metric.className = 'trackMetric';
      metric.textContent = `${formatNumber(item[metricKey])} ${metricLabel}`;
      info.appendChild(title);
      info.appendChild(sub);
      row.appendChild(rank);
      row.appendChild(info);
      row.appendChild(metric);
      row.addEventListener('click', () => focusMesh(mesh, true));
      topTracks.appendChild(row);
    });
  }

  function updateInsights(){
    const items = getVisibleItems();
    moodGrid.innerHTML = '';
    insightBadge.textContent = `${formatNumber(items.length)} shown`;

    if (!items.length) {
      insightSummary.textContent = '当前筛选没有可见歌曲。降低阈值、清空关键词，或切换到其他模式可以重新展开星图。';
      renderTopTracks(items);
      return;
    }

    const dance = average(items, item => valuePercent(item, 'danceability'));
    const energy = average(items, item => valuePercent(item, 'energy'));
    const mood = currentDataset === 'billboard'
      ? average(items, item => valuePercent(item, 'happiness'))
      : average(items, item => valuePercent(item, 'valence'));
    const tempo = currentDataset === 'billboard'
      ? average(items, item => valuePercent(item, 'weeks'))
      : Math.min(100, average(items, item => toFiniteNumber(item.tempo, 0)) / 2);
    const [dominantLabel, dominantCount] = currentDataset === 'billboard'
      ? topLabel(items, item => item.topic)
      : topLabel(items, item => item.genre);
    const topItem = [...items].sort((a, b) => {
      const key = currentDataset === 'billboard' ? 'weeks' : 'popularity';
      return toFiniteNumber(b[key]) - toFiniteNumber(a[key]);
    })[0];

    const moodWord = mood >= 66 ? '明亮积极' : mood >= 40 ? '情绪均衡' : '偏冷偏忧郁';
    const energyWord = energy >= 66 ? '高能量' : energy >= 40 ? '中等能量' : '低能量';
    const dataName = currentDataset === 'billboard' ? 'Billboard 冠军曲' : 'Spotify 曲库';
    const labelName = currentDataset === 'billboard' ? '主题' : '流派';
    insightSummary.textContent = `${dataName} 当前可见 ${formatNumber(items.length)} 首，整体呈现 ${energyWord}、${moodWord} 的声音画像；最常见${labelName}是 ${dominantLabel}（${dominantCount} 首），代表歌曲是 ${cleanText(topItem.title)}。`;

    renderMoodMetric('Dance', dance);
    renderMoodMetric('Energy', energy);
    renderMoodMetric(currentDataset === 'billboard' ? 'Happiness' : 'Valence', mood);
    renderMoodMetric(currentDataset === 'billboard' ? 'Weeks' : 'Tempo', tempo);
    renderTopTracks(items);
  }

  function setMode(modeName){
    currentMode = modeName;
    document.querySelectorAll('button.mode').forEach(btn => btn.classList.toggle('active', btn.dataset.mode === modeName));
    if (modeName === 'billboardDecades') {
      currentDataset = 'billboard';
      genreFilter.value = 'Billboard Only';
      renderLegend([{name:'Billboard #1 songs', color:'#FFD166'}]);
    } else {
      currentDataset = 'spotify';
      if (genreFilter.value === 'Billboard Only') genreFilter.value = 'All Genres';
      renderLegend(genres.map(g => ({name:g, color:(spotifyItems.find(x => x.genre === g)||{}).color || '#8ea2ff'})));
    }

    itemMeshes.forEach(mesh => {
      const item = mesh.userData.item;
      if (currentDataset === 'spotify' && item.dataset !== 'spotify') {
        mesh.userData.target.set(0, -240, 0);
        mesh.userData.visibleWanted = false;
      } else if (currentDataset === 'billboard' && item.dataset !== 'billboard') {
        mesh.userData.target.set(0, -240, 0);
        mesh.userData.visibleWanted = false;
      } else {
        const modeKey = (item.modes[modeName] ? modeName : (item.dataset === 'billboard' ? 'billboardSpiral' : 'universe'));
        const p = item.modes[modeKey] || [0,0,0];
        mesh.userData.target.set(p[0], p[1], p[2]);
        mesh.userData.visibleWanted = true;
      }
    });
    updateModeSummary();
    applyFilter();
  }

  function renderGeneratedCover(item){
    const title = cleanText(item.title, 'Music');
    const artist = cleanText(item.artist, 'Unknown Artist');
    const color = item.color || (item.dataset === 'billboard' ? '#FFD166' : '#5eead4');
    const hueShift = hashString(`${title}${artist}`) % 70;
    artBox.innerHTML = '';
    artBox.style.background = `
      radial-gradient(circle at 28% 18%, rgba(255,255,255,.38), transparent 24%),
      linear-gradient(135deg, ${color}, hsl(${180 + hueShift}, 72%, 34%) 46%, #111722)
    `;

    const wrap = document.createElement('div');
    wrap.className = 'coverFallback';
    const initials = document.createElement('div');
    initials.className = 'coverInitials';
    initials.textContent = title.split(/\s+/).slice(0, 2).map(s => s[0] || '').join('').toUpperCase() || 'MU';
    const caption = document.createElement('div');
    caption.className = 'coverCaption';
    caption.textContent = `${title} · ${artist}`;
    wrap.appendChild(initials);
    wrap.appendChild(caption);
    artBox.appendChild(wrap);
  }

  function getArtworkCache(key){
    try {
      return window.localStorage.getItem(`musicUniverseArtwork:${key}`);
    } catch (_) {
      return '';
    }
  }

  function setArtworkCache(key, value){
    try {
      window.localStorage.setItem(`musicUniverseArtwork:${key}`, value);
    } catch (_) {}
  }

  async function hydrateArtwork(item){
    const imgSrc = item.image_path || item.image_url || item.artwork_url || '';
    const requestId = ++artworkRequestId;
    if (imgSrc) {
      artBox.innerHTML = `<img alt="album art" referrerpolicy="no-referrer" loading="lazy" src="${imgSrc}" />`;
      return;
    }

    renderGeneratedCover(item);

    const query = makeSearchQuery(item);
    if (!query) return;
    const cacheKey = query.toLowerCase();
    const cached = getArtworkCache(cacheKey);
    if (cached) {
      if (currentItem && currentItem.id === item.id && requestId === artworkRequestId) {
        artBox.innerHTML = `<img alt="album art" referrerpolicy="no-referrer" loading="lazy" src="${cached}" />`;
      }
      return;
    }

    try {
      const url = `https://itunes.apple.com/search?media=music&entity=song&limit=1&term=${encodeURIComponent(query)}`;
      const response = await fetch(url);
      if (!response.ok) return;
      const data = await response.json();
      const found = data && data.results && data.results[0] && data.results[0].artworkUrl100;
      if (!found) return;
      const largeArt = found.replace('100x100bb', '600x600bb');
      setArtworkCache(cacheKey, largeArt);
      if (currentItem && currentItem.id === item.id && requestId === artworkRequestId) {
        artBox.innerHTML = `<img alt="album art" referrerpolicy="no-referrer" loading="lazy" src="${largeArt}" />`;
      }
    } catch (_) {
      // Offline or blocked network: the generated cover remains visible.
    }
  }

  function stopPreview(message){
    if (previewTimer) {
      clearTimeout(previewTimer);
      previewTimer = null;
    }
    if (previewAudio) {
      try {
        previewAudio.pause();
        previewAudio.currentTime = 0;
      } catch (_) {}
      previewAudio = null;
    }
    activeAudioNodes.forEach(node => {
      try {
        if (typeof node.stop === 'function') node.stop();
        if (typeof node.disconnect === 'function') node.disconnect();
      } catch (_) {}
    });
    activeAudioNodes = [];
    if (message) audioStatus.textContent = message;
    playPreviewBtn.textContent = 'Play Signature';
  }

  function stopSonicPreview(message){
    stopPreview(message);
  }

  function scheduleTone(time, frequency, duration, gainValue, type = 'sine'){
    const oscillator = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    oscillator.type = type;
    oscillator.frequency.setValueAtTime(frequency, time);
    gain.gain.setValueAtTime(0.0001, time);
    gain.gain.exponentialRampToValueAtTime(Math.max(0.0002, gainValue), time + 0.012);
    gain.gain.exponentialRampToValueAtTime(0.0001, time + duration);
    oscillator.connect(gain).connect(audioCtx.destination);
    oscillator.start(time);
    oscillator.stop(time + duration + 0.02);
    activeAudioNodes.push(oscillator, gain);
  }

  async function playSonicPreview(item){
    if (!item) return;
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    if (!AudioContext) {
      audioStatus.textContent = '当前浏览器不支持 Web Audio，无法播放声纹预览。';
      return;
    }

    stopPreview();
    audioCtx = audioCtx || new AudioContext();
    if (audioCtx.state === 'suspended') await audioCtx.resume();

    const tempo = Math.max(70, Math.min(180, toFiniteNumber(item.tempo, item.dataset === 'billboard' ? 118 : 120)));
    const beat = 60 / tempo;
    const energy = item.dataset === 'billboard' ? clampPercent(item.energy) / 100 : clampPercent(toFiniteNumber(item.energy) * 100) / 100;
    const dance = item.dataset === 'billboard' ? clampPercent(item.danceability) / 100 : clampPercent(toFiniteNumber(item.danceability) * 100) / 100;
    const valence = item.dataset === 'billboard' ? clampPercent(item.happiness) / 100 : clampPercent(toFiniteNumber(item.valence) * 100) / 100;
    const base = 160 + Math.round(valence * 220);
    const chord = valence >= 0.5 ? [1, 1.25, 1.5, 2] : [1, 1.2, 1.5, 1.8];
    const steps = Math.max(6, Math.min(14, Math.round(7 + dance * 7)));
    const start = audioCtx.currentTime + 0.04;

    for (let i = 0; i < steps; i++){
      const t = start + i * beat * 0.5;
      const note = chord[i % chord.length];
      const velocity = (0.035 + energy * 0.08) * 1.45;
      scheduleTone(t, base * note, beat * 0.34, velocity, i % 3 === 0 ? 'triangle' : 'sine');
      if (i % 2 === 0) scheduleTone(t, 55 + energy * 45, beat * 0.12, (0.05 + energy * 0.08) * 1.35, 'square');
      if (dance > 0.55 && i % 2 === 1) scheduleTone(t + beat * 0.22, 900 + valence * 800, beat * 0.08, (0.012 + dance * 0.025) * 1.4, 'sawtooth');
    }

    const duration = steps * beat * 0.5 + 0.35;
    playPreviewBtn.textContent = 'Replay Signature';
    audioStatus.textContent = `正在播放 ${cleanText(item.title, 'this track')} 的声纹预览：tempo ${Math.round(tempo)} BPM，energy ${Math.round(energy * 100)}，valence ${Math.round(valence * 100)}。`;
    previewTimer = setTimeout(() => {
      stopPreview('声纹预览已结束。点击星点或 Play Signature 可再次播放。');
    }, duration * 1000);
  }

  function hasHttpPreview(item){
    const u = cleanText(item.preview_url || item.previewUrl, '');
    return u.startsWith('http://') || u.startsWith('https://');
  }

  async function playTrackPreview(item, opts = {}){
    if (!item) return;
    const surprise = opts.surprise === true;
    const fastFailMs = surprise ? 2600 : null;

    if (!hasHttpPreview(item)) {
      await playSonicPreview(item);
      return;
    }

    const previewUrl = cleanText(item.preview_url || item.previewUrl, '');

    stopPreview();
    let finished = false;
    let failTimer = null;

    const clearFailTimer = () => {
      if (failTimer) {
        clearTimeout(failTimer);
        failTimer = null;
      }
    };

    const finalizeEnded = () => {
      if (finished) return;
      finished = true;
      clearFailTimer();
      stopPreview('真实试听已结束。点击 Play Preview 可再次播放。');
    };

    const fallbackSignature = async () => {
      if (finished) return;
      finished = true;
      clearFailTimer();
      try {
        if (previewAudio) {
          previewAudio.pause();
          previewAudio.removeAttribute('src');
          previewAudio.load();
        }
      } catch (_) {}
      previewAudio = null;
      await playSonicPreview(item);
    };

    previewAudio = new Audio(previewUrl);
    previewAudio.volume = 0.82;
    playPreviewBtn.textContent = 'Replay Preview';
    audioStatus.textContent = surprise
      ? `Surprise：正在加载试听… 若约 ${Math.round((fastFailMs || 0) / 100) / 10}s 仍无声音将自动播放声纹。`
      : `正在播放 ${cleanText(item.title, 'this track')} 的真实 30 秒试听片段。若网络或地区不可用，会自动回退到声纹预览。`;

    previewAudio.addEventListener('ended', finalizeEnded, { once: true });
    previewAudio.addEventListener('error', () => { void fallbackSignature(); }, { once: true });
    previewAudio.addEventListener('playing', () => {
      clearFailTimer();
      if (surprise) {
        audioStatus.textContent = `正在播放 ${cleanText(item.title, 'this track')} 的试听片段。`;
      }
    }, { once: true });

    if (fastFailMs != null) {
      failTimer = setTimeout(() => {
        if (finished) return;
        try {
          const a = previewAudio;
          if (!a) return;
          if (a.error) {
            void fallbackSignature();
            return;
          }
          if (!a.ended && a.currentTime < 0.025) {
            void fallbackSignature();
          }
        } catch (_) {
          void fallbackSignature();
        }
      }, fastFailMs);
    }

    try {
      await previewAudio.play();
    } catch (_) {
      await fallbackSignature();
    }
  }

  function updateSongCard(item){
    currentItem = item;
    songTitle.textContent = cleanText(item.title, 'Unknown');
    songSub.textContent = `${cleanText(item.artist)} · ${cleanText(item.album)}`;
    pillGenre.textContent = `${item.dataset === 'billboard' ? 'Topic' : 'Genre'}: ${cleanText(item.dataset === 'billboard' ? item.topic : item.genre)}`;
    if (item.dataset === 'billboard') {
      pillPopularity.textContent = `Weeks at #1: ${formatNumber(item.weeks)}`;
      pillExtra.textContent = `Decade: ${cleanText(item.decade)}`;
    } else {
      pillPopularity.textContent = `Popularity: ${formatNumber(item.popularity)}`;
      pillExtra.textContent = `Tempo: ${formatNumber(item.tempo)} BPM`;
    }

    const rows = [];
    const barRows = [];
    if (item.dataset === 'billboard') {
      rows.push(['Danceability', item.danceability]);
      rows.push(['Energy', item.energy]);
      rows.push(['Happiness', item.happiness]);
      rows.push(['Topic', cleanText(item.topic)]);
      barRows.push(['Dance', item.danceability]);
      barRows.push(['Energy', item.energy]);
      barRows.push(['Happy', item.happiness]);
      barRows.push(['Weeks', clampPercent((toFiniteNumber(item.weeks) / 12) * 100)]);
    } else {
      rows.push(['Danceability', item.danceability]);
      rows.push(['Energy', item.energy]);
      rows.push(['Valence', item.valence]);
      rows.push(['Genre', cleanText(item.genre)]);
      barRows.push(['Dance', toFiniteNumber(item.danceability) * 100]);
      barRows.push(['Energy', toFiniteNumber(item.energy) * 100]);
      barRows.push(['Valence', toFiniteNumber(item.valence) * 100]);
      barRows.push(['Popularity', item.popularity]);
    }

    featureBars.innerHTML = '';
    barRows.forEach(([label, value]) => {
      const pct = clampPercent(value);
      const row = document.createElement('div');
      row.className = 'featureBar';
      row.innerHTML = `<span>${label}</span><div class="featureTrack"><div class="featureFill" style="width:${pct}%"></div></div><span>${Math.round(pct)}</span>`;
      featureBars.appendChild(row);
    });

    metaList.innerHTML = rows.map(([k,v]) => `<div class="metaItem"><b>${k}</b><br/>${cleanText(v)}</div>`).join('');

    const query = makeSearchQuery(item);
    spotifyLink.href = `https://open.spotify.com/search/${encodeURIComponent(query)}`;
    youtubeLink.href = `https://www.youtube.com/results?search_query=${encodeURIComponent(query)}`;
    playPreviewBtn.textContent = cleanText(item.preview_url || item.previewUrl, '') ? 'Play Preview' : 'Play Signature';
    playPreviewBtn.disabled = false;
    stopPreviewBtn.disabled = false;
    hydrateArtwork(item);
  }

  function applyFilter(){
    const selectedGenre = genreFilter.value;
    const keyword = (document.getElementById('keywordInput').value || '').trim().toLowerCase();
    const threshold = Number(document.getElementById('popNumber').value || 0);
    let shown = 0;

    itemMeshes.forEach(mesh => {
      const item = mesh.userData.item;
      const isDatasetOk = currentDataset === item.dataset;
      let visible = isDatasetOk;

      if (visible && currentDataset === 'spotify') {
        if (selectedGenre !== 'All Genres' && selectedGenre !== 'Billboard Only') {
          visible = item.genre === selectedGenre;
        }
        if (visible && threshold > 0) visible = (item.popularity || 0) >= threshold;
      }
      if (visible && currentDataset === 'billboard') {
        if (selectedGenre === 'Billboard Only' || selectedGenre === 'All Genres') {
          visible = true;
        } else {
          visible = (item.genre || '').toLowerCase().includes(selectedGenre.toLowerCase());
        }
        if (visible && threshold > 0) visible = (item.weeks || 0) >= threshold;
      }
      if (visible && keyword) {
        const hay = `${item.title || ''} ${item.artist || ''} ${item.album || ''} ${item.genre || ''}`.toLowerCase();
        visible = hay.includes(keyword);
      }

      mesh.userData.visibleWanted = visible;
      if (!visible) {
        mesh.userData.materialTargetOpacity = 0.05;
      } else {
        mesh.userData.materialTargetOpacity = 0.92;
        shown++;
      }
    });
    visibleCount.textContent = formatNumber(shown);
    updateModeSummary();
    updateInsights();
  }

  function focusGenreCycle(){
    if (genreCycleTimer) {
      clearInterval(genreCycleTimer);
      genreCycleTimer = null;
      document.getElementById('genreCycleBtn').textContent = 'Genre Cycle';
      genreFilter.value = 'All Genres';
      applyFilter();
      return;
    }
    if (currentDataset !== 'spotify') setMode('genreRings');
    let idx = 0;
    document.getElementById('genreCycleBtn').textContent = 'Stop Cycle';
    genreCycleTimer = setInterval(() => {
      if (!genres.length) return;
      genreFilter.value = genres[idx % genres.length];
      applyFilter();
      idx++;
    }, 2200);
  }

  function focusBestVisible(shouldPlay = false){
    const metricKey = currentDataset === 'billboard' ? 'weeks' : 'popularity';
    const best = getVisibleMeshes()
      .sort((a, b) => toFiniteNumber(b.userData.item[metricKey]) - toFiniteNumber(a.userData.item[metricKey]))[0];
    if (best) focusMesh(best, shouldPlay);
  }

  function surpriseMe(){
    const visible = getVisibleMeshes();
    if (!visible.length) return;
    const withPreview = visible.filter(m => hasHttpPreview(m.userData.item));
    const pool = withPreview.length && Math.random() < 0.78 ? withPreview : visible;
    const weighted = pool
      .map(mesh => {
        const item = mesh.userData.item;
        const metric = item.dataset === 'billboard' ? toFiniteNumber(item.weeks) : toFiniteNumber(item.popularity);
        let score = metric + (hashString(item.id) % 25);
        if (hasHttpPreview(item)) score += 18;
        return { mesh, score };
      })
      .sort((a, b) => b.score - a.score)
      .slice(0, Math.min(28, pool.length));
    const pick = weighted[Math.floor(Math.random() * weighted.length)];
    if (pick) focusMesh(pick.mesh, true, { surprise: true });
  }

  function stopTour(message = 'Presentation Tour'){
    if (tourTimer) {
      clearInterval(tourTimer);
      tourTimer = null;
    }
    tourBtn.textContent = message;
  }

  function runTourStep(){
    const tourSteps = [
      { mode: 'universe', genre: 'All Genres', threshold: 70, keyword: '' },
      { mode: 'galaxyCluster', genre: 'acoustic', threshold: 60, keyword: '' },
      { mode: 'genreRings', genre: 'heavy-metal', threshold: 45, keyword: '' },
      { mode: 'coverWall', genre: 'All Genres', threshold: 75, keyword: '' },
      { mode: 'billboardDecades', genre: 'Billboard Only', threshold: 5, keyword: '' }
    ];
    const step = tourSteps[tourStep % tourSteps.length];
    tourStep++;
    setMode(step.mode);
    genreFilter.value = step.genre;
    document.getElementById('keywordInput').value = step.keyword;
    document.getElementById('popSlider').value = step.threshold;
    document.getElementById('popNumber').value = step.threshold;
    applyFilter();
    focusBestVisible(false);
  }

  function togglePresentationTour(){
    if (tourTimer) {
      stopTour();
      return;
    }
    if (genreCycleTimer) {
      clearInterval(genreCycleTimer);
      genreCycleTimer = null;
      document.getElementById('genreCycleBtn').textContent = 'Genre Cycle';
    }
    tourBtn.textContent = 'Stop Tour';
    tourStep = 0;
    runTourStep();
    tourTimer = setInterval(runTourStep, 4200);
  }

  function onPointerMove(event){
    const rect = renderer.domElement.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObjects(itemMeshes.filter(m => m.visible && m.material.opacity > 0.25));
    hoveredMesh = intersects.length ? intersects[0].object : null;
    if (hoveredMesh) {
      const item = hoveredMesh.userData.item;
      const metric = item.dataset === 'billboard'
        ? `${formatNumber(item.weeks)} weeks · ${cleanText(item.decade)}`
        : `${formatNumber(item.popularity)} popularity · ${cleanText(item.genre)}`;
      hoverTooltip.innerHTML = `<strong>${cleanText(item.title, 'Unknown')}</strong>${cleanText(item.artist)}<br><span>${metric}</span>`;
      hoverTooltip.style.display = 'block';
      const localX = event.clientX - rect.left;
      const localY = event.clientY - rect.top;
      const tooltipWidth = 260;
      hoverTooltip.style.left = `${Math.min(localX + 18, rect.width - tooltipWidth - 12)}px`;
      hoverTooltip.style.top = `${Math.max(12, localY - 18)}px`;
      renderer.domElement.style.cursor = 'pointer';
    } else {
      hoverTooltip.style.display = 'none';
      renderer.domElement.style.cursor = 'grab';
    }
  }

  function onClick(){
    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObjects(itemMeshes.filter(m => m.visible));
    if (intersects.length > 0) {
      focusMesh(intersects[0].object, true);
    }
  }

  renderer.domElement.addEventListener('pointermove', onPointerMove);
  renderer.domElement.addEventListener('click', onClick);
  renderer.domElement.addEventListener('pointerleave', () => {
    hoveredMesh = null;
    hoverTooltip.style.display = 'none';
    renderer.domElement.style.cursor = 'default';
  });
  playPreviewBtn.addEventListener('click', () => playTrackPreview(currentItem));
  stopPreviewBtn.addEventListener('click', () => stopPreview('播放已停止。'));
  surpriseBtn.addEventListener('click', surpriseMe);
  tourBtn.addEventListener('click', togglePresentationTour);

  document.querySelectorAll('button.mode').forEach(btn => btn.addEventListener('click', () => setMode(btn.dataset.mode)));
  document.getElementById('applyBtn').addEventListener('click', applyFilter);
  genreFilter.addEventListener('change', applyFilter);
  document.getElementById('keywordInput').addEventListener('input', applyFilter);
  document.getElementById('resetBtn').addEventListener('click', () => {
    document.getElementById('keywordInput').value = '';
    document.getElementById('popSlider').value = 0;
    document.getElementById('popNumber').value = 0;
    genreFilter.value = currentDataset === 'billboard' ? 'Billboard Only' : 'All Genres';
    applyFilter();
  });
  document.getElementById('rotationBtn').addEventListener('click', () => {
    autoRotate = !autoRotate;
    document.getElementById('rotationBtn').textContent = autoRotate ? 'Pause Rotation' : 'Resume Rotation';
  });
  document.getElementById('genreCycleBtn').addEventListener('click', focusGenreCycle);

  const popSlider = document.getElementById('popSlider');
  const popNumber = document.getElementById('popNumber');
  popSlider.addEventListener('input', () => {
    popNumber.value = popSlider.value;
    applyFilter();
  });
  popNumber.addEventListener('input', () => {
    popSlider.value = popNumber.value;
    applyFilter();
  });

  window.addEventListener('resize', () => {
    const w = sceneContainer.clientWidth;
    const h = sceneContainer.clientHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  });

  setMode('universe');
  applyFilter();
  const firstHighlight = itemMeshes
    .filter(mesh => mesh.userData.item.dataset === 'spotify')
    .sort((a, b) => toFiniteNumber(b.userData.item.popularity) - toFiniteNumber(a.userData.item.popularity))[0];
  if (firstHighlight) {
    selectedMesh = firstHighlight;
    updateSongCard(firstHighlight.userData.item);
  }

  const clock = new THREE.Clock();
  function animate(){
    requestAnimationFrame(animate);
    const elapsed = clock.getElapsedTime();
    centerOrb.rotation.y += 0.0025;
    centerCore.rotation.y -= 0.0032;
    pulseOrb.scale.setScalar(1 + Math.sin(elapsed * 1.4) * 0.06);
    centerCore.scale.setScalar(1 + Math.sin(elapsed * 2.2) * 0.025);
    stars.rotation.y += 0.00025;
    stars.rotation.x += 0.00004;
    starsNear.rotation.y -= 0.00018;
    starsNear.rotation.x += 0.00006;
    starsFar.rotation.y += 0.00008;
    starsFar.rotation.x -= 0.00002;
    starsDust.rotation.y -= 0.00005;
    starsDust.rotation.x += 0.00001;
    for (const layer of twinkleLayers) {
      const glow = layer.base + Math.sin(elapsed * layer.speed + layer.phase) * layer.amp;
      layer.points.material.opacity = Math.max(0.08, Math.min(0.95, glow));
    }
    if (autoRotate) meshGroup.rotation.y += 0.0018;

    itemMeshes.forEach(mesh => {
      const target = mesh.userData.target;
      mesh.position.lerp(target, 0.07);
      const visibleWanted = mesh.userData.visibleWanted;
      const targetOpacity = visibleWanted ? 0.92 : 0.05;
      mesh.material.opacity += (targetOpacity - mesh.material.opacity) * 0.08;
      let wantedScale = visibleWanted ? mesh.userData.baseScale : mesh.userData.baseScale * 0.35;
      if (visibleWanted && mesh === hoveredMesh) wantedScale *= 1.35;
      if (visibleWanted && mesh === selectedMesh) wantedScale *= 1.85;
      const currentScale = mesh.scale.x;
      const nextScale = currentScale + (wantedScale - currentScale) * 0.08;
      mesh.scale.setScalar(nextScale);
      mesh.visible = mesh.material.opacity > 0.03;
    });

    controls.update();
    renderer.render(scene, camera);
  }
  animate();

  } catch (error) {
    console.error(error);
    showError(`JS Error: ${error && error.message ? error.message : error}`);
  }
})();

