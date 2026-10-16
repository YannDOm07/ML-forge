const form = document.querySelector('#link-form');
const message = document.querySelector('#form-message');
const projectList = document.querySelector('#project-list');
const projectCount = document.querySelector('#project-count');
const detail = document.querySelector('#detail-content');
const preview = document.querySelector('#data-preview');
const dataState = document.querySelector('#data-state');
let selectedProject = null;

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (character) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[character]));
}

async function api(url, options = {}) {
  const response = await fetch(url, {headers: {'Content-Type': 'application/json'}, ...options});
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || 'Une erreur est survenue.');
  return result;
}

function renderMetrics(metrics = {}) {
  const items = Object.entries(metrics).filter(([, value]) => typeof value === 'number');
  if (!items.length) return '';
  return `<div class="metrics">${items.map(([name, value]) => `<div class="metric"><span>${escapeHtml(name.replaceAll('_', ' ').toUpperCase())}</span><strong>${Number(value).toFixed(3)}</strong></div>`).join('')}</div>`;
}

function renderSummary(summary, filename, sourceUrl) {
  const columns = [...(summary.numeric_columns || []), ...(summary.categorical_columns || []), ...(summary.datetime_columns || []), ...(summary.text_columns || [])];
  preview.classList.remove('empty-preview');
  preview.innerHTML = `<div class="preview-stats"><div class="stat"><strong>${summary.rows}</strong><span>LIGNES</span></div><div class="stat"><strong>${summary.columns}</strong><span>COLONNES</span></div><div class="stat"><strong>${summary.duplicate_rows}</strong><span>DOUBLONS</span></div></div><div class="column-list">${columns.map((column) => `<span class="column-chip ${column === summary.target_column ? 'target' : ''}">${escapeHtml(column)}${column === summary.target_column ? ' · cible' : ''}</span>`).join('')}</div><p class="field-help">Fichier : ${escapeHtml(filename)}<br>Source : ${escapeHtml(sourceUrl)}</p>`;
  dataState.textContent = summary.target_column ? 'ANALYSÉ' : 'CIBLE À CONFIRMER';
  dataState.classList.toggle('ready', Boolean(summary.target_column));
}

function renderImport(result) {
  selectedProject = result.project_id;
  renderSummary(result.summary, result.filename, result.dataset_url);
  const experiment = result.experiment;
  detail.innerHTML = `<div class="project-summary"><strong>${escapeHtml(result.project_name)}</strong><br>Projet créé automatiquement<br>Cible détectée : <strong>${escapeHtml(result.target_column || 'non identifiée')}</strong></div>${experiment ? `<div class="project-summary" style="margin-top:12px">Baseline · ${escapeHtml(experiment.model)} · ${escapeHtml(experiment.task_type)}<br>${experiment.n_samples} lignes · ${Number(experiment.duration_seconds).toFixed(2)} s</div>${renderMetrics(experiment.metrics)}` : ''}${result.page_description ? `<p class="page-description">${escapeHtml(result.page_description.slice(0, 480))}</p>` : ''}${result.warning ? `<p class="message">${escapeHtml(result.warning)}</p>` : ''}${result.target_column && !experiment ? '<button class="run-button" id="run-experiment" type="button">Lancer la baseline <span aria-hidden="true">↗</span></button>' : ''}`;
  const runButton = document.querySelector('#run-experiment');
  if (runButton) runButton.addEventListener('click', () => runExperiment(result.project_id));
  loadProjects();
}

async function loadProjects() {
  try {
    const {projects} = await api('/api/projects');
    projectCount.textContent = projects.length;
    projectList.innerHTML = projects.length ? projects.map((project) => `<button class="project-entry ${selectedProject === project.id ? 'selected' : ''}" data-project="${escapeHtml(project.id)}" title="${escapeHtml(project.name)}">${escapeHtml(project.name)}</button>`).join('') : '<p class="quiet">Aucun projet pour le moment.</p>';
    projectList.querySelectorAll('[data-project]').forEach((button) => button.addEventListener('click', () => showProject(projects.find((project) => project.id === button.dataset.project))));
    if (selectedProject) {
      const active = projects.find((project) => project.id === selectedProject);
      if (active) showProject(active);
    }
  } catch (error) { projectList.innerHTML = `<p class="quiet">${escapeHtml(error.message)}</p>`; }
}

function showProject(project) {
  if (!project) return;
  selectedProject = project.id;
  projectList.querySelectorAll('.project-entry').forEach((button) => button.classList.toggle('selected', button.dataset.project === selectedProject));
  const latest = (project.experiments || []).find((experiment) => experiment.status === 'completed');
  detail.innerHTML = `<div class="project-summary"><strong>${escapeHtml(project.name)}</strong><br>${escapeHtml(project.description || 'Projet local')}<br>Cible : <strong>${escapeHtml(project.target_column || 'non identifiée')}</strong></div>${latest ? `<div class="project-summary" style="margin-top:12px">Dernière expérience · ${escapeHtml(latest.method || 'Baseline')}<br>Statut : terminée</div>${renderMetrics(latest.metrics || {})}` : '<p class="message">Projet conservé. Collez un lien avec un dataset exploitable pour démarrer son évaluation.</p>'}`;
}

async function runExperiment(projectId) {
  detail.innerHTML = '<p class="project-summary loading">Entraînement de la baseline locale...</p>';
  try {
    const {result} = await api(`/api/projects/${encodeURIComponent(projectId)}/run`, {method: 'POST', body: JSON.stringify({hypothesis: 'Établir une baseline reproductible pour le projet importé.'})});
    detail.innerHTML = `<div class="project-summary"><strong>${escapeHtml(result.model)}</strong> · ${escapeHtml(result.task_type)}<br>${result.n_samples} lignes · ${result.n_features} variables · ${Number(result.duration_seconds).toFixed(2)} s</div>${renderMetrics(result.metrics)}<p class="message">Expérience enregistrée dans le projet.</p>`;
    loadProjects();
  } catch (error) { detail.innerHTML = `<p class="message error">${escapeHtml(error.message)}</p>`; }
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const button = form.querySelector('button[type="submit"]');
  button.disabled = true;
  button.querySelector('span').textContent = 'Lecture de la page et recherche des données...';
  message.className = 'message';
  message.textContent = 'Le clic autorise la récupération du lien public et des fichiers de données liés.';
  dataState.textContent = 'ANALYSE EN COURS';
  try {
    const result = await api('/api/import-link', {method: 'POST', body: JSON.stringify({url: form.elements.url.value.trim()})});
    message.textContent = result.experiment ? 'Projet créé, dataset analysé et baseline exécutée.' : 'Projet créé automatiquement. Consultez le diagnostic et les éventuelles étapes restantes.';
    renderImport(result);
  } catch (error) {
    message.textContent = error.message;
    message.className = 'message error';
    dataState.textContent = 'LIEN NON IMPORTÉ';
    dataState.classList.remove('ready');
  } finally {
    button.disabled = false;
    button.querySelector('span').textContent = 'Analyser le lien et démarrer';
  }
});

loadProjects();