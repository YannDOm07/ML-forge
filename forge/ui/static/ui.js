"use strict";

const POLL_MS = 5000;
const SCREEN_TITLES = {
  portfolio: ["Tous les projets", "Où en est chaque projet, ce qui attend l’utilisateur."],
  overview: ["Vue d’ensemble", "État du projet et résultats essentiels."],
  competition: ["Compétition", "Fiche citée, pages lues et dépôts décrits."],
  campaign: ["Campagne", "Progression, budget et prochaines étapes."],
  experiments: ["Expériences", "Résultats calculés par ML Forge."],
  hypotheses: ["Hypothèses", "Pistes proposées, testées et écartées."],
  data: ["Données", "Provenance et profil agrégé, sans lignes du jeu de données."],
  validation: ["Validation", "Protocole, fiabilité et métriques hors pli."],
  deliverables: ["Livrables", "Fichiers produits et historique des soumissions."],
  lessons: ["Leçons", "Ce qui marche ou échoue dans ce contexte, et pourquoi."],
  memory: ["Mémoire", "Leçons mesurées et calibration des estimations."],
  journal: ["Journal", "Événements, accès réseau et actions en attente."]
};

const screen = document.querySelector("#screen");
const picker = document.querySelector("#project-picker");
const connection = document.querySelector("#connection-state");
const updated = document.querySelector("#updated-at");
const notice = document.querySelector("#app-notice");
const emptyState = document.querySelector("#empty-state");
let selectedProject = new URLSearchParams(location.search).get("project");
const requestedScreen = new URLSearchParams(location.search).get("screen");
let selectedScreen = SCREEN_TITLES[requestedScreen] ? requestedScreen
  : new URLSearchParams(location.search).get("project") ? "overview" : "portfolio";
let state = null;
let experimentSort = {key: "created_at", direction: "desc"};
let comparisonIds = [];

function node(tag, className, value, ...children) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (value instanceof Node) element.append(value);
  else if (value !== undefined && value !== null) element.textContent = String(value);
  children.forEach((child) => {
    if (child instanceof Node) element.append(child);
    else if (child !== null && child !== undefined) element.append(document.createTextNode(String(child)));
  });
  return element;
}

function format(value, digits = 4) {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "number") return Number.isFinite(value) ? value.toFixed(digits) : "—";
  if (Array.isArray(value)) return value.join(", ");
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function percent(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${(number * 100).toFixed(0)} %` : "—";
}

function bytes(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "—";
  if (number < 1024) return `${number} o`;
  if (number < 1024 ** 2) return `${(number / 1024).toFixed(1)} Ko`;
  if (number < 1024 ** 3) return `${(number / 1024 ** 2).toFixed(1)} Mo`;
  return `${(number / 1024 ** 3).toFixed(1)} Go`;
}

function date(value) {
  if (!value) return "—";
  const parsed = new Date(value);
  return Number.isNaN(parsed.valueOf()) ? String(value) : parsed.toLocaleString("fr-FR");
}

function badge(label, tone = "") {
  return node("span", `badge${tone ? ` ${tone}` : ""}`, label || "inconnu");
}

function panel(title, content, subtitle = "") {
  const wrapper = node("article", "panel");
  const heading = node("div", "panel-title");
  const text = node("div");
  text.append(node("h2", "", title));
  if (subtitle) text.append(node("p", "", subtitle));
  heading.append(text);
  wrapper.append(heading, content);
  return wrapper;
}

function table(headers, rows, emptyMessage = "Aucun élément à afficher.") {
  const wrap = node("div", "table-wrap");
  if (!rows.length) return node("div", "empty-inline", emptyMessage);
  const tableElement = node("table");
  const thead = node("thead");
  const headerRow = node("tr");
  headers.forEach((header) => {
    const th = node("th");
    if (header instanceof Node) th.append(header);
    else th.textContent = String(header);
    headerRow.append(th);
  });
  thead.append(headerRow);
  const tbody = node("tbody");
  rows.forEach((row) => {
    const tr = node("tr");
    row.forEach((cell) => {
      const td = node("td", cell && typeof cell === "object" && cell.wrap ? "wrap" : "");
      const content = cell && typeof cell === "object" && "content" in cell ? cell.content : cell;
      if (content instanceof Node) td.append(content);
      else td.textContent = content === null || content === undefined ? "—" : String(content);
      tr.append(td);
    });
    tbody.append(tr);
  });
  tableElement.append(thead, tbody);
  wrap.append(tableElement);
  return wrap;
}

function metricCard(label, value, detail = "") {
  const card = node("article", "metric-card");
  card.append(node("div", "metric-label", label), node("div", "metric-value", value));
  if (detail) card.append(node("div", "metric-detail", detail));
  return card;
}

function heading(title, description, side = null) {
  const wrapper = node("div", "page-heading");
  const copy = node("div");
  copy.append(node("p", "eyebrow", "ML FORGE · LABORATOIRE"));
  copy.append(node("h1", "", title), node("p", "", description));
  wrapper.append(copy);
  if (side) wrapper.append(side);
  return wrapper;
}

function definitionList(pairs) {
  const list = node("dl", "kv-list");
  pairs.forEach(([key, value]) => {
    list.append(node("dt", "", key), node("dd", "", value ?? "—"));
  });
  return list;
}

function progressBar(value, maximum, label) {
  const wrapper = node("div");
  const bar = node("progress", "progress");
  bar.max = maximum || 1;
  bar.value = Math.max(0, Math.min(Number(value) || 0, Number(maximum) || 0));
  const caption = node("div", "progress-caption");
  caption.append(node("span", "", label), node("span", "", `${value} / ${maximum || "—"}`));
  wrapper.append(bar, caption);
  return wrapper;
}

function gainChart(experiments) {
  const measured = experiments.filter((item) => Number.isFinite(item.decision?.delta));
  if (!measured.length) return node("div", "empty-inline", "Aucun gain mesuré pour le moment.");
  const recent = measured.slice(-12);
  const width = 720;
  const labelWidth = 82;
  const chartWidth = width - labelWidth - 18;
  const center = labelWidth + chartWidth / 2;
  const maximum = Math.max(...recent.map((item) => Math.abs(item.decision.delta)), 0.0001);
  const rowHeight = 28;
  const height = recent.length * rowHeight + 12;
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", "Gain mesuré par expérience, positif ou négatif par rapport à la référence");
  svg.classList.add("gain-chart");
  const baseline = document.createElementNS(svg.namespaceURI, "line");
  baseline.setAttribute("x1", String(center));
  baseline.setAttribute("x2", String(center));
  baseline.setAttribute("y1", "4");
  baseline.setAttribute("y2", String(height - 4));
  baseline.setAttribute("class", "gain-zero");
  svg.append(baseline);
  recent.forEach((item, index) => {
    const y = 7 + index * rowHeight;
    const delta = item.decision.delta;
    const barWidth = Math.abs(delta) / maximum * (chartWidth / 2 - 4);
    const label = document.createElementNS(svg.namespaceURI, "text");
    label.setAttribute("x", "0");
    label.setAttribute("y", String(y + 13));
    label.setAttribute("class", "gain-label");
    label.textContent = item.id;
    svg.append(label);
    const bar = document.createElementNS(svg.namespaceURI, "rect");
    bar.setAttribute("x", String(delta < 0 ? center - barWidth : center));
    bar.setAttribute("y", String(y));
    bar.setAttribute("width", String(Math.max(barWidth, 1)));
    bar.setAttribute("height", "15");
    bar.setAttribute("rx", "4");
    bar.setAttribute("class", delta < 0 ? "gain-negative" : "gain-positive");
    svg.append(bar);
    const value = document.createElementNS(svg.namespaceURI, "text");
    value.setAttribute("x", String(delta < 0 ? center - barWidth - 7 : center + barWidth + 7));
    value.setAttribute("y", String(y + 12));
    value.setAttribute("text-anchor", delta < 0 ? "end" : "start");
    value.setAttribute("class", "gain-value");
    value.textContent = format(delta, 4);
    svg.append(value);
  });
  return svg;
}

function experimentRows(items) {
  return items.map((item) => [
    item.id,
    item.method,
    {content: node("span", "wrap", item.hypothesis), wrap: true},
    badge(item.decision?.status || item.status, item.decision?.status === "accepted" ? "good" : ""),
    format(item.decision?.metric_mean),
    format(item.decision?.delta),
  ]);
}

function renderOverview(data) {
  const overview = data.overview;
  if (!overview) return;
  const project = overview.project;
  const subtitle = [overview.platform, overview.problem_type, overview.support_level].filter(Boolean).join(" · ");
  screen.append(heading(project.name, subtitle || data.focus.message,
    badge(overview.campaign?.status || "sans campagne", overview.campaign?.active ? "good" : "")));
  const counts = overview.experiment_counts || {};
  const metric = overview.metric || "métrique à définir";
  const bestDecision = overview.best?.decision || overview.best || {};
  const bestScore = bestDecision.metric_mean;
  const expTotal = counts.total ?? 0;
  screen.append(node("div", "grid metrics"));
  const metrics = screen.lastChild;
  metrics.append(
    metricCard("MEILLEUR SCORE", format(bestScore), `${metric} · ${overview.best?.id || ""} · Δ ${format(bestDecision.delta)}`),
    metricCard("EXPÉRIENCES", expTotal, `${overview.experiments_max} maximum par campagne`),
    metricCard("ACCEPTÉES", overview.accepted_count),
    metricCard("À VÉRIFIER", (overview.open_actions?.length || 0) + (overview.pending_approvals?.length || 0),
      "actions et accords en attente")
  );
  const callout = node("div", "callout");
  callout.append(node("span", "live-dot"));
  const message = node("div");
  message.append(node("strong", "", data.focus.phase ? `Prochaine étape · ${data.focus.phase}` : "Prochaine étape"));
  message.append(node("p", "", data.focus.message));
  callout.append(message);
  screen.append(callout);
  const columns = node("div", "grid two");
  const recent = overview.recent_experiments || [];
  columns.append(
    panel("Expériences récentes", table(["ID", "Méthode", "Hypothèse", "Décision", "Score", "Δ"],
      experimentRows(recent), "Aucune expérience enregistrée.")),
    panel("État de la campagne", definitionList([
      ["État", overview.campaign?.status || "inactive"],
      ["Pilote", overview.campaign?.driver || "—"],
      ["Fiabilité validation", overview.reliability === true ? "fiable" : overview.reliability === false ? "peu fiable" : "—"],
      ["Échéance", date(overview.deadline)],
    ]))
  );
  screen.append(columns);
  screen.append(panel("Gain mesuré par expérience", gainChart(data.experiments),
    "Δ par rapport à la référence, selon la métrique du protocole."));
  const actionItems = [
    ...(overview.open_actions || []).map((item) => `Action · ${item.message}`),
    ...(overview.pending_approvals || []).map((item) => `Accord requis · ${item.action} : ${item.target}`),
  ];
  if (actionItems.length) {
    const list = node("ul", "action-list");
    actionItems.forEach((message) => list.append(node("li", "", message)));
    screen.append(panel("À traiter", list));
  }
}

function renderCampaign(data) {
  const campaign = data.campaign;
  screen.append(heading("Campagne", data.focus.message, badge(campaign?.status || "inactive",
    campaign?.status === "running" ? "good" : "")));
  if (!campaign || campaign.status === "inactive") {
    screen.append(panel("Aucune campagne active", node("p", "muted",
      "Démarrez une campagne avec la commande forge campaign start depuis le terminal.")));
    return;
  }
  const used = campaign.experiments_used || 0;
  const max = campaign.experiments_max || 0;
  const cols = node("div", "grid two");
  cols.append(
    panel("Budget d’expériences", progressBar(used, max, "Expériences depuis le début")),
    panel("Budget de temps", progressBar(campaign.hours_elapsed, campaign.budget_hours,
      `${format(campaign.hours_remaining, 1)} h restantes`))
  );
  screen.append(cols);
  screen.append(panel(campaign.active ? "Détails de la campagne active" : "Dernière campagne", definitionList([
    ["Pilote", campaign.driver], ["Mode", campaign.mode], ["Démarrée", date(campaign.started_at)],
    ["Patience", `${campaign.patience} expériences sans gain accepté`], ["Raison / arrêt", campaign.reason || "—"],
  ])));
  if (campaign.active) {
    const phase = node("div", "callout");
    phase.append(node("span", "live-dot"));
    phase.append(node("p", "", data.focus.message));
    screen.append(phase);
  }
}

function renderExperiments(data) {
  screen.append(heading("Expériences", "Comparaison des résultats et des décisions calculées."));
  screen.append(renderComparison(data.experiments));
  const sortOptions = [
    ["id", "ID"], ["parent_id", "Parent"], ["method", "Méthode"], ["hypothesis", "Hypothèse"],
    ["status", "État"], ["metric_mean", "Score"], ["metric_std", "Écart-type"], ["delta", "Δ"],
  ];
  const headers = sortOptions.map(([key, label]) => {
    const button = node("button", "sort-button", `${label}${experimentSort.key === key ? (experimentSort.direction === "asc" ? " ↑" : " ↓") : ""}`);
    button.type = "button";
    button.dataset.sortKey = key;
    return button;
  });
  const sorted = [...data.experiments].sort((first, second) => {
    let left;
    let right;
    if (experimentSort.key === "metric_mean" || experimentSort.key === "metric_std" || experimentSort.key === "delta") {
      left = first.decision?.[experimentSort.key];
      right = second.decision?.[experimentSort.key];
    } else {
      left = first[experimentSort.key];
      right = second[experimentSort.key];
    }
    const order = typeof left === "number" && typeof right === "number"
      ? left - right : String(left ?? "").localeCompare(String(right ?? ""), "fr");
    return experimentSort.direction === "asc" ? order : -order;
  });
  const rows = sorted.map((item) => [
    item.id, item.parent_id || "—", item.method,
    {content: node("span", "wrap", item.hypothesis), wrap: true},
    badge(item.decision?.status || item.status, item.decision?.status === "accepted" ? "good" : ""),
    format(item.decision?.metric_mean), format(item.decision?.metric_std), format(item.decision?.delta),
  ]);
  screen.append(panel("Registre des expériences", table(
    headers, rows,
    "Les expériences apparaîtront ici après la baseline."
  )));
}

function renderComparison(experiments) {
  const comparable = experiments.filter((item) => item.decision);
  if (comparisonIds.length !== 2 || comparisonIds.some((id) => !comparable.some((item) => item.id === id))) {
    comparisonIds = comparable.length >= 2
      ? [comparable[comparable.length - 2].id, comparable[comparable.length - 1].id] : [];
  }
  const panelContent = node("div");
  if (comparable.length < 2) {
    panelContent.append(node("p", "muted", "Deux expériences évaluées sont nécessaires pour une comparaison descriptive."));
    return panel("Comparer deux expériences", panelContent);
  }
  const controls = node("div", "comparison-controls");
  const selectors = comparisonIds.map((selected, index) => {
    const label = node("label", "comparison-select");
    label.append(node("span", "", index === 0 ? "Première expérience" : "Seconde expérience"));
    const select = node("select");
    select.dataset.comparisonIndex = String(index);
    comparable.filter((experiment) => experiment.id !== comparisonIds[1 - index]).forEach((experiment) => {
      const option = node("option", "", `${experiment.id} · ${experiment.method}`);
      option.value = experiment.id;
      select.append(option);
    });
    select.value = comparable.some((item) => item.id === selected) ? selected : comparable[index].id;
    label.append(select);
    return select;
  });
  controls.append(...selectors.map((select) => select.parentElement));
  panelContent.append(controls);
  const [first, second] = comparisonIds.map((id) => comparable.find((item) => item.id === id));
  const sameProtocol = first && second && first.protocol_id === second.protocol_id;
  panelContent.append(node("p", sameProtocol ? "comparison-note" : "comparison-note warning",
    sameProtocol ? `Même protocole (${first.protocol_id}). Comparaison descriptive ; la décision reste celle calculée par ML Forge.`
      : "Protocoles différents : les scores ne sont pas directement comparables."));
  if (first && second) {
    panelContent.append(table(["Expérience", "Méthode", "Métrique", "Score moyen", "Écart-type", "Décision"], [
      [first.id, first.method, state.validation.task?.metric || "—", format(first.decision?.metric_mean),
        format(first.decision?.metric_std), first.decision?.status || "—"],
      [second.id, second.method, state.validation.task?.metric || "—", format(second.decision?.metric_mean),
        format(second.decision?.metric_std), second.decision?.status || "—"],
    ]));
  }
  return panel("Comparer deux expériences", panelContent);
}

function renderHypotheses(data) {
  screen.append(heading("Hypothèses", "Pistes classées par gain estimé × probabilité ÷ coût."));
  const lanes = [
    ["pending", "En attente"], ["tested", "Testées"], ["rejected", "Écartées"]
  ].map(([key, title]) => {
    const items = data.hypotheses.filter((item) => item.lane === key);
    const lane = node("section", "lane");
    const label = node("div", "lane-heading");
    label.append(node("span", "", title), badge(items.length));
    lane.append(label);
    if (!items.length) lane.append(node("div", "empty-inline", "Aucune hypothèse."));
    items.forEach((item) => {
      const card = node("article", "hypothesis-card");
      card.append(badge(item.id, key === "rejected" ? "bad" : key === "tested" ? "info" : ""));
      card.append(node("h3", "", item.hypothesis));
      card.append(node("p", "", `${item.method} · score priorité ${format(item.estimate.priority, 5)}`));
      card.append(node("p", "", `Gain ${format(item.estimate.expected_gain)} · probabilité ${percent(item.estimate.success_probability)} · coût ${item.estimate.cost_minutes ?? "—"} min`));
      lane.append(card);
    });
    return lane;
  });
  const grid = node("div", "lane-grid");
  lanes.forEach((lane) => grid.append(lane));
  screen.append(grid);
}


const MARKS = {marche: "✅", echoue: "❌", neutre: "➖"};
const KIND_MARKS = {erreur: "⚠", piege: "🪤", reussite: "⭐"};

function stepTone(entry) {
  return entry && entry.problem ? "warn" : "good";
}

function renderPortfolio(data) {
  const items = data.portfolio || [];
  screen.append(heading("Tous les projets", "Cliquez sur un projet pour ouvrir son détail."));
  const stats = node("div", "grid metrics");
  stats.append(
    metricCard("PROJETS", items.length),
    metricCard("AVEC LIVRABLES", items.filter((item) => item.deliverables).length),
    metricCard("EN ATTENTE DE VOUS", items.filter((item) => item.open_actions).length),
    metricCard("PROBLÈMES SIGNALÉS", items.reduce((total, item) => total + (item.problems || 0), 0), "entrées ⚠ des journaux")
  );
  screen.append(stats);
  const grid = node("div", "card-grid");
  items.forEach((item) => {
    const card = node("button", "panel project-card");
    card.type = "button";
    card.dataset.project = item.id;
    card.append(node("h2", "", item.name));
    const tags = node("div", "tag-list");
    [item.platform, item.problem_type, item.metric].filter((value) => value && value !== "—")
      .forEach((value) => tags.append(node("span", "tag", value)));
    if (item.campaign) tags.append(badge(item.campaign, item.campaign === "finished" ? "good" : item.campaign === "waiting_user" ? "warn" : ""));
    card.append(tags);
    card.append(definitionList([
      ["Meilleur score", item.best ? `${format(item.best.score)} (${item.best.experiment_id})` : "—"],
      ["Expériences", item.experiments ?? "—"],
      ["Dernière étape", item.last_step ? `${item.last_step.step} · ${item.last_step.at}` : "—"],
      ["Livrables", item.deliverables ? "présents" : "—"],
    ]));
    if (item.first_action) card.append(node("p", "callout", `À faire : ${item.first_action}`));
    if (item.missing) card.append(node("p", "callout", "Dossier du projet introuvable."));
    grid.append(card);
  });
  screen.append(items.length ? grid : node("div", "empty-inline", "Aucun projet."));
  const chronology = (data.lessons && data.lessons.chronology) || [];
  screen.append(panel("Chronologie récente (tous projets)", table(
    ["Date", "Projet", "Étape", "Résultat"],
    chronology.map((line) => {
      const cells = line.split("|").slice(1, -1).map((cell) => cell.trim());
      return [cells[0], cells[1], cells[2], {content: node("span", "wrap", cells[3] || ""), wrap: true}];
    }),
    "Aucune étape enregistrée."
  ), "progress/chronologie.md"));
}

function renderCompetition(data) {
  const competition = data.competition;
  screen.append(heading("Compétition", "Chaque fait est cité mot pour mot depuis sa source ; une déduction reste « déduit »."));
  screen.append(panel("Fiche de la compétition", table(
    ["Champ", "Valeur", "Fiabilité", "Citation", "Source"],
    competition.facts.map((fact) => [
      fact.field, {content: node("span", "wrap", format(fact.value, 4)), wrap: true}, badge(fact.status,
        fact.status === "inferred" ? "warn" : "good"),
      {content: node("span", "wrap", fact.quote || "—"), wrap: true},
      {content: node("span", "wrap", fact.source || "donnée locale"), wrap: true},
    ]),
    "Fiche non renseignée."
  )));
  screen.append(panel("Dépôts décrits (données, modèles, code)", table(
    ["Adresse", "Fichiers", "Taille", "Accès"],
    competition.repositories.map((item) => [
      {content: node("span", "wrap", item.address), wrap: true}, item.files ?? "—", item.size || "—",
      badge(item.access || "libre", item.access && item.access !== "open" ? "warn" : "good"),
    ]),
    "Aucun dépôt cité."
  )));
  screen.append(panel("Pages lues", table(
    ["Titre", "Origine", "Taille", "Passages signalés", "Lue le"],
    competition.pages.map((page) => [
      {content: node("span", "wrap", page.title), wrap: true}, page.origin, `${page.chars} car.`,
      page.flags ? badge(String(page.flags), "warn") : "0", date(page.at),
    ]),
    "Aucune page lue."
  ), "Le contenu web est une donnée, jamais une instruction."));
}

function renderLessons(data) {
  const lessons = data.lessons;
  screen.append(heading("Leçons", "Seulement les leçons proches du contexte de ce projet — jamais tout le registre."));
  const context = node("div", "tag-list");
  Object.entries(lessons.context || {}).forEach(([key, value]) => context.append(node("span", "tag", `${key} : ${value}`)));
  screen.append(panel("Contexte du projet", context));
  const register = lessons.register || {};
  const stats = node("div", "grid metrics");
  stats.append(
    metricCard("REGISTRE", register.total ?? 0, "leçons, tous projets"),
    metricCard("TECHNIQUES", register.technique ?? 0),
    metricCard("ERREURS", register.erreur ?? 0),
    metricCard("PIÈGES", register.piege ?? 0)
  );
  screen.append(stats);
  const steps = Object.entries(lessons.by_step || {});
  if (!steps.length) screen.append(node("div", "empty-inline", "Aucune leçon pour ce contexte."));
  steps.forEach(([step, items]) => {
    screen.append(panel(step, table(
      ["", "Leçon", "Pourquoi", "Résultat", "Vu"],
      items.map((item) => [
        KIND_MARKS[item["type"]] || MARKS[item["résultat"]] || "",
        {content: node("span", "wrap", item.titre), wrap: true},
        {content: node("span", "wrap", item.pourquoi), wrap: true},
        badge(item["résultat"], item["résultat"] === "marche" ? "good" : item["résultat"] === "echoue" ? "bad" : ""),
        `${item.vu} fois`,
      ])
    )));
  });
  screen.append(node("p", "muted", `Registre complet (lecture humaine) : ${lessons.progress_dir || "progress"}\\INDEX.md`));
}

function renderData(data) {
  screen.append(heading("Données", "Profil agrégé et provenance. Les lignes brutes du jeu ne sont jamais affichées."));
  const profile = data.data.profile;
  const stats = node("div", "grid metrics");
  stats.append(
    metricCard("LIGNES", profile.rows ?? "—"),
    metricCard("VARIABLES", profile.features ?? "—"),
    metricCard("CIBLE", profile.target ?? "—"),
    metricCard("PLANCHER D’ERREUR", format(profile.twins?.error_floor))
  );
  screen.append(stats);
  const summary = data.data.summary;
  if (summary) {
    const task = summary.task || {};
    screen.append(node("div", "grid two"));
    screen.lastChild.append(
      panel("Tâche formalisée", definitionList([
        ["Entraînement", task.train], ["Jeux à prédire", Object.entries(task.sets || {}).map(([k, v]) => `${k} : ${v}`).join(" · ") || "—"],
        ["Cible", task.target], ["Identifiant", task.id_column], ["Groupe des plis", task.group_column],
        ["Exclues (fuite)", (task.train_only_columns || []).join(", ") || "—"],
        ["Médias", (task.media_dirs || []).join(", ") || "—"],
      ])),
      panel("Fichiers enregistrés", definitionList([
        ["Total", `${summary.files} fichiers · ${bytes(summary.bytes)}`],
        ...summary.roles.map((role) => [role.role, `${role.files} fichiers · ${bytes(role.bytes)}`]),
        ["Types", summary.types.map((item) => `${item.type} ${item.files}`).join(" · ")],
      ]))
    );
    screen.append(panel("Dossiers les plus fournis", table(
      ["Dossier", "Fichiers", "Taille"],
      summary.folders.map((item) => [{content: node("span", "wrap", item.folder), wrap: true}, item.files, bytes(item.bytes)]),
      "Aucun fichier."
    ), summary.more_folders ? `et ${summary.more_folders} autre(s) dossier(s)` : ""));
  }
  screen.append(panel(`Sources enregistrées${data.data.sources_total > data.data.sources.length ? ` — ${data.data.sources.length} affichées sur ${data.data.sources_total}` : ""}`, table(
    ["Fichier", "Rôle", "Taille", "Licence", "Empreinte"],
    data.data.sources.map((item) => [
      item.name, item.role, bytes(item.size_bytes), item.license || "—", item.sha256 ? item.sha256.slice(0, 12) : "—",
    ]),
    "Aucune source de données enregistrée."
  )));
  const rows = profile.columns.map((column) => [
    column.name, column.kind, percent(column.missing_rate), column.unique ?? "—",
  ]);
  screen.append(panel("Colonnes — statistiques agrégées", table(
    ["Variable", "Type", "Valeurs manquantes", "Valeurs distinctes"], rows,
    "Le profil sera disponible après l’enregistrement des données."
  )));
  if (profile.target_stats?.class_count !== undefined) {
    screen.append(panel("Résumé agrégé de la cible", definitionList([
      ["Valeurs manquantes", profile.target_stats.missing],
      ["Classes distinctes", profile.target_stats.class_count],
      ["Ratio de déséquilibre", profile.target_stats.imbalance_ratio],
    ])));
  }
  const facts = data.data.facts.map((fact) => [fact.field, format(fact.value, 2), badge(fact.status)]);
  screen.append(panel("Fiche de compétition", table(["Champ", "Valeur", "Fiabilité"], facts, "Fiche non renseignée.")));
  if (profile.leak_hints?.length) {
    screen.append(panel("Indices de fuite à examiner", node("div", "tag-list",
      ...profile.leak_hints.map((item) => node("span", "tag", item)))));
  }
}

function renderValidation(data) {
  const validation = data.validation;
  screen.append(heading("Validation", "Protocole mesuré et régimes d’évaluation disponibles."));
  if (!validation.available) {
    screen.append(panel("Protocole non défini", node("p", "muted", data.focus.message)));
    return;
  }
  const task = validation.task || {};
  const reliability = validation.reliability || {};
  const stats = node("div", "grid metrics");
  stats.append(
    metricCard("TYPE DE PROBLÈME", task.problem_type || "—"),
    metricCard("MÉTRIQUE", task.metric || "—", task.target ? `cible : ${task.target}` : ""),
    metricCard("PLIS", validation.folds),
    metricCard("FIABILITÉ", reliability.level || reliability.status || "—")
  );
  screen.append(stats);
  screen.append(panel("Configuration", definitionList([
    ["Schéma", validation.scheme || "—"], ["Identifiant", validation.id], ["Notes", (validation.notes || []).join(" · ") || "—"],
    ["Fiabilité", JSON.stringify(reliability)],
  ])));
  screen.append(panel("Métriques par régime", table(
    ["Expérience", "Régime", "Métrique", "Mesures", "Moyenne", "Écart-type"],
    validation.metrics.map((item) => [
      item.experiment_id, item.scope, item.name, item.count, format(item.mean), format(item.std),
    ]),
    "Aucune mesure de validation pour le moment."
  )));
  screen.append(panel("Métriques prises en charge", node("div", "tag-list",
    ...validation.available_metrics.map((item) => node("span", "tag",
      `${item.name} · ${item.higher_is_better ? "plus haut meilleur" : "plus bas meilleur"}${item.needs_probabilities ? " · probabilités" : ""}`)))));
}

function renderDeliverables(data) {
  const deliverables = data.deliverables;
  screen.append(heading("Livrables", "État des fichiers générés et historique déclaré."));
  const present = deliverables.checklist.filter((item) => item.present).length;
  screen.append(node("div", "grid metrics"));
  screen.lastChild.append(
    metricCard("PRÉSENTS", present, `sur ${deliverables.checklist.length} livrables suivis`),
    metricCard("SOUMISSIONS DÉCLARÉES", deliverables.submissions.length),
    metricCard("CANDIDAT ROBUSTE", data.overview?.best?.experiment_id || data.overview?.best?.id || "—"),
    metricCard("EXIGENCES", deliverables.wanted.length)
  );
  screen.append(panel("Checklist", table(
    ["Livrable", "Fichier", "Exigé", "État"],
    deliverables.checklist.map((item) => [
      item.kind, item.name, item.required ? "Oui" : "Optionnel",
      badge(item.present ? "Présent" : item.required ? "Manquant" : "Non généré",
        item.present ? "good" : item.required ? "warn" : ""),
    ]),
    "Aucun livrable attendu."
  )));
  const files = data.files || {};
  screen.append(panel("Fichiers du dossier livrables", table(
    ["Fichier", "Taille"],
    (files.deliverables || []).map((item) => [item.path, bytes(item.bytes)]),
    "Aucun fichier généré."
  ), files.folder || ""));
  screen.append(node("div", "grid two"));
  screen.lastChild.append(
    panel("Notebooks", table(["Notebook", "Taille"], (files.notebooks || []).map((item) => [item.path, bytes(item.bytes)]),
      "Aucun notebook.")),
    panel("Tables dérivées", table(["Table", "Taille"], (files.derived || []).map((item) => [item.path, bytes(item.bytes)]),
      "Aucune table dérivée."))
  );
  screen.append(panel("Historique des soumissions déclarées", table(
    ["Expérience", "Date", "Score public", "Empreinte"],
    deliverables.submissions.map((item) => [
      item.experiment_id, date(item.submitted_at), format(item.public_score), item.file_sha256 || "—",
    ]),
    "Les dépôts sont enregistrés par l’utilisateur après soumission."
  )));
}

function renderMemory(data) {
  const memory = data.memory;
  screen.append(heading("Mémoire", "Connaissances issues des décisions mesurées ; elles restent des pistes à vérifier."));
  screen.append(panel("Leçons générales", table(
    ["Famille", "Contexte", "Conclusion mesurée", "Appuis", "Contradictions", "Confiance"],
    memory.knowledge.map((item) => [
      item.family, item.context_key, {content: node("span", "wrap", item.statement), wrap: true},
      item.n_supporting, item.n_contradicting, percent(item.confidence),
    ]),
    "La mémoire se remplit après les expériences acceptées ou rejetées."
  )));
  screen.append(panel("Calibration des estimations", table(
    ["Expérience", "Famille", "Gain prévu", "Gain observé", "Date"],
    memory.calibration.map((item) => [
      item.experiment_id, item.family, format(item.expected_gain), format(item.observed_gain), date(item.recorded_at),
    ]),
    "Aucune calibration pour ce projet."
  )));
  screen.append(panel("Preuves liées au projet", table(
    ["Expérience", "Famille", "Résultat Δ", "Soutient", "Date"],
    memory.evidence.map((item) => [
      item.experiment_id, item.family, format(item.delta), item.supports ? "Oui" : "Non", date(item.recorded_at),
    ]),
    "Aucune preuve enregistrée."
  )));
}

function renderJournal(data) {
  const journal = data.journal;
  screen.append(heading("Journal", "Traçabilité des événements du projet et des actions à traiter."));
  const entries = data.project_journal || [];
  screen.append(panel("Journal du projet (JOURNAL.md)", table(
    ["Date", "Étape", "Résultat"],
    entries.map((entry) => [entry.at, badge(entry.step, stepTone(entry)), {content: node("span", "wrap", entry.text), wrap: true}]),
    "Aucune entrée."
  ), "Plus récent d’abord ; ⚠ marque un problème."));
  screen.append(panel("Événements récents", table(
    ["Date", "Type", "Expérience", "Message"],
    journal.events.map((item) => [
      date(item.at), item.kind, item.experiment_id || "—", {content: node("span", "wrap", item.message), wrap: true},
    ]),
    "Aucun événement enregistré."
  )));
  screen.append(panel("Accès réseau", table(
    ["Date", "Méthode", "URL", "Résultat", "Octets", "Erreur"],
    journal.network.map((item) => [
      date(item.at), item.method, {content: node("span", "wrap", item.url), wrap: true},
      item.status ?? "—", bytes(item.bytes), item.error || "—",
    ]),
    "Aucun accès réseau enregistré."
  )));
  const actionRows = journal.actions.map((item) => [
    item.kind, {content: node("span", "wrap", item.message), wrap: true}, badge(item.status),
    date(item.created_at),
  ]);
  const approvals = journal.approvals.map((item) => [
    item.action, item.target, badge(item.status, item.status === "pending" ? "warn" : ""),
    date(item.created_at),
  ]);
  screen.append(node("div", "grid two"));
  screen.lastChild.append(
    panel("Actions utilisateur", table(["Type", "Action", "État", "Créée"], actionRows, "Aucune action.")),
    panel("Accords", table(["Action", "Cible", "État", "Créé"], approvals, "Aucun accord requis."))
  );
  screen.append(panel("Conformité", table(
    ["Date", "Élément", "Justification", "Source"],
    journal.compliance.map((item) => [
      date(item.at), item.item, {content: node("span", "wrap", item.justification), wrap: true}, item.source || "—",
    ]),
    "Aucune entrée au registre de conformité."
  )));
}

const RENDERERS = {
  portfolio: renderPortfolio,
  overview: renderOverview,
  competition: renderCompetition,
  campaign: renderCampaign,
  experiments: renderExperiments,
  hypotheses: renderHypotheses,
  data: renderData,
  validation: renderValidation,
  deliverables: renderDeliverables,
  lessons: renderLessons,
  memory: renderMemory,
  journal: renderJournal,
};

function setScreen(name) {
  if (!SCREEN_TITLES[name]) return;
  selectedScreen = name;
  document.querySelectorAll(".nav-link").forEach((button) => {
    const active = button.dataset.screen === name;
    button.classList.toggle("active", active);
    button.setAttribute("aria-current", active ? "page" : "false");
  });
  if (state) render();
}

function updateProjectPicker(projects, selected) {
  const current = picker.value;
  picker.replaceChildren();
  projects.forEach((project) => {
    const option = document.createElement("option");
    option.value = project.id;
    option.textContent = project.name;
    picker.append(option);
  });
  const value = projects.some((project) => project.id === selected) ? selected :
    projects.some((project) => project.id === current) ? current : projects[0]?.id;
  if (value) picker.value = value;
  picker.disabled = projects.length === 0;
}

function render() {
  if (!state) return;
  screen.replaceChildren();
  screen.setAttribute("aria-busy", "false");
  const [title, description] = SCREEN_TITLES[selectedScreen];
  document.title = `${title} · ML Forge`;
  RENDERERS[selectedScreen](state);
  updated.textContent = `Mis à jour ${new Date(state.generated_at).toLocaleTimeString("fr-FR")}`;
  document.querySelector("#selected-project-label").textContent =
    state.selected_project ? `· ${state.selected_project}` : "";
  emptyState.hidden = state.projects.length > 0;
  screen.hidden = state.projects.length === 0;
  notice.hidden = true;
}

async function refresh() {
  try {
    const query = selectedProject ? `?project_id=${encodeURIComponent(selectedProject)}` : "";
    const response = await fetch(`/api/state${query}`, {cache: "no-store"});
    if (!response.ok) throw new Error(`Réponse HTTP ${response.status}`);
    state = await response.json();
    selectedProject = state.selected_project;
    updateProjectPicker(state.projects, selectedProject);
    connection.classList.remove("offline");
    connection.querySelector("span:last-of-type").textContent = "Connecté";
    const editing = screen.contains(document.activeElement)
      && document.activeElement.matches("select, input, textarea");
    if (!editing) render();
  } catch (error) {
    connection.classList.add("offline");
    connection.querySelector("span:last-of-type").textContent = "Déconnecté";
    notice.hidden = false;
    notice.textContent = `Impossible de lire ML Forge : ${error.message}`;
    screen.setAttribute("aria-busy", "false");
  }
}

document.querySelector("#navigation").addEventListener("click", (event) => {
  const button = event.target.closest("[data-screen]");
  if (button) setScreen(button.dataset.screen);
});
function selectProject(id, screenName) {
  selectedProject = id || null;
  const url = new URL(location.href);
  if (selectedProject) url.searchParams.set("project", selectedProject);
  else url.searchParams.delete("project");
  history.replaceState(null, "", url);
  if (screenName) setScreen(screenName);
  refresh();
}

screen.addEventListener("click", (event) => {
  const card = event.target.closest("[data-project]");
  if (card) {
    selectProject(card.dataset.project, "overview");
    return;
  }
  const button = event.target.closest("[data-sort-key]");
  if (button) {
    const key = button.dataset.sortKey;
    experimentSort = {
      key,
      direction: experimentSort.key === key && experimentSort.direction === "asc" ? "desc" : "asc",
    };
    render();
  }
});
screen.addEventListener("change", (event) => {
  const select = event.target.closest("[data-comparison-index]");
  if (!select) return;
  comparisonIds[Number(select.dataset.comparisonIndex)] = select.value;
  render();
});
picker.addEventListener("change", () => selectProject(picker.value));
setScreen(selectedScreen);
refresh();
window.setInterval(refresh, POLL_MS);
