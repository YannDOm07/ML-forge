# Note d'adaptation — ML Forge UI

## 1. Arborescence du dépôt et conventions

- **Point d'entrée CLI** : `forge/__main__.py` → `forge.cli:main` (argparse, sous-commandes)
- **Configuration** : `forge.config:load_config()` → `MLFORGE_CONFIG` ou `config/mlforge.toml`
- **Workspace** : `forge.workspace:Workspace` (root, data_dir, projects_dir, logs_dir, global_db)
- **Projets** : chaque projet a son dossier `projects/<id>/` avec `project.db` (WAL) et `manifest.json`
- **Migrations** : versionnées dans `forge.schema:PROJECT_MIGRATIONS` / `GLOBAL_MIGRATIONS`
- **Tests** : `pytest` dans `tests/` ; conventions : fixtures, nommage `test_*.py`
- **Style code** : type hints obligatoires, docstrings françaises, pas de commentaires inutiles, `snake_case` pour fonctions/variables

## 2. Schéma réel des bases

### forge.db (global)
| Table | Colonnes |
|-------|----------|
| `projects` | `id`, `name`, `competition_url`, `status` (active/archived/deleted), `created_at`, `updated_at` |
| `configuration_history` | `id`, `config_hash`, `content`, `recorded_at` |
| `network_log` | `id`, `project_id`, `method`, `url`, `purpose`, `status`, `bytes`, `error`, `at` |
| `knowledge_items` | `id`, `family`, `context_key`, `context_json`, `statement`, `n_supporting`, `n_contradicting`, `n_projects`, `mean_delta`, `confidence`, `updated_at` |
| `knowledge_evidence` | `id`, `knowledge_id`, `project_id`, `experiment_id`, `delta`, `supports`, `recorded_at` |
| `estimate_calibration` | `id`, `project_id`, `experiment_id`, `family`, `expected_gain`, `observed_gain`, `recorded_at` |

### project.db (par projet)
| Table | Colonnes |
|-------|----------|
| `data_sources` | `id`, `role` (official/external/inbox), `path`, `sha256`, `size_bytes`, `mtime_ns`, `origin_url`, `license`, `registered_at` |
| `experiments` | `id`, `parent_id`, `question`, `hypothesis`, `method`, `protocol_id`, `card_json`, `code_hash`, `data_hash`, `status` (registered/running/completed/failed/interrupted), `runner_pid`, `created_at`, `started_at`, `finished_at`, `runner_created` |
| `experiment_metrics` | `experiment_id`, `name`, `scope` (fold/seed/simulation), `idx`, `value` |
| `execution_events` | `id`, `experiment_id`, `kind`, `message`, `at` |
| `decisions` | `id`, `action`, `target`, `details_json`, `outcome` (approved/refused), `decided_by`, `at` |
| `protocols` | `id`, `spec_json`, `folds_json`, `reliability_json`, `created_at` |
| `data_profiles` | `id`, `data_hash`, `profile_json`, `created_at` |
| `experiment_code` | `code_hash`, `content` |
| `experiment_decisions` | `experiment_id`, `reference_id`, `status` (baseline/accepted/rejected/not_significant), `metric_mean`, `metric_std`, `delta`, `noise`, `rule`, `computed_at` |
| `web_pages` | `id`, `url`, `origin` (web/api/user/data), `status`, `title`, `text`, `links_json`, `flags_json`, `content_hash`, `fetched_at` |
| `competition_facts` | `id`, `field`, `value_json`, `status` (found/inferred/user), `source_url`, `quote`, `recorded_by`, `recorded_at` |
| `compliance_log` | `id`, `item`, `justification`, `source`, `at` |
| `user_actions` | `id`, `kind`, `message`, `status` (open/done), `created_at`, `resolved_at` |
| `campaigns` | `id`, `mode` (interactive/unattended), `driver`, `status` (running/paused/waiting_user/stopped/finished), `budget_hours`, `max_experiments`, `started_at`, `ended_at`, `reason`, `runner_pid`, `runner_created` |
| `approval_requests` | `id`, `action`, `target`, `details_json`, `status` (pending/approved/refused/used), `created_at`, `decided_at` |
| `submissions` | `id`, `experiment_id`, `file_sha256`, `submitted_at`, `public_score`, `note` |

## 3. Projet d'exemple / données disponibles

Aucun projet n'existe encore dans `C:\MLForge\`. La première utilisation créera un projet via `forge create`.

Types de tâches supportés (`forge.task.PROBLEM_TYPES`) : `binary`, `multiclass`, `regression`
Métriques (`forge.task.METRICS`) : 12 métriques avec `higher_is_better` et `needs_probabilities`
Niveaux de prise en charge : `validated` (problème connu), `assisted` (partiel), `experimental` (inconnu)

## 4. Table de correspondance — Écran → Source

| Écran / Élément | Table / Colonne / Fonction | Statut |
|-----------------|----------------------------|--------|
| **01 Vue d'ensemble** |
| Surtitre « plateforme · type · niveau » | `competition_facts` (platform, problem_type) + `_support_level()` | ✅ Disponible |
| Pastille état campagne | `campaigns.status` | ✅ Disponible |
| Meilleur score, gain vs baseline | `experiment_decisions.metric_mean` + `current_best()` | ✅ Disponible |
| Expériences n/max, acceptées/rejetées | `experiments` + `experiment_decisions.status` + config `max_experiments` | ✅ Disponible |
| Fiabilité validation + écart-type | `protocols.reliability_json` | ✅ Disponible |
| Échéance + soumissions jour | `competition_facts.deadline` + `submissions` | ✅ Disponible |
| Graphique gain par expérience | `experiment_metrics` + `experiment_decisions.delta` | ✅ Disponible |
| Livrables (checklist) | `competition_facts.deliverables` + `artifacts` (pas de table artifacts) | ⚠️ **[manquant]** table `artifacts` |
| Dernières expériences (5) | `experiments` + `experiment_decisions` (ORDER BY created_at DESC LIMIT 5) | ✅ Disponible |
| Hypothèses en attente + priorité | `experiments` (status pending/running) + estimation dans `card_json` | ✅ Disponible |
| Note d'autorisation | `user_actions` (status=open) + `approval_requests` (status=pending) | ✅ Disponible |
| **02 Campagne** |
| Phase courante (étapes 1-7) | `campaign_summary()['étape']` | ✅ Disponible |
| Budgets temps/expériences/sans-gain | `campaigns.budget_hours`, `max_experiments`, patience | ✅ Disponible |
| Conditions arrêt | `stop_reason()` | ✅ Disponible |
| Derniers événements | `execution_events` + `campaigns` | ✅ Disponible |
| Prochain jalon | `finalize_hours_before_deadline` + `deadline` | ✅ Disponible |
| **03 Expériences** |
| Tableau triable toutes expériences | `experiments` JOIN `experiment_decisions` | ✅ Disponible |
| Fiche expérience (A.2) | `experiments.card_json` + métriques + décisions | ✅ Disponible |
| Comparaison 2 expériences | `compare()` dans `forge.runner` (même protocole) | ✅ Disponible |
| **04 Hypothèses** |
| Colonnes En attente/Testées/Rejetées | `experiments.status` + `card_json.estimate` | ✅ Disponible |
| Score priorité (gain×prob÷coût) | calculé depuis `card_json.estimate` | ✅ Disponible |
| Biais calibration | `estimate_calibration` | ✅ Disponible |
| **05 Données** |
| Fiche compétition + source/URL | `competition_facts` + `web_pages` | ✅ Disponible |
| Sources données (provenance, empreinte, taille, licence) | `data_sources` | ✅ Disponible |
| Profil agrégé | `data_profiles.profile_json` | ✅ Disponible |
| Jumeaux contradictoires + plancher erreur | `profile_json.twins` | ✅ Disponible |
| Risques fuite | `profile_json.leak_hints` | ✅ Disponible |
| **06 Validation** |
| Schéma validation | `protocols.spec_json` + `folds_json` | ✅ Disponible |
| Variance plis/seeds | `experiment_metrics` (scope=fold/seed) | ✅ Disponible |
| Corrélation découpage officiel | non stocké | ⚠️ **[manquant]** |
| État simulateur test | `experiment_metrics` scope=simulation | ✅ Disponible |
| Régime acceptation | `protocols.reliability_json` → simple/double | ✅ Disponible |
| **07 Livrables** |
| Checklist complète | `competition_facts.deliverables` + `artifacts` | ⚠️ **[manquant]** table `artifacts` |
| Vérification fichier soumission | `submissions.file_sha256` | ✅ Disponible |
| Plan soumission | `submission_plan()` dans `forge.deliverables` | ✅ Disponible |
| Candidat robuste / ambitieux | `deliverables.finalize()` | ✅ Disponible |
| Historique soumissions | `submissions` | ✅ Disponible |
| **08 Mémoire** |
| Leçons + preuves | `knowledge_items` + `knowledge_evidence` | ✅ Disponible |
| Biais calibration agent | `estimate_calibration` | ✅ Disponible |
| Hypothèses rejetées contexte comparable | `memory_digest()` | ✅ Disponible |
| **09 Journal** |
| Événements exécution | `execution_events` | ✅ Disponible |
| Journal réseau | `network_log` | ✅ Disponible |
| Registre conformité | `compliance_log` | ✅ Disponible |
| Historique config | `configuration_history` | ✅ Disponible |

## 5. Éléments **[manquants]** et propositions

| Élément | Proposition |
|---------|-------------|
| Table `artifacts` (livrables générés) | Ajouter migration v5 : `CREATE TABLE artifacts (id INTEGER PRIMARY KEY, experiment_id TEXT, kind TEXT, path TEXT, sha256 TEXT, created_at TEXT)` — utilisé par `deliverables.finalize()` |
| Corrélation découpage officiel | Calcul à la volée : comparer `experiment_metrics` (fold) avec scores officiels si disponibles |
| Fichiers de soumission générés | `deliverables.finalize()` écrit dans `project.path/submissions/` — lister via `Path.glob()` |

## 6. Conventions du dépôt à respecter

- Requêtes SQL nommées dans `queries.py` (une fonction par requête, typée `sqlite3.Row` → `dict`)
- `viewmodel.py` : fonctions pures, pas de logique d'acceptation, adaptation déterministe (§7.2)
- `server.py` : `http.server` / `asyncio` uniquement, binding `127.0.0.1`, CSP strict
- CSS : **copie exacte de l'annexe A** dans `forge/ui/static/ui.css`
- JS : natif, pas de build, polling 5s via `fetch`
- Tests : `tests/ui/test_queries.py`, `tests/ui/test_viewmodel.py`

La note disponible ne contient pas le texte de l'annexe A ni les détails des sections §7.2 et §7.4. Le CSS doit donc être considéré comme une implémentation responsive provisoire, et non comme une copie conforme de l'annexe. Ces spécifications restent à intégrer lorsqu'elles seront fournies.

## 7. Points **[à vérifier]**

- [x] Port par défaut 8765 (pas 8080)
- [x] Commande `forge ui [--project <id>] [--port <n>]`
- [x] SQLite en lecture seule : `file:path?mode=ro` + `uri=True`
- [x] Binding `127.0.0.1`, vérification du client local et du header `Host`
- [x] CSP : `default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'`
- [x] Contenu externe rendu comme texte, sans HTML actif
- [x] Pas de lignes brutes de dataset dans les réponses JSON ; faits tirés de données locales masqués
- [x] Responsive < 900px : sidebar horizontale scrollable, indicateurs 2×2, tableaux avec scroll interne

## 8. Résumé

L'adaptation est **principalement du câblage** : les données existent déjà dans les tables SQLite et les fonctions Python (`campaign_summary`, `competition_card`, `decisions_by_experiment`, `memory_digest`, `submission_plan`, `finalize`). Le travail consiste à :
1. Exposer ces données via des routes JSON filtrées (`queries.py` + `viewmodel.py`)
2. Rendre l'UI avec le CSS exact de l'annexe A
3. Implémenter les 9 écrans avec leurs états vides (§7.4)
4. Gérer l'adaptation selon le type de tâche / niveau / phase (§7.2)