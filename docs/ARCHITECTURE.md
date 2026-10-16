# ML Forge — Architecture (version 1.0)

Référence des exigences : [`prompt_v2.md`](../prompt_v2.md). Ce document décrit **comment** elles sont réalisées et ce qui reste à vérifier.

## 1. Vue d'ensemble

```
            utilisateur
                │  lien de compétition, accords, dépôts inbox/
                ▼
┌──────────────────────────┐   MCP (stdio)   ┌──────────────────────────────────────┐
│ opencode + LLM (distant) │ ──────────────► │ ML Forge (Python, local)             │
│ raisonne, propose,       │ ◄────────────── │ méthode · état · garde-fous           │
│ écrit le code            │  sorties filtrées│ web_read/download · bac à sable       │
└──────────────────────────┘                 │ évaluation · mémoire · livrables      │
          ▲                                  └───────────────┬──────────────────────┘
          │ opencode run --attach (cycles)                    │
          └──────────── CLI `python -m forge campaign` ───────┤
                                                              ▼
                                   C:\MLForge\  forge.db · projects/<id>/project.db · data/ · backups/
```

- **ML Forge ne dépend pas d'opencode** : la CLI fonctionne seule (consultation, sauvegarde, export).
- **opencode ne contourne pas ML Forge** : `opencode.json` (généré par `setup-opencode`) refuse `bash`, `edit`, `webfetch`, `task`, `external_directory` et la lecture de `kaggle.json` ; seule la recherche web reste permise. Lectures de pages et téléchargements passent par ML Forge.

## 2. Modules et phase d'introduction

| Module | Responsabilité | Phase |
|---|---|---|
| `forge/config.py` | Lecture, validation et génération de la configuration TOML ; refus des racines synchronisées | 2 |
| `forge/db.py` | Connexions SQLite (WAL), transactions, migrations versionnées, instantanés | 2 |
| `forge/schema.py` | Migrations des bases globale et projet | 2+ |
| `forge/workspace.py` | Racine des données, base globale, historique de configuration, espace disque | 2 |
| `forge/projects.py` | Création, liste, ouverture, archivage, suppression contrôlée | 2 |
| `forge/guardrails.py` | Les 5 actions soumises à accord, journalisation des décisions | 2 |
| `forge/experiments.py` | Registre des expériences (fiche A.2), statuts, métriques, reprise après arrêt brutal | 2 |
| `forge/datasources.py` | Empreintes des données, détection de fichier manquant ou modifié | 2 |
| `forge/backup.py` | Sauvegarde et restauration des projets et de la base globale | 2 |
| `forge/system.py` | Détection du matériel, processus vivant, avertissements de configuration | 2 |
| `forge/cli.py` | Interface en ligne de commande | 2+ |
| `forge/sandbox.py`, `forge/sandbox_boot.py` | Exécution isolée : sous-processus unique (Job Object), audit hooks, environnement sans secrets | 3 |
| `forge/task.py` | Définition d'une tâche supervisée et des métriques disponibles | 3 |
| `forge/data_engine.py` | Profil, jumeaux contradictoires, sensibilité de la métrique, fuites | 3 |
| `forge/evaluation.py` | Métriques, plis, fiabilité de la validation, critère d'acceptation | 3 |
| `forge/tabular.py` | Contrat des scripts d'expérience et modèles intégrés (exécuté dans le bac à sable) | 3 |
| `forge/runner.py` | Protocole, profil, prérequis, exécution, score et décision calculés, conservation | 3 |
| `forge/report.py`, `forge/submission.py`, `forge/demo.py` | Rapport en place, soumission vérifiée, démonstration reproductible | 3 |
| `forge/web.py` | Accès réseau journalisé et borné, HTML → texte, détection d'injection, contenu encadré | 4 |
| `forge/intake.py` | Plateforme, exploration des pages, faits cités et vérifiés, fiche A.1, questions | 4 |
| `forge/kaggle.py` | Client minimal de l'API Kaggle v1 (jeton jamais journalisé ni transmis lors d'une redirection) | 4 |
| `forge/acquisition.py` | Données officielles (Kaggle, liens publics, inbox), ressources externes avec fiche d'acceptation | 4 |
| `forge/actions.py` | Actions attendues de l'utilisateur, jusqu'à leur résolution | 4 |
| `forge/mcp_server.py` | Serveur MCP (JSON-RPC sur stdio, bibliothèque standard), sorties bornées, aucune ligne brute sans autorisation | 5 |
| `forge/campaign.py` | Étapes du lien aux livrables, cycles, résumé `campaign_next`, conditions d'arrêt, pilotes intégré et opencode, pause et reprise | 5 |
| `forge/approvals.py` | Accords donnés par l'utilisateur en CLI, jamais par le LLM | 5 |
| `forge/opencode_setup.py`, `AGENTS.md` | Configuration opencode restrictive ; instructions de méthode de l'agent | 5 |
| `forge/knowledge.py` | Mémoire générale mesurée (familles × contextes, preuves), calibration des estimations, classement des hypothèses | 6 |
| `forge/deliverables.py` | Stabilité multi-seed, soumission, solution autonome, notebook, rapport technique, liste de contrôle, plan de soumission | 6 |
| (dans `evaluation.py`, `tabular.py`, `runner.py`) | Simulateur des conditions du test ; étape de décision hors pli ; registre des hypothèses rejetées ; signalement des scores suspects | 6 |
| (dans `evaluation.py`, `tabular.py`) | Validation chronologique ; variables de dates ; texte (TF-IDF, `text_linear`) ; boosters s'ils sont installés | 7 |

Un module n'est créé qu'à la phase où il est implémenté et testé.

## 3. Schéma de données cible

Les tables sont ajoutées par migration à la phase qui les utilise (`PRAGMA user_version`).

**Base globale `forge.db`**

| Table | Contenu | Phase |
|---|---|---|
| `projects` | id, nom, lien, statut (active/archived/deleted), dates | 2 |
| `configuration_history` | empreinte et contenu de chaque configuration utilisée | 2 |
| `network_log` | URL, motif, taille, statut de chaque accès réseau de ML Forge | 4 |
| `knowledge_items` | famille × contexte (type, taille, équilibre, métrique), leçon, acceptations, Δ moyen, confiance de Laplace | 6 |
| `knowledge_evidence` | lien leçon → (projet, expérience), Δ mesuré, soutien ou non | 6 |
| `estimate_calibration` | gain estimé contre gain observé, par famille d'hypothèses | 6 |

**Base projet `projects/<id>/project.db`**

| Table | Contenu | Phase |
|---|---|---|
| `data_sources` | chemin, rôle, sha256, taille, mtime, provenance, licence | 2 |
| `protocols` | tâche, classes, plis, fiabilité (identifiant = empreinte tâche + données + plis) | 3 |
| `data_profiles` | profil mesuré, par empreinte des données | 3 |
| `experiment_code` | code de chaque expérience, par empreinte (dédupliqué) | 3 |
| `experiment_decisions` | décision calculée, référence, Δ, bruit, règle appliquée (ajout seul) | 3 |
| `experiments` | fiche A.2 (colonnes clés + JSON complet), statut, horodatage | 2 |
| `experiment_metrics` | métrique × portée (pli/seed/simulation) × index, en ajout seul | 2 |
| `execution_events` | journal des transitions et incidents | 2 |
| `decisions` | accords demandés, réponse, auteur | 2 |
| `web_pages` | pages lues (web, API, texte collé, preuves tirées des données) : texte, liens, passages signalés | 4 |
| `competition_facts` | fiche A.1, un fait par ligne, citation vérifiée, statut found / inferred / user (ajout seul) | 4 |
| `user_actions` | ce que l'utilisateur doit faire (jeton, règlement, inbox, texte), ouvert puis fait | 4 |
| `compliance_log` | informations utilisées au-delà de l'entraînement, justification | 4 |
| `campaigns` | mode, pilote, statut (running, paused, waiting_user, stopped, finished), budget, raison | 5 |
| `approval_requests` | demandes d'accord en attente, accordées, refusées, consommées | 5 |
| `submissions` | soumissions déclarées par l'utilisateur, empreinte du fichier, score public | 6 |

Les hypothèses rejetées ne sont pas une table à part : elles se lisent dans `experiments` × `experiment_decisions`. Les artefacts conservés sont les dossiers des expériences de référence et des expériences acceptées.

**Intégrité imposée par la base** (déclencheurs SQLite) : la fiche d'une expérience est immuable, une expérience n'est jamais supprimée, les métriques sont en ajout seul. Une correction est une nouvelle expérience (`parent_id`).

## 4. Contrat des outils MCP

Outils : `project_list`, `project_create`, `campaign_next`, `campaign_prepare`, `competition_card`, `page_list`, `page_read`, `fact_record`, `data_profile`, `data_sample`, `memory_search`, `hypotheses_rank`, `experiment_submit`, `experiment_run`, `experiment_history`, `compare`, `external_request`, `deliverables_build`, `submission_plan`. Aucun outil ne donne d'accord.

Chaque erreur attendue revient en `isError` avec un message en français, sans arrêter le serveur. Chaque sortie est bornée à `llm.cycle_summary_tokens × 4` caractères. Les pages sont encadrées comme contenu externe. Les lignes brutes ne sont transmises que si `rules.data_sharing = allowed` est cité (statut found ou user), et 20 au maximum.

Le protocole est implémenté sans SDK, avec la bibliothèque standard : le SDK officiel dépend de pydantic-core, un module compilé que Smart App Control risque de bloquer.

## 5. Décisions et compromis

| Décision | Raison | Compromis accepté |
|---|---|---|
| Configuration TOML lue avec `tomllib` (bibliothèque standard) | Aucune dépendance ; lisible ; commentaires possibles | Écriture générée par ML Forge, pas de réécriture partielle |
| Clés inconnues refusées | Une faute de frappe ne doit pas être ignorée en silence | Un ancien fichier doit être mis à jour après une migration de configuration |
| Une base par projet + une base globale | Isolation physique ; sauvegarde par projet simple | Requêtes interprojets via la base globale seulement |
| Chemins de projet déduits de l'identifiant | Sauvegardes portables, pas de chemin absolu en base | La racine doit rester stable (configurable) |
| Connexions courtes (ouvrir / fermer) | Pas de fichiers `-wal`/`-shm` qui traînent, reprise simple | Léger coût d'ouverture, négligeable ici |
| `synchronous=FULL` | Robustesse aux coupures brutales | Écritures un peu plus lentes, volume faible |
| Sauvegarde automatique avant suppression d'un projet | Une suppression acceptée reste réversible | Un zip de plus dans `backups/` (petit : pas de datasets) |
| Pas de nettoyage automatique des sauvegardes | Aucune suppression automatique (§4.3) | `disk` affiche la place prise ; l'utilisateur décide |
| Python 3.14 dans `C:\MLForge\venv` | Les wheels 3.14 existent ; environnement hors OneDrive | Versions fixées à cause de Smart App Control (voir §6) |
| Le bac à sable lance l'interpréteur de base avec `__PYVENV_LAUNCHER__` | Le `python.exe` d'un venv Windows est un lanceur qui crée un processus enfant, interdit par le Job Object | Dépend d'un mécanisme interne de CPython, testé à chaque exécution des tests |
| ML Forge calcule les scores à partir de `oof.csv` | Un script ne peut pas déclarer son propre score | Le script reçoit les labels d'entraînement : une fuite volontaire reste possible ; un score quasi parfait est signalé |
| Pilote intégré sans LLM (plan de référence + étape de décision) | Le scénario « un lien en entrée, une soumission en sortie » fonctionne même sans modèle de langage | Pas d'hypothèse créative : c'est le rôle du pilote opencode |
| Faits critiques : found ou user seulement | Ne jamais deviner (§5.B) | Une déduction ambiguë demande une confirmation |
| Accord via la CLI, consommé une fois | Le LLM ne peut pas s'accorder une permission | En mode sans surveillance, la piste attend l'utilisateur ; les autres continuent |
| Solution exportée autonome (copie de `tabular.py`) | Reproductible hors ligne, sans ML Forge | Une copie à régénérer à chaque finalisation (jamais modifiée à la main) |
| Second régime = tirages de la taille du test dans les prédictions hors pli | Disponible sans données supplémentaires | Approximation : ne remplace pas un vrai jeu de test |
| Sorties conservées : baseline et expériences acceptées | Sobriété (§16) ; le code de toutes les expériences reste en base | Une expérience rejetée doit être relancée pour retrouver ses prédictions |
| Pas de processus enfants dans le bac à sable | Le Job Object les interdit, ce qui empêche une évasion | Parallélisme par threads uniquement |

## 6. Vérifications faites (12/10/2026)

| Point | Résultat | Source |
|---|---|---|
| Serveur MCP local dans opencode | `opencode.json` → `"mcp": {"mlforge": {"type": "local", "command": [...], "cwd": ..., "environment": {...}}}` | opencode.ai/docs/mcp-servers |
| Permissions opencode | Clés `bash`, `edit`, `webfetch`, `websearch`, `external_directory`… ; valeurs `allow`/`ask`/`deny` ; motifs, la dernière règle gagne | opencode.ai/docs/permissions |
| Mode non interactif | `opencode run --format json`, `--attach` à un `opencode serve`, `-c/-s` pour reprendre une session, `--auto` approuve ce qui n'est pas explicitement refusé | opencode.ai/docs/cli |
| Fichier d'instructions | `AGENTS.md` à la racine du projet (sinon `CLAUDE.md`) + clé `instructions` | opencode.ai/docs/rules |
| Matériel | Ryzen 7 7445HS 12 threads, 15,2 Go, RTX 4050 6 141 Mio (pilote 592.82), Python 3.14.0, SQLite 3.50.4 | détection locale |
| Smart App Control | **Activé.** Bloque numpy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1, scipy 1.18.1. Se chargent : numpy 2.5.2, scipy 1.16.3, pandas 2.3.3, scikit-learn 1.7.2 | essais locaux (phase 3) |
| Job Object sans droits administrateur | Fonctionne : limite mémoire, un seul processus, arrêt à la fermeture | tests `test_sandbox.py` |
| Page Kaggle sans jeton | Rendue en JavaScript : aucun texte exploitable ; ML Forge le dit et demande le jeton | essai réel sur /competitions/titanic |
| Ressources | Démonstration complète en environ 15 s ; 2 à 7 s et environ 0,9 Go maximum par exécution isolée | `forge demo`, journal `resources` |
| Migration d'un espace existant | v1 → v3 (global), v2 → v4 (projets), sauvegarde avant chaque migration | `C:\MLForge` réel |

Conséquence pour la sécurité : en mode sans surveillance, `--auto` approuve tout ce qui n'est pas explicitement `deny`. `opencode.json` **refuse** donc tout ce qui n'est pas indispensable.

## 7. Restant à vérifier

- API Kaggle avec un vrai jeton : forme des réponses JSON (`evaluationMetric`, `isKernelsSubmissionsOnly`, liste des fichiers) et code 403 quand le règlement n'est pas accepté. Implémenté et testé sur des réponses simulées.
- opencode (non installé sur ce PC) : comportement réel de `opencode run --format json` avec le modèle choisi, fenêtre de contexte, limites de débit, fiabilité des appels d'outils MCP.
- Quotas gratuits d'Exa pour la recherche web d'opencode.
- Chargement de LightGBM, XGBoost, CatBoost et Optuna sous Smart App Control.
- Zindi et ITU : traités par l'adaptateur générique puis `inbox/` ; aucune API spécifique n'est supposée.

## 8. Risques principaux

| Risque | Parade |
|---|---|
| Modèle gratuit trop faible pour appeler les outils correctement | Outils simples, erreurs explicites, méthode imposée par ML Forge ; test d'aptitude du modèle en phase 5 |
| Isolation imparfaite sous Windows sans Docker | Modèle de menace documenté ; audit hooks + Job Object ; permissions opencode restrictives |
| Réutilisation d'un PID après arrêt brutal (expérience vue « en cours ») | Verrou de campagne en phase 3 ; commande de reprise explicite |
| Pages de compétition illisibles (JavaScript, connexion) | Demande de copier-coller ou dépôt dans `inbox/`, jamais de supposition |
