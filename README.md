# ML Forge

Laboratoire local et traçable pour compétitions de machine learning. **Vous donnez le lien d'une compétition ; ML Forge lit les règles, récupère les données, mène une campagne d'expériences rigoureuse et vous remet des livrables vérifiés.** Vous déposez vous-même la soumission.

- Spécification : [`prompt_v2.md`](prompt_v2.md)
- Architecture : [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- Instructions de l'agent : [`AGENTS.md`](AGENTS.md)
- Plan suivi pour tout projet : [`docs/PLAN.md`](docs/PLAN.md)
- Journal de la v3, étapes et problèmes rencontrés : [`docs/JOURNAL_V3.md`](docs/JOURNAL_V3.md)

Version 1.0 : les phases 1 à 8 sont terminées. Travail v3 en cours (du lien seul jusqu'aux livrables, éprouvé de bout en bout sur CLIP2) : 189 tests, dont un seul propre à chaque système, passent sous Windows. La CI GitHub les rejoue à chaque envoi.

## Installation (Windows)

```powershell
python -m venv C:\MLForge\venv
C:\MLForge\venv\Scripts\python -m pip install -r requirements-dev.txt   # numpy, scipy, pandas, scikit-learn, soundfile, pytest
cd "C:\Users\<vous>\OneDrive\Desktop\ML FORGE"
C:\MLForge\venv\Scripts\python -m forge init          # config\mlforge.toml + racine C:\MLForge
C:\MLForge\venv\Scripts\python -m forge config check  # matériel et chargement des bibliothèques ML
```

Toutes les commandes se lancent depuis le dossier du code avec `C:\MLForge\venv\Scripts\python -m forge …`. Ci-dessous, ce préfixe est noté `forge`.

- **Smart App Control** est actif sur ce PC. Il bloque des modules compilés trop récents (« An Application Control policy has blocked this file »). Les versions ci-dessus se chargent, et `config check` signale tout blocage.
- **Kaggle** : placez votre `kaggle.json` dans `C:\Users\<vous>\.kaggle\`. Vous l'obtenez sur kaggle.com, dans Settings > API > Create New Token. Ne collez jamais ce jeton dans une conversation. Pour chaque compétition, acceptez le règlement sur le site.
- **Boosters (optionnels)** : `pip install lightgbm xgboost catboost`. Ils sont utilisés s'ils se chargent ; vérifiez avec `config check`.

## Installation avec Docker (toute plateforme)

L'image est une Linux, Python 3.12, avec un utilisateur non privilégié. Les données vivent dans le volume `mlforge-data`, monté sur `/data`, et survivent aux conteneurs.

```bash
docker compose build
docker compose run --rm mlforge demo                    # démonstration reproductible
docker compose run --rm mlforge test                    # suite de tests complète dans le conteneur
docker compose run --rm mlforge create "Titanic" --url https://www.kaggle.com/competitions/titanic
docker compose run --rm mlforge campaign start <id>
```

- **Jeton Kaggle** : placez `kaggle.json` dans `./secrets/kaggle/`, ignoré par Git. Vous pouvez aussi définir `KAGGLE_DIR` vers votre dossier `.kaggle`. Il est monté en lecture seule.
- **opencode sur l'hôte, ML Forge dans le conteneur** : `forge setup-opencode --docker mlforge:1.0.0` écrit un `opencode.json` qui lance le serveur MCP avec `docker run -i … mcp`.
- **Mémoire du bac à sable** : `sandbox.memory_limit_gb` vaut 10 Go par défaut. Il doit rester sous la mémoire allouée à Docker ; `config check` vous avertit sinon.
- **Différence avec Windows** : sous Linux, la limite mémoire passe par `RLIMIT_AS`, et l'interdiction de créer des processus repose sur les audit hooks seuls, sans Job Object.

## Du lien aux livrables

Après la configuration initiale d'OpenCode (`forge setup-opencode`), envoyez le lien de compétition à OpenCode. Son outil MCP `project_create` crée le dossier et lance automatiquement la campagne dans un processus local détaché : l'appel MCP rend la main immédiatement, sans garder la conversation ouverte. Le worker prépare les données, puis lance un appel OpenCode non interactif (`opencode run --auto`) par cycle d'expérience, et génère les livrables à la fin. Ces cycles ont besoin du réseau pour joindre le modèle configuré ; ils ne gardent toutefois ni l'appel MCP ni une session de conversation interactive occupés pendant toute la campagne. L'appel initial renvoie l'identifiant, le dossier et le journal. La campagne peut s'arrêter en attente d'une action explicite de votre part (par exemple jeton Kaggle, acceptation du règlement ou fichier à déposer).

Si le serveur MCP ne trouve pas la commande `opencode` (notamment lorsqu'il tourne dans Docker), il ne prétend pas qu'une campagne est partie : l'agent doit poursuivre le flux avec les outils MCP de la session courante et appeler `deliverables_build` à la fin. La commande CLI ci-dessous reste disponible pour lancer ou reprendre une campagne manuellement.

```powershell
forge new <lien>                             # projet nommé d'après la compétition, préparé jusqu'au premier blocage
forge campaign start <id>                    # plan de référence, sans LLM, budget de nuit (8 h)
forge campaign start <id> --driver opencode  # l'agent LLM via opencode
forge status <id>                            # où en est-on, que dois-je faire ?
```

Une campagne suit les neuf étapes de [`docs/PLAN.md`](docs/PLAN.md) :
1. **comprendre** : plateforme, API (EvalAI…), sites des organisateurs, dépôts cités (Hugging Face, GitHub, Zenodo) ;
2. **acquérir** : téléchargement avec reprise, archives extraites, fichiers trop gros signalés ;
3. **organiser** : dossier, `JOURNAL.md`, notebooks régénérés ;
4. **formaliser** : tables de tout format, jeux à prédire, cible, identifiant, groupes, colonnes de fuite ;
5. **calculer** : descripteurs des médias (audio, images) joints aux tables ;
6. **valider** : plis stratifiés, groupés ou temporels, fiabilité mesurée ;
7. **expérimenter** : baseline puis hypothèses, décisions calculées ;
8. **livrer** : soumission vérifiée (fichier exemple ou spécification du règlement), solution reproductible, rapport ;
9. **capitaliser** : journal du projet, `PROGRESSION.md` commun à tous les projets, mémoire mesurée.

Quand elle a besoin de vous (jeton, règlement à accepter, fichiers à déposer, texte à coller, accord), elle s'arrête sur un message précis. Vous agissez, puis vous la relancez. Le PC ne se met pas en veille pendant une campagne sans surveillance.

**Livrables** produits dans `C:\MLForge\projects\<id>\deliverables\` :

| Fichier | Contenu |
|---|---|
| `submission.csv` | Soumission au format vérifié contre l'exemple de la compétition |
| `submission\` | Sans fichier exemple : un fichier par jeu, selon la spécification citée du règlement |
| `solution\` | Solution autonome et reproductible hors ligne : `python make_submission.py --data <dossier>` |
| `technical_report.md` | Rapport technique, si la compétition l'exige |
| `solution.ipynb` | Notebook, si exigé ou pour une compétition « code » (expérimental) |
| `README.md` | Liste de contrôle des livrables et recommandation de soumission |

## Brancher opencode

```powershell
npm install -g opencode-ai             # si opencode n'est pas installé
forge setup-opencode                   # écrit opencode.json : serveur MCP ML Forge et permissions
opencode                               # dans le dossier du code ; choisissez votre modèle
```

Cette configuration est à faire une fois. Ensuite, depuis OpenCode, l'envoi du lien déclenche la création et la campagne automatiquement ; consultez `forge status <id>` ou `forge campaign status <id>` pour suivre son état.

- ML Forge est un serveur MCP local. Les outils `mlforge_*` sont les seules actions de l'agent : shell, écriture de fichiers et lecture web directe sont refusés (`deny`).
- Le LLM ne reçoit jamais le dataset brut. Il ne reçoit des lignes que si le règlement autorise explicitement le partage, et 20 au maximum. Les sorties d'outils sont bornées.
- Les accords (données externes, modèle préentraîné, remplacement d'un livrable) ne passent jamais par le LLM. Vous les donnez avec `forge approvals <id>` puis `forge approve <id> <n>` (ou `refuse`).
- En cas de limite de débit ou de LLM indisponible, la campagne se met en pause. Reprise : `forge campaign resume <id>`.

## Interface locale en lecture seule

```powershell
forge ui                         # http://127.0.0.1:8765
forge ui --project <id>          # ouvrir un projet précis
forge ui --project <id> --port 9000
```

Le tableau de bord présente douze vues :
- **tous les projets** (vue d'accueil) : où en est chacun, meilleur score, ce qui vous attend, chronologie commune ;
- par projet : aperçu, **compétition** (fiche citée, dépôts décrits, pages lues), campagne, expériences, hypothèses, données (tâche formalisée, fichiers résumés par type et dossier), validation, livrables (fichiers, notebooks, tables dérivées), **leçons** du contexte du projet, mémoire et journal (`JOURNAL.md` et événements).

Une vue précise s'ouvre par l'adresse : `http://127.0.0.1:8765/?project=<id>&screen=lessons`. Il écoute uniquement sur `127.0.0.1`, interroge SQLite en lecture seule et actualise les données toutes les cinq secondes. Il n'affiche pas les lignes brutes des jeux de données (les faits mesurés sur les fichiers sont masqués), résume les dizaines de milliers de fichiers d'un projet au lieu de les lister, et ne permet aucune action de modification.

Le tableau de pilotage FastAPI (`forge web-server`, extra `web`) sert **le même tableau de bord** sous `/ui/`, avec la même source (`/api/state`) : les deux interfaces montrent exactement le même état.

## Commandes

| Commande | Effet |
|---|---|
| `new <lien> [--name N]` · `notebooks <id>` | Projet depuis le seul lien (nom, dossiers, préparation) ; notebooks régénérés |
| `create "Nom" --url <lien>` · `list [--all]` · `status <id>` | Projets ; `status` affiche aussi les actions qui vous attendent |
| `campaign start/stop/resume/status <id> [--driver builtin\|opencode] [--interactive]` | Campagne d'expériences (budget 8 h ; 2 h avec `--interactive`) |
| `intake <id>` · `card <id>` · `paste <id> regles.txt` · `fact <id> <champ> <valeur> --source URL --quote "…"` | Fiche de compétition : chaque fait est cité et vérifié |
| `acquire <id> [--source <dépôt>] [--max-gb N]` · `inbox <id>` | Données officielles : API Kaggle, dépôt cité, liens publics, ou fichiers déposés dans `projects\<id>\inbox\` ; `--max-gb` autorise un gros téléchargement pour cette fois |
| `history <id>` · `best <id>` · `compare <id> E0002 E0003` | Résultats et comparaisons (même protocole obligatoire) |
| `finalize <id>` · `submissions <id> [--declare E0004 --score 0.81]` | Livrables ; plan de soumission respectant la limite quotidienne |
| `approvals <id>` · `approve <id> <n>` · `refuse <id> <n>` | Vos accords |
| `memory <id>` | Leçons mesurées des autres projets et calibration des estimations |
| `lessons search <id> [--step] [--query]` · `lessons add <id> --step --kind --title --why [--outcome] [--scope]` · `lessons render` | Registre des leçons : chercher par contexte, consigner, régénérer `progress/` |
| `backup [<id>]` · `restore <zip>` · `archive <id>` · `delete <id>` · `disk` | Sauvegarde, restauration, archivage, suppression (confirmée et sauvegardée), espace |
| `demo` · `config show/check` · `setup-opencode` · `ui [--project <id>] [--port <n>]` | Démonstration, configuration, opencode, interface locale en lecture seule |

## Ce qui est mémorisé, et comment cela sert

- **Registre des expériences** (`project.db`), qui fait foi. Chaque expérience y est enregistrée avec :
  - sa fiche, immuable : question, hypothèse, justification, estimation préalable ;
  - son code, avec son empreinte ;
  - l'empreinte des données utilisées ;
  - ses métriques pli par pli, en ajout seul ;
  - la décision calculée : règle appliquée, Δ et bruit ;
  - son journal d'exécution, avec durée et mémoire.

  Une expérience n'est jamais modifiée ni supprimée. Une correction crée une nouvelle expérience.
- **Fiche de compétition** : chaque fait est accompagné de la citation exacte et de sa source. Une déduction ambiguë reste « inferred » et ne débloque pas les champs critiques.
- **Mémoire générale** (`forge.db`). Pour chaque famille d'hypothèses (par exemple `modèle:lightgbm` ou `features:dates`) et chaque contexte (type de problème, taille, déséquilibre, métrique), elle garde :
  - le nombre de fois où un gain a été accepté ;
  - le Δ moyen ;
  - les projets et expériences qui le prouvent.

  Elle est alimentée **uniquement par des décisions calculées**, jamais par du texte généré. Elle est présentée à l'agent comme des **pistes à vérifier**, jamais appliquée automatiquement.
- **Registre des leçons** (`forge.db`, rendu dans `C:\MLForge\progress\`) : chaque technique tentée, chaque erreur commise, chaque piège découvert, avec **son contexte** (modalité, problème, métrique, plateforme, taille), **son pourquoi** et ses preuves. ML Forge y verse lui-même le résultat de chaque expérience comparée ; l'agent y ajoute le pourquoi (`lesson_record`, ou `forge lessons add`). La recherche (`lessons_search`, `forge lessons search <id> --step <étape>`) ne renvoie que les leçons proches du contexte : **personne ne lit tout le registre**. `progress/INDEX.md` puis un fichier par modalité, découpé par étape, en donnent une lecture humaine ; `progress/chronologie.md` liste chaque étape de chaque projet.
- **Calibration** : ML Forge compare le gain que l'agent annonçait au gain mesuré, par famille d'hypothèses. Ce ratio corrige le classement des hypothèses suivantes (`hypotheses_rank`).
- **Ce que ce n'est pas** : le modèle de langage n'est jamais réentraîné. « Apprendre » signifie ici enregistrer des mesures et les relire avant de décider.

## Garanties de méthode

Elles sont imposées par le code, quel que soit le modèle de langage :
- aucune expérience sans fiche de compétition complète, profil des données et baseline ;
- le score est calculé par ML Forge, jamais déclaré par le script ;
- un résultat n'est accepté que si le gain dépasse le bruit mesuré entre les plis ;
- si la validation est peu fiable, il faut en plus un second régime : la simulation des conditions du test ;
- une hypothèse déjà rejetée n'est retestée qu'avec une raison nouvelle et écrite ;
- un score quasi parfait est signalé comme fuite possible ;
- le candidat final est contrôlé sur plusieurs seeds ;
- l'étape de décision (poids par classe pour une métrique portant sur des étiquettes) est ajustée hors pli, puis évaluée comme une expérience.

## Sauvegarde et restauration

- `forge backup` sauvegarde la base globale et tous les projets dans `C:\MLForge\backups\` (zip par projet). `forge backup <id>` sauvegarde un seul projet.
- Une sauvegarde est faite automatiquement avant chaque migration de schéma, à la fin de chaque campagne et avant toute suppression.
- `forge restore <fichier.zip>` restaure un projet supprimé ou perdu. Il refuse d'écraser un projet existant.
- Rien n'est supprimé automatiquement, y compris les anciennes sauvegardes : `forge disk` montre la place occupée.

## Tests et démonstration

```powershell
C:\MLForge\venv\Scripts\python -m pytest     # 189 tests, environ 5 min ; aucun accès réseau (un test de reprise utilise un serveur HTTP local)
forge demo                                    # environ 15 s : compétition synthétique, 3 modèles, soumission vérifiée
```

Ressources mesurées sur ce PC (Ryzen 7 7445HS, 15 Go) : 2 à 7 s par expérience de démonstration, environ 0,9 Go de mémoire maximale par exécution isolée.

## Limites connues

- **API Kaggle** : implémentée et testée sur des réponses simulées. À valider avec votre jeton : forme des réponses et code 403 quand le règlement n'est pas accepté.
- **Pages en JavaScript** : Kaggle et EvalAI (eval.ai, ITU…) passent par leur API ; les sites des organisateurs cités sont lus. Pour une plateforme sans API lisible (Zindi…), ML Forge vous demande de coller le texte des règles ou de déposer les fichiers dans `inbox\`.
- **opencode** n'est pas installé sur ce PC : le pilote LLM est testé avec un faux opencode. Le format exact des réponses de votre modèle reste à observer.
- **Isolation** : elle protège contre les erreurs du code généré, pas contre un code malveillant déterminé (du code natif ou `ctypes` peut contourner les audit hooks). Le Job Object limite quand même la mémoire et interdit les processus enfants. Le parallélisme passe donc par des threads uniquement.
- **Modalités** :
  - validé : tabulaire (classification, régression, anomalies supervisées), dates, séries temporelles (validation chronologique) ;
  - assisté : texte (TF-IDF), audio et images (descripteurs de base joints aux tables ; `pillow` requis pour les images) ;
  - expérimental : apprentissage profond. PyTorch n'est pas installé ; il le serait seulement avec votre accord.
- **Une seule tâche par compétition** (plusieurs jeux à prédire sont gérés) : les défis à plusieurs tâches, comme SoM 2026, se traitent en projets séparés.
- **Notebook de compétition « code »** : généré, mais non exécuté dans l'environnement de la plateforme.
- **Détection d'injection** : par motifs. La vraie protection est qu'aucune action sensible ne dépend du contenu web.

## Extensions possibles

- Navigateur automatisé (Playwright) pour les pages en JavaScript.
- Adaptateurs Zindi et ITU s'il existe une API officielle.
- Affectation sous contrainte d'effectifs (cas ITU).
- Modalités vision et NLP avec modèles préentraînés (avec accord).
- Interface graphique au-dessus de la même base.

## Paquet `ml_forge`

Le dossier `ml_forge/` contient un autre prototype, avec une interface de bureau (PySide6) et une interface web lancées par `main.py` et `main_web.py`. Son installation et son usage sont décrits dans [`docs/ml_forge.md`](docs/ml_forge.md).
