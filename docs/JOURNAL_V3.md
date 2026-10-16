# Journal de la v3 — leçons tirées des compétitions réelles

Chaque étape est d'abord menée sur une vraie compétition. Ce qui bloque devient une capacité
générale de ML Forge (jamais un cas particulier). Ce journal alimente le plan final.

## Compétition 1 — SoM Challenge 2026 (ITU AI for Good, EvalAI)

Lien : https://competition.aiforgood.itu.int/web/challenges/challenge-page/494/overview

### Étape 1 — Comprendre (14/10/2026)

Constats :
- La page est une application JavaScript (EvalAI) : le HTML ne contient aucun texte.
- L'API n'est pas sur le même domaine (`health.aiaudit.org`) ; son adresse est déclarée dans
  `/dist/js/config.min.js`. L'appeler sur le domaine du site renvoie 400.
- Les données sont sur Hugging Face, les codes sur GitHub : liens sans extension de fichier,
  donc invisibles pour l'ancienne détection.
- Un mot de passe d'accès au jeu de préentraînement est imprimé dans les consignes, lisibles
  par l'API sans inscription. Il est réservé aux inscrits.
- Le README du baseline cite d'autres ressources (poids WiFo-2) : il faut lire les dépôts
  cités par les dépôts, avec une borne.
- La compétition est multi-tâche et multimodale (CSI complexes, images, cartes de profondeur),
  score composite (F1 + NMSE en dB + MAE + paramètres + FLOPs), modèle de fondation obligatoire,
  > 80 % de poids partagés, < 20 % de paramètres affinés. La fiche actuelle (colonne cible,
  métrique unique) ne sait pas le décrire : c'est le chantier de l'étape 3.

Capacités ajoutées :
- `forge/evalai.py` : reconnaît toute instance EvalAI par son chemin, trouve l'API, lit
  challenge et phases (échéance et limite quotidienne de la phase active).
- `forge/repositories.py` : reconnaît et décrit les dépôts Hugging Face (données, modèles)
  et GitHub : fichiers, tailles, licence, accès sur demande, README ; 10 dépôts au plus.
- Masquage des secrets d'accès à l'enregistrement de chaque page (`redact_secrets`).
- Outil agent `web_read` : lire n'importe quelle page avec les mêmes protections.

Leçons pour le plan :
1. Une page vide n'est pas une impasse : chercher l'API de la plateforme avant de demander
   à l'utilisateur de coller du texte.
2. Les sources de données se reconnaissent par leur hôte (Hugging Face, GitHub, Drive…),
   pas par l'extension.
3. Un secret trouvé sur une page ne s'utilise jamais : l'inscription et l'accès restent à
   l'utilisateur.
4. Mesurer tôt les contraintes matérielles : ici 10 Go de données, apprentissage profond,
   GPU de 6 Go, 15 Go de RAM.

## Compétition 2 — CLIP2, Cadenza Lyrics Intelligibility (eval.ai)

Lien : https://eval.ai/web/challenges/challenge-page/2703/overview — choisie pour tester le
téléchargement sans authentification : données ouvertes sur Zenodo (CC-BY-4.0).

### Étape 1 bis — Comprendre, sur une autre instance EvalAI (15/10/2026)

Constats :
- Le lecteur EvalAI fonctionne tel quel sur eval.ai (API déclarée sur le même domaine).
- Les vraies consignes (données, règles, métrique) sont sur le site des organisateurs,
  cité par la plateforme ; les données sont sur Zenodo, cité par ce site.
- Le site cite aussi l'édition précédente (CLIP1) : lue en premier, elle épuisait le quota.
- Les pages citent des outils de code (black, pylint…) qui épuisaient le quota de dépôts.

Capacités ajoutées :
- Lecture des sites cités par la plateforme, section par section ; la section la plus citée
  (l'édition en cours) passe en premier.
- Zenodo (`zenodo.org/records/<id>`, DOI `10.5281/zenodo.<id>`) dans `forge/repositories.py`.
- Dépôts décrits par priorité : données, puis modèles, puis code.

### Étape 2 — Acquérir (15/10/2026)

Capacités ajoutées :
- Acquisition depuis le dépôt de données officiel : choisi par proximité avec le titre de la
  compétition (CLIP2 plutôt que CLIP1), accès libre d'abord ; `forge acquire --source` corrige
  le choix.
- Fichiers au-delà de `acquisition.max_download_gb` : listés avec leur taille et une action
  ouverte, jamais ignorés en silence ; les autres sont téléchargés.
- Archives `.tar.gz`, `.tgz`, `.tar.bz2`, `.tar.xz` : extraites avec le filtre `data`
  (ni traversée de chemin, ni liens), une seule fois (marqueur), puis supprimées.
- Reprise des téléchargements interrompus (requêtes `Range`) : le fichier `.part` est conservé.

Mesures réelles : débit ≈ 2,5 Mo/s ; 2,9 Go en ≈ 20 min ; le fichier d'entraînement
(13,8 Go) prendrait ≈ 1 h 30.

Leçons pour le plan :
5. Les consignes réelles peuvent vivre hors de la plateforme : suivre les liens qu'elle cite.
6. Toujours distinguer l'édition en cours des éditions passées.
7. Un budget de lecture se dépense d'abord sur les données.
8. Un gros téléchargement doit pouvoir reprendre ; estimer sa durée avant de le lancer.

Reste à faire pour l'acquisition : vérifier les sommes de contrôle publiées (Zenodo : md5,
Hugging Face : sha256 LFS) ; télécharger seulement les fichiers utiles à la tâche.

Résultat réel : 2,9 Go (évaluation + validation) téléchargés et extraits en 19,7 min ;
8 612 fichiers enregistrés avec leur empreinte. Archive d'entraînement (13,8 Go) relancée
avec `forge acquire <id> --max-gb 16` (autorisation ponctuelle, journalisée).

Problèmes rencontrés et corrections :
- La commande affichait les 8 612 chemins un par un : la sortie résume désormais par dossier
  et par type de fichier.
- Message de limite incohérent (5 Gio affichés « 5,37 Go ») : le message reprend la valeur
  configurée et indique l'option `--max-gb`.
- Constat clé : **les étiquettes ne sont que dans l'archive d'entraînement**. Sans elle, aucun
  entraînement : il faut lire la spécification des données avant de choisir quoi télécharger.

### Étape 3 — Organiser (15/10/2026)

Capacités ajoutées (`forge/organize.py`) :
- `forge new <lien>` : nom du projet = titre de la compétition (lu sur la plateforme), puis
  lecture, acquisition, formalisation… jusqu'au premier blocage.
- Arborescence fixe : `README.md`, `JOURNAL.md`, `notebooks/` (01_competition, 02_donnees,
  03_experiences, régénérés à chaque étape), `notebooks/perso/` (jamais touché), `src/`,
  `submissions/`, `reports/`, `inbox/`.
- Chaque étape de `campaign_prepare` écrit dans `JOURNAL.md` (résultat, problème ⚠) ; une
  erreur inattendue y est consignée avant d'être remontée.
- `project_create` (MCP) n'exige plus de nom.

### Étape 4 — Formaliser (15/10/2026)

Constats sur CLIP2 : tables JSON (pas de CSV) ; deux jeux à prédire (validation, évaluation) ;
la table d'entraînement a deux colonnes absentes des autres (`correctness`, la cible, et
`phoneme_correctness`, une fuite) ; chaque phrase est présente 3 fois (3 niveaux de perte
auditive) et le règlement recommande de séparer les plis par phrase.

Capacités ajoutées :
- `read_table` unique (CSV, TSV, JSON en liste ou enveloppe, JSONL, Parquet, Feather), utilisé
  partout, y compris dans le bac à sable et la solution exportée.
- `TaskSpec` : `prediction_sets` (plusieurs jeux à prédire, le principal en premier),
  `media_dirs` (médias lisibles par les expériences, inclus dans l'empreinte des données),
  `train_only_columns` (exclues des variables : garde contre les fuites).
- `forge/formalize.py` : rôles des tables d'après leurs noms et dossiers (train / valid / eval /
  test / exemple ; manifestes et sommes de contrôle ignorés) ; cible = colonne absente des jeux à
  prédire, départagée par la phrase de la documentation qui la désigne ; identifiant = colonne
  commune, unique par ligne ; colonne de groupe tirée d'une phrase « split by unique prompts ».
  Chaque conclusion est un fait cité ; une cible ambiguë reste « déduite » et attend confirmation.
- Fiche : nouveaux champs `group_column` et `submission_spec`.

Problème rencontré : collision de noms — le runner importait déjà un autre `read_table` ; le
nombre de lignes était pris pour une liste de colonnes. Corrigé (import unique, argument nommé).

### Étape 5 — Calculer (15/10/2026)

Constats : aucune bibliothèque audio installée ; extraits FLAC stéréo 44,1 kHz d'environ 5 s ;
un fichier « signals » (entendu, perte simulée) et un « unprocessed » (référence) par ligne.

Capacités ajoutées (`forge/media.py`, `formalize.with_media_descriptors`) :
- `soundfile` 0.14.0 ajouté aux dépendances (Smart App Control ne le bloque pas, vérifié).
- Descripteurs de base de chaque média (audio : niveau, crête, dynamique, silence, centroïde,
  rolloff, planéité, ZCR, modulation 2–8 Hz, énergie par bande, équilibre gauche/droite ;
  images et tableaux `.npy` aussi), et différences entre deux fichiers d'une même ligne.
- Tables dérivées `data/derived/*__media.csv`, enregistrées comme données « dérivées »
  (migration v5 du schéma), qui remplacent les tables d'origine dans la tâche.
- Cache par fichier (chemin, taille, date) sous l'espace de travail ; calcul parallèle ; un
  fichier illisible n'arrête jamais les autres ; couverture < 50 % : descripteurs non joints.

Mesure : 200 fichiers en 2,8 s. Le descripteur `diff__band_4000_6000_db` vaut 0 dB (sans
perte), −22 dB (légère), −39 dB (modérée) : l'information de la perte auditive est captée.

### Étape 8 — Livrer, sans fichier exemple (15/10/2026)

Constat : CLIP2 demande deux CSV `ICASSP2027_<jeu>_<TEAM_ID>.csv`, colonnes
`signal_id, intelligibility_score`, score en pourcentage (0–100) alors que la cible est en 0–1.

Capacités ajoutées (`forge/submission.py`) :
- `submission_spec` (fichiers par jeu, colonnes, échelle, bornes), validée à l'enregistrement.
- Un fichier vérifié par jeu : mêmes identifiants que le jeu, sans doublon, sans valeur
  manquante, dans les bornes ; le remplacement d'un livrable validé exige toujours un accord.
- La solution exportée reproduit tous les jeux et applique la même spécification.

Problème rencontré pendant l'implémentation : les scripts de modification passés par le
shell transformaient les `\n` des chaînes Python en vrais retours à la ligne (syntaxe cassée
dans trois modules). Détecté par la compilation des modules, réparé automatiquement, et les
modifications suivantes passent par des fichiers de script. Plus tard, des accents graves dans
une commande ont été interprétés par le shell et ont amputé `AGENTS.md` : réparé ; tout texte
passe désormais par l'outil d'édition.

### Étape 9 — Capitaliser (15/10/2026)

Capacités ajoutées :
- `PROGRESSION.md` à la racine de l'espace de travail : une ligne par étape, tous projets
  confondus ; ⚠ marque un problème à ne pas reproduire.
- Outil agent `journal_read` : fin du journal du projet + problèmes rencontrés ailleurs.
- `docs/PLAN.md` : le plan en neuf étapes suivi pour tout projet ; `AGENTS.md` y renvoie et
  demande de relire le journal au début de chaque cycle.

### Déroulé réel complet sur CLIP2 (15/10/2026)

1. Archive d'entraînement : 13,8 Go en 86,5 min (≈ 2,7 Mo/s) ; 49 215 fichiers enregistrés.
2. Formaliser (seul, cité) : régression, identifiant `signal`, groupe `prompt` (« we recommend
   splitting by unique prompts first »), fuite `phoneme_correctness` exclue ; cible
   `correctness` déduite parmi deux candidates, donc « à confirmer ».
3. Rôle de l'agent (joué via l'outil MCP `fact_record`) : cible, `submission_format`,
   `submission_spec` et `rules.pretrained_models = allowed`, chacun avec sa citation exacte.
4. Calculer : 84 descripteurs par ligne, couverture 100 % (20 297 + 2 117 + 2 189 lignes),
   40 594 fichiers d'entraînement décrits en ≈ 10 min (≈ 80 fichiers/s).
5. Valider : 5 plis groupés par phrase, validation jugée fiable.
6. Expérimenter (plan de référence, 26 min) :

   | Exp. | Modèle | RMSE (0–1) | Décision |
   |---|---|---|---|
   | E0001 | constante | 0,37537 | référence |
   | E0002 | linéaire | 0,32236 | acceptée (Δ +0,053) |
   | E0003 | forêt aléatoire | 0,32675 | rejetée |
   | E0004 | gradient boosting | 0,32265 | rejetée |
   | E0005 | extra trees | 0,32604 | rejetée |

7. Livrer : `ICASSP2027_valid_TEAM_ID.csv` (2 117 lignes) et `ICASSP2027_eval_TEAM_ID.csv`
   (2 189 lignes), scores 0–100 vérifiés ; solution reproductible (reproduction **identique**
   vérifiée hors de ML Forge) ; rapport technique ; plan de soumission (5 par jour).

Problèmes rencontrés pendant ce déroulé, et corrections :
- Le téléchargement avait été lancé avec l'ancien code : les déductions de l'étape 4 n'ont pas
  été faites. Relancer `acquire` les a produites — d'où la correction suivante.
- Ré-enregistrer 49 215 fichiers déjà connus prenait 375 s (empreinte recalculée, une connexion
  par fichier) : fichier inchangé (taille, date) = non relu, et enregistrement par lot → 84 s.
- Une acquisition partielle (fichier trop gros en attente) laissait la campagne formaliser un
  jeu incomplet : tant qu'une action d'acquisition est ouverte, l'étape reste « Acquérir ».
- Une règle déduite d'un README sans rapport (« modèles préentraînés : interdits ») masquait la
  règle réelle (« allowed but must be fully declared ») : corrigée par une citation ; d'où
  l'importance que l'agent vérifie chaque règle déduite.
- Le journal recevait une entrée par palier de 2 000 fichiers : l'avancement va désormais dans
  le journal des événements ; le journal du projet garde un bilan par table.
- Les cycles d'expériences et la livraison n'écrivaient rien dans le journal, et les notebooks
  n'étaient pas régénérés en fin de campagne : corrigé (une entrée par cycle avec score, Δ et
  décision ; une entrée « Livrer » ; notebooks régénérés).
- Rapport : la cible s'affichait comme un dictionnaire brut ; README de la solution : `--out`
  et origine des tables dérivées mal décrits. Corrigé pour les prochaines finalisations (les
  livrables de CLIP2 déjà validés ne sont pas remplacés sans votre accord).

## Registre des leçons (15/10/2026)

Demande : que l'agent consigne ses tâches, ses erreurs, ses réussites et les techniques qui
marchent ou non, dans quel contexte et pourquoi ; un plan général suivi par tout agent ; et que
l'agent ne lise que ce qui concerne son contexte, le registre grandissant avec le temps.

Capacités ajoutées (`forge/lessons.py`, schéma global v4) :
- Une leçon = étape du plan, type (technique, erreur, réussite, piège), résultat (marche,
  échoue, neutre), **pourquoi obligatoire**, contexte (modalité, problème, métrique, plateforme,
  taille ; « * » = tout), preuves, nombre de fois vue. Réenregistrer la même leçon la renforce ;
  une même expérience ne compte jamais deux fois ; des preuves contraires la rendent « neutre ».
- ML Forge verse lui-même chaque décision calculée ; l'agent ajoute le pourquoi
  (`lesson_record`, `forge lessons add`), avec une portée : projet, modalité ou général.
- Recherche par contexte (`lessons_search`, `forge lessons search`) : une leçon d'une autre
  modalité ou d'un autre type de problème n'est jamais renvoyée ; quelques leçons seulement.
- `progress/` : `INDEX.md` court, un fichier par modalité découpé par étape, `chronologie.md`.
- 19 leçons de fondation livrées avec ML Forge (tirées de ce journal) ; l'état donné à l'agent
  à chaque cycle contient les leçons de l'étape en cours ; `docs/PLAN.md` et `AGENTS.md` fixent
  quand lire et quoi consigner.

Problèmes rencontrés : une récursion infinie (le chargement des leçons de fondation appelait
l'enregistrement, qui rechargeait les fondations) ; une recherche par mots-clés écrasée par le
poids du contexte (corrigé : une question explicite pèse plus que le contexte).

## Interface web reliée à tous les projets (15/10/2026)

Capacités ajoutées (`forge/ui/details.py`, écrans du tableau de bord) : vue « Tous les
projets » (accueil), vue Compétition (fiche citée, dépôts, pages), vue Leçons (contexte du
projet), Données (tâche formalisée, fichiers résumés), Livrables (fichiers, notebooks, tables
dérivées), Journal (`JOURNAL.md`) ; adresse directe `?project=<id>&screen=<vue>`. Le serveur
FastAPI sert le même tableau de bord sous `/ui/` avec la même source `/api/state`.

Problèmes rencontrés et corrections :
- L'écran Données envoyait les 49 218 fichiers de CLIP2 toutes les 5 s : fichiers choisis par
  SQL (tables d'abord, 100 au plus), agrégats sur chaînes et mis en cache → état construit en
  0,25 s au lieu de 3,5 s.
- Vue d'ensemble : le total d'expériences additionnait les compteurs (12 au lieu de 7), le
  meilleur score était lu au mauvais endroit, la fiabilité jamais transmise, le menu surlignait
  le mauvais écran, « Connecté » s'ajoutait à « Connexion… », la prochaine étape proposait de
  démarrer une campagne terminée. Tous corrigés et vérifiés par capture d'écran.
- Fuite de confidentialité attrapée par un test existant : la vue Compétition montrait la
  valeur des faits mesurés sur les fichiers ; elle est désormais masquée comme ailleurs.
- Une action « informations à fournir » restait ouverte une fois la fiche complétée par
  l'agent : elle se ferme quand le dernier fait critique est cité.

## Bilan et suite

Ce qui est démontré : à partir du seul lien CLIP2, ML Forge lit la compétition (plateforme,
site des organisateurs, Zenodo), télécharge 16,7 Go, comprend une tâche non tabulaire
(audio + métadonnées JSON, deux jeux à prédire, fuite, groupes), calcule une représentation,
valide correctement, expérimente et livre des fichiers conformes et reproductibles. L'agent
n'intervient que là où un jugement est nécessaire (confirmer une cible ambiguë, écrire la
spécification de soumission), toujours avec citation.

Ce qui reste à faire, honnêtement :
- **Le score est un plancher** : RMSE 0,322 (32,2 points sur l'échelle du classement) avec
  des descripteurs simples. Le baseline officiel s'appuie sur Whisper (autorisé s'il est
  déclaré) : c'est la prochaine hypothèse, mais elle exige PyTorch et le modèle préentraîné,
  donc votre accord (`external_request`).
- **L'agent LLM (opencode) n'a pas encore été testé en conditions réelles** : son rôle a été
  joué par les mêmes outils MCP.
- **SoM 2026** reste à l'étape 2 : inscription, accès sur demande et PyTorch dépendent de vous.
- Vérifier les sommes de contrôle publiées (manifestes md5 de CLIP2, sha256 Hugging Face) ;
  ne télécharger que les fichiers utiles à la tâche.
- Rapport technique : aujourd'hui un brouillon factuel ; CLIP2 demande un rapport de deux
  pages au format ICASSP, à rédiger à partir de lui.
