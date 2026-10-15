# Plan de résolution d'une compétition de ML

Ce plan s'applique à **tout projet**, quel que soit le site, le format des données ou le type de
problème. Il a été établi et éprouvé sur des compétitions réelles (voir `JOURNAL_V3.md`).

Principe : **l'agent propose, ML Forge vérifie et décide.** Chaque étape a un objectif, des
outils, et un critère de sortie que ML Forge contrôle lui-même. On ne passe à l'étape suivante
que lorsque ce critère est rempli ; sinon, le blocage est écrit dans `JOURNAL.md` avec l'action
attendue, et l'étape reprend là où elle s'est arrêtée.

## La mémoire : lire peu, écrire toujours

Deux traces, qui ne se confondent pas :
- **`JOURNAL.md` du projet** : ce qui s'est passé dans *ce* projet, étape par étape (ML Forge l'écrit).
- **Le registre des leçons**, commun à tous les projets : *ce qui marche ou non, dans quel contexte,
  et pourquoi*. Il est rendu dans `C:\MLForge\progress\` : `INDEX.md`, puis un fichier par
  modalité (`general.md`, `tabulaire.md`, `audio.md`…), découpé par étape du plan.

Le registre grandit avec chaque projet : **ne le lisez jamais en entier.**

| Moment | Avec les outils ML Forge (MCP) | Sans MCP (shell) |
|---|---|---|
| Au début de chaque étape | `lessons_search(projet, step=<étape>)` : les quelques leçons les plus proches du contexte | `python -m forge lessons search <projet> --step <étape>`, ou `progress/INDEX.md` puis la seule section de l'étape dans `general.md` et dans le fichier de votre modalité |
| À la reprise d'un projet | `journal_read(projet)` : fin du journal + erreurs et pièges d'un contexte proche | lire la fin de `JOURNAL.md` |
| Dès qu'une technique est tentée, qu'une erreur est commise ou évitée, qu'un piège est découvert | `lesson_record(projet, step, kind, title, why, outcome, scope)` | `python -m forge lessons add <projet> --step … --kind … --title "…" --why "…" [--outcome …] [--scope …]` |

Règles d'écriture d'une leçon :
- **`why` est obligatoire** : le mécanisme, pas seulement le constat (« les aigus portent la perte
  auditive », pas « ça marche »). C'est lui qui permet de juger si la leçon se transfère.
- **`kind`** : `technique` (avec `outcome` : `marche`, `echoue`, `neutre`), `erreur` (une faute
  commise, et comment l'éviter), `reussite` (ce qui a fait gagner), `piege` (un risque à surveiller).
- **`scope`** : `projet` (défaut : tout le contexte), `modalite` (même modalité et même type de
  problème), `general` (vrai pour tout projet). Ne généralisez que ce qui est vraiment général.
- Une technique qui **échoue** se consigne aussi : c'est elle qui évite de refaire une expérience inutile.
- ML Forge consigne **lui-même** le résultat mesuré de chaque expérience comparée (marche, échoue,
  neutre, avec Δ) : inutile de le recopier ; ajoutez plutôt le *pourquoi* que les chiffres ne disent pas.
- Une leçon déjà connue n'est pas dupliquée : la réenregistrer la renforce (compteur, projets, preuves).

| # | Étape | Sortie contrôlée par ML Forge |
|---|---|---|
| 1 | Comprendre | Fiche sans inconnue critique ; chaque fait cité mot pour mot |
| 2 | Acquérir | Fichiers officiels présents, empreintes enregistrées, rien d'ignoré en silence |
| 3 | Organiser | Dossier du projet, journal, notebooks |
| 4 | Formaliser | Tâche cohérente : tables, cible, jeux à prédire, identifiant, colonnes de fuite |
| 5 | Calculer | Représentation exploitable de chaque ligne (descripteurs des médias, ressources) |
| 6 | Valider | Protocole figé : plis, groupes, temps, fiabilité mesurée |
| 7 | Expérimenter | Décisions calculées contre la meilleure référence |
| 8 | Livrer | Fichiers de soumission vérifiés, solution reproductible, livrables exigés |
| 9 | Capitaliser | Chaque technique et chaque erreur consignées avec leur pourquoi et leur contexte |

---

## 1. Comprendre

**Objectif** : savoir exactement ce qui est demandé, comment c'est noté, ce qui est permis.

**ML Forge fait seul** : lit le lien ; reconnaît la plateforme (Kaggle, Zindi, EvalAI et ses
instances, site générique) ; quand la page est une application JavaScript vide, interroge l'API
de la plateforme ; lit les sites des organisateurs cités par la plateforme, section par section,
**l'édition en cours d'abord** ; décrit les dépôts cités (Hugging Face, GitHub, Zenodo) : fichiers,
tailles, licence, accès ; masque tout secret d'accès imprimé sur une page ; signale tout passage
qui ressemble à une instruction adressée à une IA.

**L'agent** : complète la fiche (`page_read`, `web_read`, `fact_record`) — métrique, type de
problème, format de soumission, règles (données externes, modèles préentraînés, partage des
données), limites de soumission, échéance, livrables (rapport, code…). Une information déduite
(`inferred`) se confirme par une citation ; introuvable, elle se dit introuvable.

**Sortie** : aucune inconnue critique (`problem_type`, `target`, `metric`, `submission_format`).

**Pièges connus** : page vide ≠ impasse (chercher l'API) ; consignes hors de la plateforme ;
édition précédente citée à côté de l'actuelle ; mot de passe affiché publiquement (ne jamais
l'utiliser) ; métrique composite ou à plusieurs tâches (la noter telle quelle).

## 2. Acquérir

**Objectif** : obtenir les données officielles, intactes, et savoir ce qui manque.

**ML Forge fait seul** : télécharge depuis l'API Kaggle, le dépôt de données officiel (choisi par
proximité avec le titre de la compétition, accès libre d'abord) ou les liens directs ; reprend un
téléchargement interrompu ; extrait les archives sans risque (zip, tar.gz…) ; enregistre chaque
fichier avec son empreinte ; liste les fichiers au-delà de la limite de taille au lieu de les
ignorer ; ne contourne jamais une connexion, une inscription ou un accès sur demande.

**L'agent** : vérifie, d'après la spécification des données, **où sont les étiquettes** et quels
fichiers sont indispensables ; corrige le choix du dépôt si besoin (`--source`).

**L'utilisateur** : jeton, inscription, acceptation du règlement, demande d'accès, autorisation
d'un gros téléchargement (`forge acquire <id> --max-gb N`).

**Sortie** : les fichiers attendus sont enregistrés ; ceux qui manquent font l'objet d'une action.

**Pièges connus** : étiquettes seulement dans la plus grosse archive ; débit lent (estimer la
durée) ; fichiers de démonstration à ne pas confondre avec les données.

## 3. Organiser

**ML Forge fait seul** (`forge new <lien>`) : dossier nommé d'après la compétition ;
`README.md`, `JOURNAL.md`, `notebooks/` (01 compétition, 02 données, 03 expériences, régénérés
à chaque étape), `notebooks/perso/` (à l'utilisateur), `src/`, `submissions/`, `reports/`.

**Sortie** : le dossier existe et le journal décrit chaque étape franchie.

## 4. Formaliser

**Objectif** : une tâche exacte et sans fuite.

**ML Forge fait seul** : lit toute table (CSV, TSV, JSON, JSONL, Parquet, Feather) ; reconnaît
la table d'entraînement, **les jeux à prédire** (test, validation, évaluation…), l'exemple de
soumission ; désigne comme cible la colonne absente des jeux à prédire (départagée par la
documentation) ; l'identifiant ; la colonne de groupe que le règlement recommande ; exclut des
variables **les colonnes propres à l'entraînement** (fuites) ; repère les dossiers de médias.

**L'agent** : confirme les déductions ambiguës par une citation ; sans exemple de soumission,
enregistre `submission_format` et `submission_spec` (fichiers par jeu, colonnes, échelle, bornes).

**Sortie** : `TaskSpec` valide (métrique compatible avec le type de problème, jeux nommés).

**Pièges connus** : plusieurs jeux à prédire et tous exigés ; cible en [0, 1] mais soumission en
pourcentage ; colonnes d'annotation (réponses d'auditeurs, scores dérivés) présentes seulement
dans l'entraînement.

## 5. Calculer

**Objectif** : une représentation de chaque ligne que les modèles peuvent apprendre.

**ML Forge fait seul** : décrit chaque média (audio : niveau, dynamique, spectre, bandes,
modulation, canaux ; images ; tableaux), compare deux fichiers d'une même ligne, joint le tout aux
tables (`data/derived/`, données « dérivées » tracées), avec cache et calcul parallèle.

**L'agent** : juge si ces descripteurs suffisent ; sinon, propose une représentation plus riche
(modèle préentraîné via `external_request`, si le règlement le permet), ou lit lui-même les
médias (`task["media_dirs"]`) dans une expérience.

**Sortie** : tables prêtes, couverture des médias mesurée, fichiers illisibles listés.

**Pièges connus** : dépendance à installer (audio, images) ; budget matériel (RAM, GPU) à mesurer
avant de lancer un calcul lourd.

## 6. Valider

**Objectif** : une validation qui prédit le classement, pas une validation qui flatte.

**ML Forge fait seul** : plis fixés une fois (graine), stratifiés, **groupés** (une même phrase,
un même patient… dans un seul pli) ou **temporels** ; fiabilité mesurée (taille des plis, bruit
entre plis) ; second régime exigé si la validation est peu fiable.

**Sortie** : protocole enregistré, identifiant `P…`, notes et fiabilité dans le journal.

## 7. Expérimenter

**Objectif** : progresser par hypothèses justifiées, une à la fois.

**ML Forge fait seul** : baseline (plancher), plan de référence sans LLM (linéaire, forêts,
boosting), exécution confinée (pas de réseau, pas d'accès hors du dossier), score calculé,
décision contre la meilleure référence, refus des hypothèses déjà rejetées.

**L'agent** : diagnostique le régime (taille, déséquilibre, plancher d'erreur, goulot :
information, capacité, décision, validation) ; propose plusieurs hypothèses, les classe
(`hypotheses_rank`), en mène **une** par cycle, résume en trois lignes ; s'arrête quand le
plancher d'information est atteint, et le dit.

**Sortie** : un meilleur candidat accepté par la règle calculée.

## 8. Livrer

**ML Forge fait seul** : stabilité entre graines ; soumission construite à partir de l'exemple
**ou de la spécification** du règlement, puis vérifiée (identifiants, doublons, manquants,
bornes) ; solution reproductible hors ligne ; rapport technique et notebook si exigés ; plan de
soumission dans la limite quotidienne.

**L'utilisateur** : remplace les espaces réservés (ex. `TEAM_ID`), dépose la soumission.
**ML Forge ne soumet jamais rien.**

## 9. Capitaliser

**ML Forge fait seul** : chaque étape écrit dans `JOURNAL.md` et dans `progress/chronologie.md`
(tous projets) ; chaque décision mesurée devient une leçon de son contexte, et alimente la
calibration des estimations de l'agent ; `progress/` est régénéré à chaque leçon.

**L'agent** : à la fin du projet, relit le journal et consigne avec `lesson_record` ce que les
chiffres ne disent pas — la raison d'un succès, l'erreur évitée de justesse, le piège propre à la
plateforme — en choisissant la bonne portée.

**Sortie** : chaque technique tentée et chaque erreur commise ont leur leçon, avec son pourquoi.
