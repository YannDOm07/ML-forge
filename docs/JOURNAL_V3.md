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
