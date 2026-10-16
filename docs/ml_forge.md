# ML Forge

ML Forge est un laboratoire local de machine learning pensé pour la recherche, la traçabilité et la reproductibilité. Il permet de créer des projets isolés, d’inspecter des datasets, de lancer des expériences de référence ou avancées, d’enregistrer des résultats dans SQLite et de réutiliser des connaissances pertinentes entre projets.

## 1. Architecture

Le projet s’appuie sur une architecture modulaire et légère :

- `ml_forge/core` : configuration, logging, gestion des projets, persistance.
- `ml_forge/engines` : inspection des données, moteur d’expérimentation, génération de features.
- `ml_forge/agents` : orchestration locale et plan d’expérimentation.
- `ml_forge/storage` : schéma SQLite, sauvegardes et restauration.
- `ml_forge/gui` : interface de bureau minimale (PySide6), extensible.

## 2. Installation locale

1. Créer et activer un environnement virtuel.
2. Installer les dépendances :

```powershell
cd "C:\DEV\ML FORGE"
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Le projet utilise scikit-learn, pandas, numpy, pytest et PySide6. Aucun service distant n’est requis pendant l’usage local.

## 3. Exécution

### Application GUI

```powershell
cd "C:\DEV\ML FORGE"
.\.venv\Scripts\python.exe main.py
```

### Interface web HTML/CSS/JavaScript

```powershell
cd "C:\DEV\ML FORGE"
.\.venv\Scripts\python.exe main_web.py
```

Le serveur local ouvre l'interface dans le navigateur sur `http://127.0.0.1:8765`. Collez une URL publique et cliquez une fois pour autoriser sa lecture : ML Forge crée le projet, récupère un dataset direct ou lié, puis lance une baseline si la cible est reconnue. L'interface est en HTML/CSS/JavaScript ; le serveur Python local exécute scikit-learn et conserve projets et résultats dans SQLite.

Vous pouvez aussi coller une URL publique de page HTML ou un lien direct vers un CSV/TSV. Après le clic explicite sur **Analyser le lien et démarrer**, ML Forge essaie de trouver un CSV lié à la page, crée le projet, archive un brief de la page et lance une baseline si un nom de cible reconnu est trouvé. Les pages privées, liens exigeant une authentification, fichiers autres que CSV/TSV et téléchargements générés par JavaScript ne sont pas pris en charge. Les consignes sont conservées comme texte source ; en l'absence d'un interpréteur LLM local, elles ne sont pas encore automatiquement comprises ni appliquées à la stratégie ML. Si la cible ne peut pas être déterminée sans ambiguïté, le projet est créé mais l'entraînement est arrêté pour éviter une évaluation trompeuse.

### Démonstration reproductible

```powershell
cd "C:\DEV\ML FORGE"
.\.venv\Scripts\python.exe examples\demo_reproducible.py
```

## 4. Exemple d’usage en Python

```python
from ml_forge.core.project_manager import ProjectManager

pm = ProjectManager()
project_id = pm.create_project(name="Demo")
project = pm.open_project(project_id)

result = project.run_baseline_experiment(
    "C:/path/to/dataset.csv",
    target_column="target",
    hypothesis="La baseline doit établir un point de référence stable.",
    method="Baseline LogisticRegression",
)

print(result["metrics"])
```

## 5. Mémoire et traçabilité

La mémoire est persistée dans SQLite :

- `projects` : métadonnées des projets,
- `experiments` : historique des expériences,
- `experiment_metrics` : scores et mesures,
- `knowledge_items` : leçons réutilisables,
- `decisions` : décisions claires associées aux résultats,
- `configuration_history` : évolution de la configuration.

Les résultats ne sont pas reconstruits à partir de souvenirs conversationnels ; ils sont lus dans la base de données.

## 6. Sauvegarde et restauration

```python
from ml_forge.core.project_manager import ProjectManager

pm = ProjectManager()
project_id = pm.create_project(name="Projet important")
backup_path = pm.backup_project(project_id)
print(backup_path)
```

La restauration se fait par la méthode `restore_project()` depuis un chemin de sauvegarde.

## 7. Limites connues

- Le système reste orienté local et léger ; il n’implémente pas un LLM distant.
- L’orchestration est locale et déterministe, sans apprentissage général en dehors de l’enregistrement de résultats.
- Les modèles de type vision, NLP ou séries temporelles ne sont pas encore supportés explicitement.
- Les performances de la GUI dépendent de l’environnement d’affichage local.

## 8. Validation

Le projet est validé par pytest. La suite actuelle couvre la persistance, la mémoire, l’isolation des projets, l’expérimentation ML et l’orchestration locale.

```powershell
cd "C:\DEV\ML FORGE"
.\.venv\Scripts\python.exe -m pytest -q
```
