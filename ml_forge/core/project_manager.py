"""Gestion complète des projets ML Forge.

Responsable de la création, lecture, mise à jour et archivage
des projets de manière persistante et isolée.
"""

import json
import uuid
import shutil
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

from ml_forge.agents import LocalExperimentOrchestrator
from ml_forge.core.config import config
from ml_forge.core.logging import logger
from ml_forge.engines import DataEngine, ExperimentEngine, FeatureEngineer
from ml_forge.storage.database import Database


class ProjectManager:
    """Gère le cycle de vie des projets."""

    def __init__(self):
        self.projects_dir = config.projects_dir
        self.global_db = Database(self.projects_dir / "projects.db")

    def create_project(
        self,
        name: str,
        description: str = "",
        dataset_path: Optional[str] = None,
        target_column: Optional[str] = None,
    ) -> Optional[str]:
        """Crée un nouveau projet indépendant.
        
        Returns:
            project_id si succès, None sinon.
        """
        project_id = str(uuid.uuid4())[:8]
        project_dir = self.projects_dir / project_id
        
        # Vérifier qu'aucun projet n'existe avec cet ID (très improbable)
        if project_dir.exists():
            logger.error(f"Un projet avec l'ID {project_id} existe déjà")
            return None

        # Créer le répertoire du projet
        project_dir.mkdir(parents=True, exist_ok=True)
        
        # Créer la base de données du projet
        project_db_path = project_dir / "project.db"
        project_db = Database(project_db_path)

        # Enregistrer dans la base de données globale ET dans la base locale
        dataset_hash = None
        if dataset_path:
            dataset_hash = self._hash_file(dataset_path)

        project_data = {
            "id": project_id,
            "name": name,
            "description": description,
            "dataset_path": dataset_path,
            "dataset_hash": dataset_hash,
            "target_column": target_column,
            "objective": "",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "status": "active",
            "archived": 0,
        }

        # Insérer dans la base globale
        if self.global_db.insert("projects", project_data) is None:
            logger.error(f"Impossible de créer le projet {name}")
            shutil.rmtree(project_dir, ignore_errors=True)
            return None

        # Insérer dans la base locale aussi (elle est la source de vérité pour ce projet)
        if project_db.insert("projects", project_data) is None:
            logger.error(f"Impossible d'insérer le projet dans la base locale {name}")
            shutil.rmtree(project_dir, ignore_errors=True)
            self.global_db.delete("projects", {"id": project_id})
            return None

        logger.info(f"Projet créé : {name} ({project_id})")
        return project_id

    def open_project(self, project_id: str) -> Optional["Project"]:
        """Ouvre un projet existant."""
        project_data = self.global_db.query_one(
            "SELECT * FROM projects WHERE id = ?", (project_id,)
        )
        
        if not project_data:
            logger.error(f"Projet non trouvé : {project_id}")
            return None

        project_dir = self.projects_dir / project_id
        if not project_dir.exists():
            logger.error(f"Répertoire du projet manquant : {project_dir}")
            return None

        # Lire les données depuis la base de projet locale (source de vérité)
        project_db = Database(project_dir / "project.db")
        local_project_data = project_db.query_one(
            "SELECT * FROM projects WHERE id = ?", (project_id,)
        )
        
        if local_project_data:
            project_data = dict(local_project_data)

        return Project(project_id, project_dir, dict(project_data))

    def list_projects(self, archived: bool = False) -> List[Dict[str, Any]]:
        """Liste tous les projets."""
        query = "SELECT * FROM projects WHERE archived = ? ORDER BY updated_at DESC"
        return self.global_db.query(query, (1 if archived else 0,))

    def search_global_knowledge(
        self,
        task_type: Optional[str] = None,
        query: str = "",
        category: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Recherche dans la mémoire globale issue des connaissances validées."""
        clauses: List[str] = []
        params: List[Any] = []

        if task_type:
            task_pattern = f"%{task_type}%"
            clauses.append("(content LIKE ? OR category LIKE ? OR ref_sources LIKE ?)")
            params.extend([task_pattern, task_pattern, task_pattern])

        if query:
            query_pattern = f"%{query}%"
            clauses.append("(content LIKE ? OR category LIKE ? OR ref_sources LIKE ?)")
            params.extend([query_pattern, query_pattern, query_pattern])

        if category:
            clauses.append("category = ?")
            params.append(category)

        sql = "SELECT * FROM knowledge_items"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        results = self.global_db.query(sql, tuple(params))
        if task_type and not results:
            return self.global_db.query(
                "SELECT * FROM knowledge_items ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
        return results

    def archive_project(self, project_id: str) -> bool:
        """Archive un projet (le marque comme archivé sans le supprimer)."""
        return self.global_db.update(
            "projects",
            {"archived": 1, "updated_at": datetime.now().isoformat()},
            {"id": project_id},
        )

    def delete_project(self, project_id: str, confirm: bool = False) -> bool:
        """Supprime définitivement un projet (demande confirmation).
        
        Cette opération est destructive et irréversible.
        """
        if not confirm:
            logger.warning(f"Suppression de {project_id} refusée : confirmation manquante")
            return False

        project_dir = self.projects_dir / project_id
        try:
            shutil.rmtree(project_dir, ignore_errors=True)
            self.global_db.delete("projects", {"id": project_id})
            logger.info(f"Projet supprimé définitivement : {project_id}")
            return True
        except Exception as e:
            logger.error(f"Erreur lors de la suppression du projet : {e}")
            return False

    def backup_project(self, project_id: str) -> Optional[Path]:
        """Crée une sauvegarde complète du projet."""
        project_dir = self.projects_dir / project_id
        if not project_dir.exists():
            logger.error(f"Projet non trouvé : {project_id}")
            return None

        backup_dir = config.backups_dir / f"{project_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            shutil.copytree(project_dir, backup_dir)
            logger.info(f"Sauvegarde du projet : {backup_dir}")
            return backup_dir
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde : {e}")
            return None

    def restore_project(self, backup_path: Path) -> Optional[str]:
        """Restaure un projet à partir d'une sauvegarde."""
        if not backup_path.exists():
            logger.error(f"Sauvegarde non trouvée : {backup_path}")
            return None

        try:
            # Générer un nouvel ID pour éviter les collisions
            project_id = str(uuid.uuid4())[:8]
            project_dir = self.projects_dir / project_id
            shutil.copytree(backup_path, project_dir)
            
            # Charger et mettre à jour la base du projet restauré
            project_db = Database(project_dir / "project.db")
            
            # Mettre à jour l'ID du projet dans la base restaurée
            old_project_id = None
            old_proj = project_db.query_one("SELECT id FROM projects LIMIT 1")
            if old_proj:
                old_project_id = old_proj["id"]
                project_db.update(
                    "projects",
                    {"id": project_id, "updated_at": datetime.now().isoformat()},
                    {"id": old_project_id},
                )
                
                # Mettre à jour les expériences avec le nouvel ID de projet
                if old_project_id:
                    project_db.update(
                        "experiments",
                        {"project_id": project_id},
                        {"project_id": old_project_id},
                    )
            
            # Enregistrer le projet restauré dans la base globale
            project_info = project_db.query_one(
                "SELECT * FROM projects WHERE id = ?", (project_id,)
            )
            
            if project_info:
                self.global_db.insert("projects", dict(project_info))
            
            logger.info(f"Projet restauré : {project_id}")
            return project_id
        except Exception as e:
            logger.error(f"Erreur lors de la restauration : {e}")
            return None

    def export_project(self, project_id: str, export_path: Path) -> bool:
        """Exporte un projet (prêt pour la compétition ou le partage)."""
        project_dir = self.projects_dir / project_id
        if not project_dir.exists():
            logger.error(f"Projet non trouvé : {project_id}")
            return False

        try:
            export_path.mkdir(parents=True, exist_ok=True)
            shutil.copytree(project_dir, export_path / project_id, dirs_exist_ok=True)
            logger.info(f"Projet exporté : {export_path / project_id}")
            return True
        except Exception as e:
            logger.error(f"Erreur lors de l'export : {e}")
            return False

    def update_project_objective(self, project_id: str, objective: str) -> bool:
        """Met à jour l'objectif d'un projet dans la base globale ET dans la base du projet."""
        # Mise à jour dans la base globale
        success = self.global_db.update(
            "projects",
            {"objective": objective, "updated_at": datetime.now().isoformat()},
            {"id": project_id},
        )
        
        # Mise à jour dans la base du projet
        if success:
            project_dir = self.projects_dir / project_id
            if project_dir.exists():
                project_db = Database(project_dir / "project.db")
                project_db.update(
                    "projects",
                    {"objective": objective, "updated_at": datetime.now().isoformat()},
                    {"id": project_id},
                )
        
        return success

    @staticmethod
    def _hash_file(file_path: str) -> str:
        """Calcule le hash SHA256 d'un fichier."""
        sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256.update(chunk)
            return sha256.hexdigest()
        except Exception as e:
            logger.error(f"Erreur lors du calcul du hash : {e}")
            return ""


class Project:
    """Représente un projet ML actif."""

    def __init__(self, project_id: str, project_dir: Path, metadata: Dict[str, Any]):
        self.id = project_id
        self.dir = project_dir
        self.metadata = metadata
        self.db = Database(project_dir / "project.db")

    def get_name(self) -> str:
        return self.metadata.get("name", "")

    def get_objective(self) -> str:
        return self.metadata.get("objective", "")

    def get_dataset_path(self) -> Optional[str]:
        return self.metadata.get("dataset_path")

    def get_target_column(self) -> Optional[str]:
        return self.metadata.get("target_column")

    def inspect_dataset(self, dataset_path: str, target_column: Optional[str] = None) -> Dict[str, Any]:
        """Inspecte un dataset et retourne une synthèse exploitable."""
        data_engine = DataEngine()
        df = data_engine.load_dataset(dataset_path)
        return data_engine.inspect_dataset(df, target_column=target_column)

    def run_baseline_experiment(
        self,
        dataset_path: str,
        target_column: str,
        hypothesis: Optional[str] = None,
        method: Optional[str] = None,
        code: str = "",
    ) -> Dict[str, Any]:
        """Lance une baseline ML sur un dataset du projet."""
        dataset_file = Path(dataset_path)
        if not dataset_file.exists():
            raise FileNotFoundError(f"Dataset introuvable : {dataset_path}")

        data_engine = DataEngine()
        df = data_engine.load_dataset(dataset_file)
        summary = data_engine.inspect_dataset(df, target_column=target_column)

        if target_column not in df.columns:
            raise ValueError(f"La colonne cible '{target_column}' est absente du dataset.")

        X = df.drop(columns=[target_column])
        y = df[target_column]
        experiment_engine = ExperimentEngine(random_state=42)
        result = experiment_engine.run_experiment(X, y, task_type=summary.get("task_type"))

        if not hypothesis:
            hypothesis = summary.get("suggested_hypotheses", [f"Baseline {summary.get('task_type', 'unknown')}"])[0]
        if not method:
            method = f"Baseline {summary.get('task_type', 'unknown')} with {result['model']}"

        experiment_id = self.add_experiment(hypothesis=hypothesis, method=method, code=code or "")
        if not experiment_id:
            raise RuntimeError("L'ajout de l'expérience a échoué.")

        metrics = result.get("metrics", {})
        for metric_name, metric_value in metrics.items():
            if metric_name == "classification_report":
                continue
            self.db.insert(
                "experiment_metrics",
                {
                    "experiment_id": experiment_id,
                    "metric_name": str(metric_name),
                    "metric_value": float(metric_value),
                    "metric_type": "score",
                },
            )

        updated_at = datetime.now().isoformat()
        self.metadata["dataset_path"] = str(dataset_file)
        self.metadata["target_column"] = target_column
        self.metadata["updated_at"] = updated_at

        self.db.update(
            "projects",
            {
                "dataset_path": str(dataset_file),
                "target_column": target_column,
                "updated_at": updated_at,
            },
            {"id": self.id},
        )

        global_db = Database(self.dir.parent / "projects.db")
        global_db.update(
            "projects",
            {
                "dataset_path": str(dataset_file),
                "target_column": target_column,
                "updated_at": updated_at,
            },
            {"id": self.id},
        )

        self.db.update(
            "experiments",
            {
                "status": "completed",
                "started_at": updated_at,
                "ended_at": updated_at,
                "duration_seconds": float(result.get("duration_seconds", 0)),
                "data_hash": hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest(),
                "code": code or method,
            },
            {"id": experiment_id},
        )

        result["experiment_id"] = experiment_id
        result["dataset_summary"] = summary
        result["project_id"] = self.id
        return result

    def run_advanced_experiment(
        self,
        dataset_path: str,
        target_column: str,
        hypothesis: Optional[str] = None,
        method: Optional[str] = None,
        code: str = "",
        feature_strategy: str = "ratio",
        optimize_hyperparameters: bool = False,
    ) -> Dict[str, Any]:
        """Lance une expérience avancée avec features dérivées et optimisation légère."""
        dataset_file = Path(dataset_path)
        if not dataset_file.exists():
            raise FileNotFoundError(f"Dataset introuvable : {dataset_path}")

        df = DataEngine().load_dataset(dataset_file)
        if target_column not in df.columns:
            raise ValueError(f"La colonne cible '{target_column}' est absente du dataset.")

        X = df.drop(columns=[target_column])
        y = df[target_column]
        summary = DataEngine().inspect_dataset(df, target_column=target_column)

        feature_engine = FeatureEngineer(strategy=feature_strategy)
        X_engineered = feature_engine.generate_features(X, strategy=feature_strategy)

        experiment_engine = ExperimentEngine(random_state=42)
        result = experiment_engine.run_experiment(
            X_engineered,
            y,
            task_type=summary.get("task_type"),
            feature_generator=lambda df: df,
            optimize_hyperparameters=optimize_hyperparameters,
            feature_strategy="none",
        )

        baseline = experiment_engine.run_experiment(
            X,
            y,
            task_type=summary.get("task_type"),
            feature_strategy="none",
            optimize_hyperparameters=False,
        )

        if not hypothesis:
            hypothesis = summary.get("suggested_hypotheses", [f"Expérience avancée {summary.get('task_type', 'unknown')}"])[0]
        if not method:
            method = f"Advanced {summary.get('task_type', 'unknown')} with {result['model']}"

        experiment_id = self.add_experiment(hypothesis=hypothesis, method=method, code=code or method)
        if not experiment_id:
            raise RuntimeError("L'ajout de l'expérience avancée a échoué.")

        for metric_name, metric_value in result.get("metrics", {}).items():
            if metric_name == "classification_report":
                continue
            self.db.insert(
                "experiment_metrics",
                {
                    "experiment_id": experiment_id,
                    "metric_name": str(metric_name),
                    "metric_value": float(metric_value),
                    "metric_type": "score",
                },
            )

        self.db.update(
            "experiments",
            {
                "status": "completed",
                "started_at": datetime.now().isoformat(),
                "ended_at": datetime.now().isoformat(),
                "duration_seconds": float(result.get("duration_seconds", 0)),
                "data_hash": hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest(),
                "code": code or method,
            },
            {"id": experiment_id},
        )

        baseline_score = baseline.get("metrics", {}).get("accuracy", baseline.get("metrics", {}).get("r2", 0.0))
        candidate_score = result.get("metrics", {}).get("accuracy", result.get("metrics", {}).get("r2", 0.0))
        result["experiment_id"] = experiment_id
        result["project_id"] = self.id
        result["dataset_summary"] = summary
        result["comparison"] = {
            "baseline_score": baseline_score,
            "candidate_score": candidate_score,
            "delta": candidate_score - baseline_score,
            "reference": "baseline",
        }
        return result

    def set_objective(self, objective: str) -> bool:
        """Met à jour l'objectif du projet."""
        self.metadata["objective"] = objective
        self.metadata["updated_at"] = datetime.now().isoformat()
        
        # Mettre à jour à la fois la base du projet et la base globale
        success = self.db.update(
            "projects",
            {"objective": objective, "updated_at": self.metadata["updated_at"]},
            {"id": self.id},
        )
        
        # Mettre à jour aussi la base globale
        if success:
            pm_db = Database(self.dir.parent / "projects.db")
            pm_db.update(
                "projects",
                {"objective": objective, "updated_at": self.metadata["updated_at"]},
                {"id": self.id},
            )
        
        return success

    def add_knowledge(
        self,
        category: str,
        content: str,
        confidence_score: float = 0.7,
        ref_sources: str = "",
    ) -> Optional[int]:
        """Ajoute une connaissance dans la mémoire persistante du projet et dans la mémoire globale."""
        payload = {
            "category": category,
            "content": content,
            "confidence_score": float(confidence_score),
            "created_at": datetime.now().isoformat(),
            "verified_at": datetime.now().isoformat(),
            "ref_sources": ref_sources,
        }
        project_id = self.db.insert("knowledge_items", payload)
        if project_id is None:
            return None

        global_db = Database(self.dir.parent / "projects.db")
        global_db.insert("knowledge_items", payload)
        return project_id

    def list_knowledge(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Récupère les connaissances mémorisées du projet."""
        if category:
            return self.db.query(
                "SELECT * FROM knowledge_items WHERE category = ? ORDER BY created_at DESC",
                (category,),
            )
        return self.db.query("SELECT * FROM knowledge_items ORDER BY created_at DESC")

    def get_relevant_knowledge(self, query: str = "", category: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
        """Récupère les connaissances pertinentes pour le projet courant."""
        if query:
            return self.db.query(
                "SELECT * FROM knowledge_items WHERE content LIKE ? OR category LIKE ? ORDER BY created_at DESC LIMIT ?",
                (f"%{query}%", f"%{query}%", limit),
            )
        if category:
            return self.db.query(
                "SELECT * FROM knowledge_items WHERE category = ? ORDER BY created_at DESC LIMIT ?",
                (category, limit),
            )
        return self.db.query("SELECT * FROM knowledge_items ORDER BY created_at DESC LIMIT ?", (limit,))

    def record_decision(self, decision_text: str, reason: str, experiment_id: Optional[str] = None) -> Optional[int]:
        """Enregistre une décision de projet dans la base locale et la globale."""
        payload = {
            "project_id": self.id,
            "decision_text": decision_text,
            "reason": reason,
            "experiment_id": experiment_id,
            "created_at": datetime.now().isoformat(),
        }
        decision_id = self.db.insert("decisions", payload)
        if decision_id is not None:
            global_db = Database(self.dir.parent / "projects.db")
            global_db.insert("decisions", payload)
        return decision_id

    def compare_experiments(self, first_experiment_id: str, second_experiment_id: str) -> Dict[str, Any]:
        """Compare deux expériences en se basant sur leurs métriques communes."""
        first_metrics = {
            row["metric_name"]: row["metric_value"]
            for row in self.db.query(
                "SELECT metric_name, metric_value FROM experiment_metrics WHERE experiment_id = ?",
                (first_experiment_id,),
            )
        }
        second_metrics = {
            row["metric_name"]: row["metric_value"]
            for row in self.db.query(
                "SELECT metric_name, metric_value FROM experiment_metrics WHERE experiment_id = ?",
                (second_experiment_id,),
            )
        }

        metric_name = None
        for candidate in ["accuracy", "f1", "r2", "precision", "recall", "mae", "mse", "rmse"]:
            if candidate in first_metrics and candidate in second_metrics:
                metric_name = candidate
                break

        if metric_name is None:
            return {
                "first_experiment_id": first_experiment_id,
                "second_experiment_id": second_experiment_id,
                "metric_name": None,
                "first_value": None,
                "second_value": None,
                "delta": 0.0,
                "direction": "equal",
            }

        first_value = float(first_metrics[metric_name])
        second_value = float(second_metrics[metric_name])
        delta = second_value - first_value
        direction = "improvement" if delta >= 0 else "regression"

        return {
            "first_experiment_id": first_experiment_id,
            "second_experiment_id": second_experiment_id,
            "metric_name": metric_name,
            "first_value": first_value,
            "second_value": second_value,
            "delta": delta,
            "direction": direction,
        }

    def build_experiment_plan(self, dataset_path: str, target_column: Optional[str] = None) -> Dict[str, Any]:
        """Construit un plan d'expérience local à partir du dataset et de l'analyse."""
        summary = self.inspect_dataset(dataset_path, target_column=target_column)
        orchestrator = LocalExperimentOrchestrator()
        plan = orchestrator.build_plan(summary, project_name=self.get_name())
        self.add_knowledge(
            "experiment_plan",
            json.dumps(plan, ensure_ascii=False),
            confidence_score=0.8,
            ref_sources=str(dataset_path),
        )
        return plan

    def add_experiment(
        self,
        hypothesis: str,
        method: str,
        code: str = "",
    ) -> Optional[str]:
        """Ajoute une nouvelle expérience au projet."""
        experiment_id = str(uuid.uuid4())[:8]
        
        experiment_data = {
            "id": experiment_id,
            "project_id": self.id,
            "hypothesis": hypothesis,
            "method": method,
            "code": code,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
        }
        
        if self.db.insert("experiments", experiment_data) is None:
            logger.error(f"Impossible d'ajouter une expérience au projet {self.id}")
            return None

        logger.info(f"Expérience créée : {experiment_id}")
        return experiment_id

    def get_experiments(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retourne les expériences du projet."""
        if status:
            query = "SELECT * FROM experiments WHERE project_id = ? AND status = ? ORDER BY created_at DESC"
            return self.db.query(query, (self.id, status))
        else:
            query = "SELECT * FROM experiments WHERE project_id = ? ORDER BY created_at DESC"
            return self.db.query(query, (self.id,))

    def get_best_experiment(self) -> Optional[Dict[str, Any]]:
        """Retourne la meilleure expérience basée sur les métriques."""
        query = """
        SELECT e.* FROM experiments e
        WHERE e.project_id = ? AND e.status = 'completed'
        ORDER BY e.id DESC
        LIMIT 1
        """
        return self.db.query_one(query, (self.id,))
