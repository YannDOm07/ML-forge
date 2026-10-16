"""Local orchestration logic for experiment planning."""

from __future__ import annotations

from typing import Any, Dict, List


class LocalExperimentOrchestrator:
    """Orchestre l'analyse et le plan d'expérimentation d'un projet."""

    def build_plan(self, dataset_summary: Dict[str, Any], project_name: str = "Projet") -> Dict[str, Any]:
        """Construit un plan de secours déterministe ; OpenCode fournit le raisonnement IA."""
        task_type = dataset_summary.get("task_type", "unknown")
        return {
            "project_name": project_name,
            "task_type": task_type,
            "objective": self._default_objective(task_type, project_name),
            "steps": self._fallback_steps(task_type, dataset_summary),
            "recommended_metrics": self._recommended_metrics(task_type),
            "risk_flags": self._risk_flags(dataset_summary),
            "llm_generated": False,
        }

    @staticmethod
    def _default_objective(task_type: str, project_name: str) -> str:
        if task_type == "classification":
            return f"Améliorer la qualité de prédiction du modèle sur {project_name}"
        if task_type == "regression":
            return f"Réduire l'erreur de prédiction sur {project_name}"
        return f"Établir une baseline fiable pour {project_name}"

    @staticmethod
    def _recommended_metrics(task_type: str) -> List[str]:
        if task_type == "classification":
            return ["accuracy", "precision", "recall", "f1"]
        if task_type == "regression":
            return ["r2", "mae", "mse", "rmse"]
        return ["score", "stability"]

    @staticmethod
    def _risk_flags(dataset_summary: Dict[str, Any]) -> List[str]:
        flags: List[str] = []
        if dataset_summary.get("missing_values"):
            flags.append("Valeurs manquantes détectées : imputation ou filtrage requis.")
        if dataset_summary.get("duplicate_rows", 0) > 0:
            flags.append("Doublons détectés : risque de fuite ou surapprentissage.")
        if dataset_summary.get("class_distribution"):
            imbalance = max(dataset_summary["class_distribution"].values()) - min(dataset_summary["class_distribution"].values())
            if imbalance > 0.25:
                flags.append("Déséquilibre de classes : attention à la métrique et au calibrage.")
        return flags

    @staticmethod
    def _fallback_steps(task_type: str, dataset_summary: Dict[str, Any]) -> List[str]:
        steps = [
            "Inspecter le dataset et vérifier les colonnes, valeurs manquantes et types.",
            "Créer une baseline simple sur les variables utiles avec validation cohérente.",
            "Mesurer les métriques de référence et comparer la stabilité de la validation.",
            "Recommencer avec des transformations ou des modèles plus adaptés selon les diagnostics.",
        ]
        if task_type == "classification":
            steps.insert(2, "Vérifier le déséquilibre de classes et la robustesse de la métrique principale.")
        return steps
