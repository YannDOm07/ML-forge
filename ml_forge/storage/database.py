"""Couche de stockage SQLite.

Gère la persistance des projets, expériences, métriques et connaissances.
"""

import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Any, List, Dict, Optional, Tuple
from contextlib import contextmanager

from ml_forge.core.logging import logger


class Database:
    """Gestionnaire de base de données SQLite."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def connection(self):
        """Context manager pour les connexions SQLite."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_schema(self):
        """Initialise le schéma de base de données."""
        with self.connection() as conn:
            cursor = conn.cursor()

            # Table projects
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    dataset_path TEXT,
                    dataset_hash TEXT,
                    target_column TEXT,
                    objective TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    status TEXT,
                    archived INTEGER DEFAULT 0
                )
                """
            )

            # Table experiments
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    hypothesis TEXT,
                    method TEXT,
                    status TEXT,
                    created_at TEXT,
                    started_at TEXT,
                    ended_at TEXT,
                    duration_seconds REAL,
                    code TEXT,
                    error_message TEXT,
                    data_hash TEXT,
                    FOREIGN KEY (project_id) REFERENCES projects (id)
                )
                """
            )

            # Table experiment_metrics
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS experiment_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL,
                    metric_name TEXT,
                    metric_value REAL,
                    metric_type TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (id)
                )
                """
            )

            # Table hypotheses
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS hypotheses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id TEXT NOT NULL,
                    hypothesis_text TEXT,
                    status TEXT,
                    justification TEXT,
                    created_at TEXT,
                    tested_at TEXT,
                    result TEXT,
                    FOREIGN KEY (project_id) REFERENCES projects (id)
                )
                """
            )

            # Table decisions
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id TEXT NOT NULL,
                    decision_text TEXT,
                    reason TEXT,
                    experiment_id TEXT,
                    created_at TEXT,
                    FOREIGN KEY (project_id) REFERENCES projects (id),
                    FOREIGN KEY (experiment_id) REFERENCES experiments (id)
                )
                """
            )

            # Table knowledge_items
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT,
                    content TEXT,
                    confidence_score REAL,
                    created_at TEXT,
                    verified_at TEXT,
                    ref_sources TEXT
                )
                """
            )

            # Table artifacts
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS artifacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id TEXT,
                    experiment_id TEXT,
                    artifact_type TEXT,
                    path TEXT NOT NULL,
                    size_bytes INTEGER,
                    hash TEXT,
                    created_at TEXT,
                    kept_reason TEXT,
                    FOREIGN KEY (project_id) REFERENCES projects (id),
                    FOREIGN KEY (experiment_id) REFERENCES experiments (id)
                )
                """
            )

            # Table execution_events
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS execution_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT,
                    event_type TEXT,
                    event_data TEXT,
                    timestamp TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (id)
                )
                """
            )

            # Table configuration_history
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS configuration_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id TEXT,
                    config_key TEXT,
                    config_value TEXT,
                    changed_at TEXT,
                    changed_by TEXT,
                    FOREIGN KEY (project_id) REFERENCES projects (id)
                )
                """
            )

            conn.commit()
            logger.info(f"Schéma de base de données initialisé : {self.db_path}")

    def insert(self, table: str, data: Dict[str, Any]) -> Optional[Any]:
        """Insère une ligne et retourne son ID."""
        columns = ", ".join(data.keys())
        placeholders = ", ".join(["?" for _ in data])
        query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        
        with self.connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(query, tuple(data.values()))
                conn.commit()
                return cursor.lastrowid
            except sqlite3.Error as e:
                logger.error(f"Erreur d'insertion dans {table}: {e}")
                conn.rollback()
                return None

    def update(self, table: str, data: Dict[str, Any], where: Dict[str, Any]) -> bool:
        """Met à jour des lignes."""
        set_clause = ", ".join([f"{k} = ?" for k in data.keys()])
        where_clause = " AND ".join([f"{k} = ?" for k in where.keys()])
        query = f"UPDATE {table} SET {set_clause} WHERE {where_clause}"
        
        with self.connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(query, tuple(data.values()) + tuple(where.values()))
                conn.commit()
                return cursor.rowcount > 0
            except sqlite3.Error as e:
                logger.error(f"Erreur de mise à jour dans {table}: {e}")
                conn.rollback()
                return False

    def query(self, query_str: str, params: Tuple = ()) -> List[Dict[str, Any]]:
        """Exécute une requête SELECT et retourne les résultats."""
        with self.connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(query_str, params)
                return [dict(row) for row in cursor.fetchall()]
            except sqlite3.Error as e:
                logger.error(f"Erreur lors de la requête : {e}")
                return []

    def query_one(self, query_str: str, params: Tuple = ()) -> Optional[Dict[str, Any]]:
        """Exécute une requête SELECT et retourne le premier résultat."""
        results = self.query(query_str, params)
        return results[0] if results else None

    def delete(self, table: str, where: Dict[str, Any]) -> bool:
        """Supprime des lignes."""
        where_clause = " AND ".join([f"{k} = ?" for k in where.keys()])
        query = f"DELETE FROM {table} WHERE {where_clause}"
        
        with self.connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(query, tuple(where.values()))
                conn.commit()
                return cursor.rowcount > 0
            except sqlite3.Error as e:
                logger.error(f"Erreur de suppression dans {table}: {e}")
                conn.rollback()
                return False

    def backup(self, backup_path: Path) -> bool:
        """Crée une sauvegarde de la base de données."""
        try:
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            with self.connection() as src_conn:
                with sqlite3.connect(backup_path) as dst_conn:
                    src_conn.backup(dst_conn)
            logger.info(f"Sauvegarde de la base de données : {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde : {e}")
            return False

    def restore(self, backup_path: Path) -> bool:
        """Restaure la base de données à partir d'une sauvegarde."""
        try:
            if not backup_path.exists():
                logger.error(f"Fichier de sauvegarde non trouvé : {backup_path}")
                return False
            with sqlite3.connect(backup_path) as src_conn:
                with self.connection() as dst_conn:
                    src_conn.backup(dst_conn)
            logger.info(f"Base de données restaurée : {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Erreur lors de la restauration : {e}")
            return False
