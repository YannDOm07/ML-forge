"""Versioned schemas. Append a migration; never edit one that has shipped."""

GLOBAL_MIGRATIONS: list[str] = [
    # v1 — project registry and configuration history
    """
    CREATE TABLE projects (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        competition_url TEXT,
        status TEXT NOT NULL CHECK (status IN ('active', 'archived', 'deleted')),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    CREATE TABLE configuration_history (
        id INTEGER PRIMARY KEY,
        config_hash TEXT NOT NULL,
        content TEXT NOT NULL,
        recorded_at TEXT NOT NULL
    );
    """,
    # v2 — every network access made by ML Forge
    """
    CREATE TABLE network_log (
        id INTEGER PRIMARY KEY,
        project_id TEXT,
        method TEXT NOT NULL,
        url TEXT NOT NULL,
        purpose TEXT NOT NULL,
        status INTEGER,
        bytes INTEGER,
        error TEXT,
        at TEXT NOT NULL
    );
    """,
    # v3 — general memory built from measured results, and calibration of the agent's estimates
    """
    CREATE TABLE knowledge_items (
        id INTEGER PRIMARY KEY,
        family TEXT NOT NULL,
        context_key TEXT NOT NULL,
        context_json TEXT NOT NULL,
        statement TEXT NOT NULL,
        n_supporting INTEGER NOT NULL,
        n_contradicting INTEGER NOT NULL,
        n_projects INTEGER NOT NULL,
        mean_delta REAL NOT NULL,
        confidence REAL NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE (family, context_key)
    );
    CREATE TABLE knowledge_evidence (
        id INTEGER PRIMARY KEY,
        knowledge_id INTEGER NOT NULL REFERENCES knowledge_items(id),
        project_id TEXT NOT NULL,
        experiment_id TEXT NOT NULL,
        delta REAL NOT NULL,
        supports INTEGER NOT NULL CHECK (supports IN (0, 1)),
        recorded_at TEXT NOT NULL,
        UNIQUE (project_id, experiment_id)
    );
    CREATE TABLE estimate_calibration (
        id INTEGER PRIMARY KEY,
        project_id TEXT NOT NULL,
        experiment_id TEXT NOT NULL,
        family TEXT NOT NULL,
        expected_gain REAL NOT NULL,
        observed_gain REAL NOT NULL,
        recorded_at TEXT NOT NULL,
        UNIQUE (project_id, experiment_id)
    );
    """,
]

PROJECT_MIGRATIONS: list[str] = [
    # v1 — data sources, experiment registry, events, decisions
    """
    CREATE TABLE data_sources (
        id INTEGER PRIMARY KEY,
        role TEXT NOT NULL CHECK (role IN ('official', 'external', 'inbox')),
        path TEXT NOT NULL UNIQUE,
        sha256 TEXT NOT NULL,
        size_bytes INTEGER NOT NULL,
        mtime_ns INTEGER NOT NULL,
        origin_url TEXT,
        license TEXT,
        registered_at TEXT NOT NULL
    );

    CREATE TABLE experiments (
        id TEXT PRIMARY KEY,
        parent_id TEXT REFERENCES experiments(id),
        question TEXT NOT NULL,
        hypothesis TEXT NOT NULL,
        method TEXT NOT NULL,
        protocol_id TEXT NOT NULL,
        card_json TEXT NOT NULL,
        code_hash TEXT,
        data_hash TEXT,
        status TEXT NOT NULL
            CHECK (status IN ('registered', 'running', 'completed', 'failed', 'interrupted')),
        runner_pid INTEGER,
        created_at TEXT NOT NULL,
        started_at TEXT,
        finished_at TEXT
    );
    CREATE TRIGGER experiments_card_immutable
    BEFORE UPDATE OF id, parent_id, question, hypothesis, method, protocol_id,
                     card_json, code_hash, data_hash, created_at ON experiments
    BEGIN
        SELECT RAISE(ABORT, 'experiment card is immutable: register a new experiment');
    END;
    CREATE TRIGGER experiments_never_deleted BEFORE DELETE ON experiments
    BEGIN
        SELECT RAISE(ABORT, 'experiments are never deleted');
    END;

    CREATE TABLE experiment_metrics (
        experiment_id TEXT NOT NULL REFERENCES experiments(id),
        name TEXT NOT NULL,
        scope TEXT NOT NULL CHECK (scope IN ('fold', 'seed', 'simulation')),
        idx INTEGER NOT NULL,
        value REAL NOT NULL,
        PRIMARY KEY (experiment_id, name, scope, idx)
    );
    CREATE TRIGGER metrics_append_only_update BEFORE UPDATE ON experiment_metrics
    BEGIN
        SELECT RAISE(ABORT, 'metrics are append-only');
    END;
    CREATE TRIGGER metrics_append_only_delete BEFORE DELETE ON experiment_metrics
    BEGIN
        SELECT RAISE(ABORT, 'metrics are append-only');
    END;

    CREATE TABLE execution_events (
        id INTEGER PRIMARY KEY,
        experiment_id TEXT REFERENCES experiments(id),
        kind TEXT NOT NULL,
        message TEXT NOT NULL,
        at TEXT NOT NULL
    );

    CREATE TABLE decisions (
        id INTEGER PRIMARY KEY,
        action TEXT NOT NULL,
        target TEXT NOT NULL,
        details_json TEXT NOT NULL,
        outcome TEXT NOT NULL CHECK (outcome IN ('approved', 'refused')),
        decided_by TEXT NOT NULL,
        at TEXT NOT NULL
    );
    """,
    # v2 — validation protocols, data profiles, experiment code, computed decisions
    """
    ALTER TABLE experiments ADD COLUMN runner_created INTEGER;

    CREATE TABLE protocols (
        id TEXT PRIMARY KEY,
        spec_json TEXT NOT NULL,
        folds_json TEXT NOT NULL,
        reliability_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    );

    CREATE TABLE data_profiles (
        id INTEGER PRIMARY KEY,
        data_hash TEXT NOT NULL,
        profile_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    );

    CREATE TABLE experiment_code (
        code_hash TEXT PRIMARY KEY,
        content TEXT NOT NULL
    );

    CREATE TABLE experiment_decisions (
        experiment_id TEXT PRIMARY KEY REFERENCES experiments(id),
        reference_id TEXT REFERENCES experiments(id),
        status TEXT NOT NULL CHECK (status IN ('baseline', 'accepted', 'rejected', 'not_significant')),
        metric_mean REAL NOT NULL,
        metric_std REAL NOT NULL,
        delta REAL,
        noise REAL,
        rule TEXT NOT NULL,
        computed_at TEXT NOT NULL
    );
    CREATE TRIGGER decisions_append_only_update BEFORE UPDATE ON experiment_decisions
    BEGIN
        SELECT RAISE(ABORT, 'computed decisions are append-only');
    END;
    CREATE TRIGGER decisions_append_only_delete BEFORE DELETE ON experiment_decisions
    BEGIN
        SELECT RAISE(ABORT, 'computed decisions are append-only');
    END;
    """,
    # v3 — competition intake: pages read, cited facts, compliance, actions awaited from the user
    """
    CREATE TABLE web_pages (
        id INTEGER PRIMARY KEY,
        url TEXT NOT NULL,
        origin TEXT NOT NULL CHECK (origin IN ('web', 'api', 'user', 'data')),
        status INTEGER,
        title TEXT,
        text TEXT NOT NULL,
        links_json TEXT NOT NULL,
        flags_json TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        fetched_at TEXT NOT NULL
    );

    CREATE TABLE competition_facts (
        id INTEGER PRIMARY KEY,
        field TEXT NOT NULL,
        value_json TEXT NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('found', 'inferred', 'user')),
        source_url TEXT NOT NULL,
        quote TEXT NOT NULL,
        recorded_by TEXT NOT NULL,
        recorded_at TEXT NOT NULL
    );
    CREATE TRIGGER facts_append_only_update BEFORE UPDATE ON competition_facts
    BEGIN
        SELECT RAISE(ABORT, 'facts are append-only: record a correction');
    END;

    CREATE TABLE compliance_log (
        id INTEGER PRIMARY KEY,
        item TEXT NOT NULL,
        justification TEXT NOT NULL,
        source TEXT,
        at TEXT NOT NULL
    );

    CREATE TABLE user_actions (
        id INTEGER PRIMARY KEY,
        kind TEXT NOT NULL,
        message TEXT NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('open', 'done')),
        created_at TEXT NOT NULL,
        resolved_at TEXT
    );
    """,
    # v4 — campaigns, approvals given out of band, declared submissions
    """
    CREATE TABLE campaigns (
        id INTEGER PRIMARY KEY,
        mode TEXT NOT NULL CHECK (mode IN ('interactive', 'unattended')),
        driver TEXT NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('running', 'paused', 'waiting_user', 'stopped', 'finished')),
        budget_hours REAL NOT NULL,
        max_experiments INTEGER NOT NULL,
        started_at TEXT NOT NULL,
        ended_at TEXT,
        reason TEXT,
        runner_pid INTEGER,
        runner_created INTEGER
    );

    CREATE TABLE approval_requests (
        id INTEGER PRIMARY KEY,
        action TEXT NOT NULL,
        target TEXT NOT NULL,
        details_json TEXT NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('pending', 'approved', 'refused', 'used')),
        created_at TEXT NOT NULL,
        decided_at TEXT
    );

    CREATE TABLE submissions (
        id INTEGER PRIMARY KEY,
        experiment_id TEXT NOT NULL REFERENCES experiments(id),
        file_sha256 TEXT,
        submitted_at TEXT NOT NULL,
        public_score REAL,
        note TEXT
    );
    """,
]
