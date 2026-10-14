"""View model: transforms raw DB rows into display objects + adaptation rules (§7.2).

Pure functions, no acceptance logic, deterministic adaptation.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from forge.task import METRICS


# ---------- Helpers ----------

def _parse_json(value: str | None) -> Any:
    if value is None:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _fmt_number(value: float | None, decimals: int = 5) -> str:
    if value is None:
        return "—"
    return f"{value:.{decimals}f}"


def _fmt_delta(delta: float | None) -> str:
    if delta is None:
        return "—"
    sign = "+" if delta > 0 else ""
    return f"{sign}{_fmt_number(delta)}"


def _status_tag(status: str) -> tuple[str, str]:
    """Return (class, label) for status tag."""
    tags = {
        "completed": ("t-ok", "TERMINÉE"),
        "running": ("t-ok", "EN COURS"),
        "pending": ("t-no", "EN ATTENTE"),
        "failed": ("t-bad", "ÉCHOUÉE"),
        "interrupted": ("t-bad", "INTERROMPUE"),
        "registered": ("t-no", "ENREGISTRÉE"),
        "accepted": ("t-ok", "ACCEPTÉE"),
        "rejected": ("t-bad", "REJETÉE"),
        "not_significant": ("t-no", "NON SIGNIF."),
        "baseline": ("t-ok", "BASELINE"),
        "running": ("t-ok", "ACTIVE"),
        "paused": ("t-no", "EN PAUSE"),
        "waiting_user": ("t-bad", "ATTENTE USER"),
        "stopped": ("t-no", "ARRÊTÉE"),
        "finished": ("t-ok", "TERMINÉE"),
        "open": ("t-bad", "OUVERT"),
        "done": ("t-ok", "FAIT"),
        "approved": ("t-ok", "ACCORDÉ"),
        "refused": ("t-bad", "REFUSÉ"),
        "used": ("t-no", "UTILISÉ"),
        "pending": ("t-bad", "EN ATTENTE"),
        "ok": ("t-ok", "OK"),
        "warning": ("t-no", "ATTENTION"),
        "error": ("t-bad", "ERREUR"),
        "missing": ("t-bad", "MANQUANT"),
        "inferred": ("t-no", "DÉDUIT"),
        "found": ("t-ok", "TROUVÉ"),
        "user": ("t-ok", "UTILISATEUR"),
    }
    return tags.get(status, ("t-no", status.upper()))


def _empty_state(message: str) -> dict[str, Any]:
    return {"empty": True, "message": message}


# ---------- Adaptation rules (§7.2) ----------

def adapt_overview(project: dict, campaign: dict | None, protocol: dict | None,
                   profile: dict | None, card: dict, best: dict | None,
                   recent: list[dict], rejected: list[dict],
                   actions: list[dict], approvals: list[dict],
                   memory: dict) -> dict:
    """Build overview screen data (screen 01)."""
    # Surtitre
    platform = card.get("platform", "inconnu")
    problem_type = fact_value(card, "problem_type", "inconnu")
    support = card.get("support_level", "inconnu")
    subtitle = f"{platform} · {problem_type} · {support}"

    # Campaign status
    campaign_status = campaign["status"] if campaign else "aucune"
    campaign_tag = _status_tag(campaign_status)

    # Best score
    best_score = None
    best_gain = None
    if best and best.get("score") is not None:
        best_score = _fmt_number(best["score"])
        if best.get("delta") is not None:
            best_gain = _fmt_delta(best["delta"])

    # Experiments count
    total_exp = len(recent) + len(rejected)  # approximation
    accepted = sum(1 for e in recent if e.get("decision") == "accepted")
    rejected_count = sum(1 for e in recent if e.get("decision") in ("rejected", "not_significant"))
    max_exp = campaign["max_experiments"] if campaign else 40

    # Validation reliability
    reliability = protocol.get("reliability") if protocol else None
    rel_text = "—"
    rel_tag = ("t-no", "INCONNU")
    if reliability:
        rel_text = reliability.get("level", "—")
        if reliability.get("level") == "fragile":
            rel_tag = ("t-bad", "FRAGILE")
        elif reliability.get("level") == "ok":
            rel_tag = ("t-ok", "OK")
        else:
            rel_tag = ("t-no", rel_text.upper())

    # Deadline & submissions
    deadline = fact_value(card, "deadline")
    days_left = "—"
    if deadline:
        try:
            dt = datetime.fromisoformat(deadline.replace("Z", "+00:00"))
            days_left = str((dt - datetime.now(timezone.utc)).days)
        except (ValueError, AttributeError):
            pass

    daily_limit = fact_value(card, "rules.daily_submission_limit")
    submissions_today = 0  # TODO: count from submissions table

    # Chart data (gain per experiment)
    chart_data = []
    for exp in recent[-20:]:  # last 20
        if exp.get("delta") is not None:
            decision = exp.get("decision", "pending")
            chart_data.append({
                "id": exp["id"],
                "delta": exp["delta"],
                "decision": decision,
                "tag": _status_tag(decision)[0],
            })

    # Deliverables checklist
    deliverables = []
    for fact in card.get("fields", {}).get("deliverables", []):
        if isinstance(fact, dict):
            deliverables.append({
                "name": fact["value"],
                "status": fact["status"],
                "tag": _status_tag(fact["status"])[0],
            })

    # Pending hypotheses
    pending_hypotheses = []
    for exp in recent:
        if exp.get("status") in ("pending", "running"):
            pending_hypotheses.append({
                "id": exp["id"],
                "hypothesis": exp["hypothesis"][:80],
                "priority": "—",  # TODO: compute from estimate
            })

    # Authorization note
    auth_items = []
    for a in actions:
        auth_items.append({"kind": a["kind"], "message": a["message"]})
    for a in approvals:
        auth_items.append({"kind": "approval", "message": f"{a['action']} {a['target']}"})

    return {
        "subtitle": subtitle,
        "campaign_status": campaign_status,
        "campaign_tag": campaign_tag,
        "best_score": best_score,
        "best_gain": best_gain,
        "exp_total": total_exp,
        "exp_max": max_exp,
        "exp_accepted": accepted,
        "exp_rejected": rejected_count,
        "reliability_text": rel_text,
        "reliability_tag": rel_tag,
        "deadline_days": days_left,
        "submissions_today": submissions_today,
        "daily_limit": daily_limit or "?",
        "chart_data": chart_data,
        "deliverables": deliverables,
        "recent_experiments": recent[:5],
        "pending_hypotheses": pending_hypotheses,
        "authorization_items": auth_items,
        "memory_summary": memory.get("summary", "—"),
    }


def adapt_campaign(campaign: dict | None, project: dict, protocol: dict | None,
                   card: dict, workspace_config: dict) -> dict:
    """Build campaign screen data (screen 02)."""
    if not campaign:
        return _empty_state("Aucune campagne. Lancez-en une depuis la ligne de commande.")

    # Current phase from next_step logic
    phase = "experiment"  # default
    phase_label = "Expérimentation"

    # Budgets
    budget_h = campaign["budget_hours"]
    max_exp = campaign["max_experiments"]
    patience = workspace_config.get("campaign", {}).get("patience", 5)

    # Stop conditions (from stop_reason)
    stop_conditions = []
    # Time
    try:
        started = datetime.fromisoformat(campaign["started_at"].replace("Z", "+00:00"))
        elapsed_h = (datetime.now(timezone.utc) - started).total_seconds() / 3600
        stop_conditions.append({
            "name": "Budget temps",
            "current": f"{elapsed_h:.1f} h",
            "limit": f"{budget_h} h",
            "critical": elapsed_h >= budget_h * 0.9,
        })
    except (ValueError, AttributeError):
        pass

    # Experiments count
    # TODO: get actual count from DB
    stop_conditions.append({
        "name": "Max expériences",
        "current": "0",
        "limit": str(max_exp),
        "critical": False,
    })

    # Patience
    stop_conditions.append({
        "name": "Sans gain (patience)",
        "current": "0",
        "limit": str(patience),
        "critical": False,
    })

    # Deadline
    deadline = fact_value(card, "deadline")
    if deadline:
        try:
            dt = datetime.fromisoformat(deadline.replace("Z", "+00:00"))
            remaining_h = (dt - datetime.now(timezone.utc)).total_seconds() / 3600
            finalize_h = workspace_config.get("campaign", {}).get("finalize_hours_before_deadline", 24)
            stop_conditions.append({
                "name": "Finalisation (échéance)",
                "current": f"{remaining_h:.1f} h",
                "limit": f"{finalize_h} h avant",
                "critical": remaining_h <= finalize_h,
            })
        except (ValueError, AttributeError):
            pass

    # Recent events
    events = []  # TODO: from execution_events

    return {
        "campaign": campaign,
        "phase": phase,
        "phase_label": phase_label,
        "stop_conditions": stop_conditions,
        "events": events,
    }


def adapt_experiments(experiments: list[dict], decisions: list[dict],
                      protocol_id: str | None) -> dict:
    """Build experiments screen data (screen 03)."""
    # Map decisions by experiment_id
    dec_map = {d["experiment_id"]: d for d in decisions}

    rows = []
    for exp in experiments:
        dec = dec_map.get(exp["id"])
        rows.append({
            "id": exp["id"],
            "status": exp["status"],
            "status_tag": _status_tag(exp["status"]),
            "method": exp["method"],
            "hypothesis": exp["hypothesis"][:120],
            "score": _fmt_number(dec["metric_mean"]) if dec else "—",
            "delta": _fmt_delta(dec["delta"]) if dec and dec.get("delta") is not None else "—",
            "decision": dec["status"] if dec else "—",
            "decision_tag": _status_tag(dec["status"]) if dec else ("t-no", "—"),
            "created_at": exp["created_at"],
        })

    return {
        "rows": rows,
        "protocol_id": protocol_id,
    }


def adapt_experiment_detail(exp: dict, dec: dict | None, metrics: list[dict],
                            protocol: dict | None) -> dict:
    """Build single experiment detail (screen 03 click)."""
    card = _parse_json(exp.get("card_json"))

    # Metrics by scope
    folds = [m for m in metrics if m["scope"] == "fold"]
    seeds = [m for m in metrics if m["scope"] == "seed"]
    sims = [m for m in metrics if m["scope"] == "simulation"]

    # Justification with tags
    justification = card.get("justification", []) if card else []
    just_items = []
    for j in justification:
        kind = j.get("kind", "observed_fact")
        kind_labels = {
            "observed_fact": "FAIT OBSERVÉ",
            "domain_knowledge": "CONNAISSANCE MÉTIER",
            "memory": "MÉMOIRE",
            "public_source": "SOURCE PUBLIQUE",
        }
        just_items.append({
            "kind": kind_labels.get(kind, kind.upper()),
            "text": j.get("text", ""),
        })

    # Estimate
    estimate = card.get("estimate", {}) if card else {}

    return {
        "experiment": exp,
        "decision": dec,
        "folds": folds,
        "seeds": seeds,
        "simulations": sims,
        "justification": just_items,
        "estimate": estimate,
        "protocol": protocol,
    }


def adapt_hypotheses(experiments: list[dict], decisions: list[dict]) -> dict:
    """Build hypotheses screen (screen 04)."""
    dec_map = {d["experiment_id"]: d for d in decisions}

    pending = []
    tested = []
    rejected = []

    for exp in experiments:
        dec = dec_map.get(exp["id"])
        card = _parse_json(exp.get("card_json"))
        estimate = card.get("estimate", {}) if card else {}

        # Priority score = gain × prob ÷ cost (penalized by risk)
        gain = estimate.get("expected_gain", 0)
        prob = estimate.get("success_probability", 0)
        cost = estimate.get("cost_minutes", 1)
        risk = estimate.get("risk", "low")
        risk_penalty = {"low": 1.0, "medium": 0.7, "high": 0.4}.get(risk, 1.0)
        priority = (gain * prob / cost) * risk_penalty if cost > 0 else 0

        item = {
            "id": exp["id"],
            "hypothesis": exp["hypothesis"][:120],
            "method": exp["method"],
            "priority": _fmt_number(priority, 4),
            "gain": _fmt_number(gain, 5),
            "prob": _fmt_number(prob, 2),
            "cost": f"{cost} min",
            "risk": risk,
            "status": exp["status"],
            "decision": dec["status"] if dec else "—",
        }

        if exp["status"] in ("pending", "running"):
            pending.append(item)
        elif exp["status"] == "completed":
            if dec and dec["status"] in ("accepted", "baseline"):
                tested.append(item)
            else:
                rejected.append(item)
        else:
            rejected.append(item)

    return {
        "pending": pending,
        "tested": tested,
        "rejected": rejected,
    }


def adapt_data(project_db_path: str, card: dict, data_sources: list[dict],
               profile: dict | None) -> dict:
    """Build data screen (screen 05)."""
    # Competition facts with source
    facts = []
    for name, fact in card.get("fields", {}).items():
        if isinstance(fact, list):
            for f in fact:
                facts.append({
                    "field": name,
                    "value": f["value"],
                    "status": f["status"],
                    "status_tag": _status_tag(f["status"]),
                    "source": f["source_url"],
                    "quote": f["quote"][:200],
                })
        elif fact:
            facts.append({
                "field": name,
                "value": fact["value"],
                "status": fact["status"],
                "status_tag": _status_tag(fact["status"]),
                "source": fact["source_url"],
                "quote": fact["quote"][:200],
            })

    # Data sources
    sources = []
    for ds in data_sources:
        sources.append({
            "role": ds["role"],
            "path": ds["path"],
            "sha256": ds["sha256"][:16] + "...",
            "size_mb": f"{ds['size_bytes'] / 1024 / 1024:.1f}",
            "license": ds["license"] or "—",
            "origin": ds["origin_url"] or "—",
        })

    # Profile summary
    profile_summary = _empty_state("Pas encore de profil. Disponible après l'étape profile.")
    if profile:
        p = _parse_json(profile.get("profile_json"))
        if p:
            profile_summary = {
                "rows": p.get("rows", "—"),
                "features": p.get("features", "—"),
                "target": p.get("target", "—"),
                "kinds": p.get("kinds", {}),
                "error_floor": _fmt_number(p.get("twins", {}).get("error_floor"), 5),
                "leak_hints": p.get("leak_hints", [])[:5],
                "high_missing": [c["name"] for c in p.get("columns", []) if c.get("missing_rate", 0) > 0.2][:10],
            }

    return {
        "facts": facts,
        "sources": sources,
        "profile": profile_summary,
    }


def adapt_validation(protocol: dict | None, experiments: list[dict],
                     decisions: list[dict], metrics: list[dict]) -> dict:
    """Build validation screen (screen 06)."""
    if not protocol:
        return _empty_state("Pas encore de protocole. Disponible après campaign_prepare.")

    spec = _parse_json(protocol.get("spec_json"))
    folds = _parse_json(protocol.get("folds_json"))
    reliability = _parse_json(protocol.get("reliability_json"))

    # Variance between folds/seeds
    dec_map = {d["experiment_id"]: d for d in decisions}
    fold_variance = []
    for exp in experiments:
        dec = dec_map.get(exp["id"])
        if dec:
            exp_metrics = [m for m in metrics if m["experiment_id"] == exp["id"] and m["scope"] == "fold"]
            if exp_metrics:
                vals = [m["value"] for m in exp_metrics]
                mean = sum(vals) / len(vals)
                std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5 if len(vals) > 1 else 0
                fold_variance.append({
                    "exp_id": exp["id"],
                    "method": exp["method"],
                    "folds": len(vals),
                    "mean": _fmt_number(mean),
                    "std": _fmt_number(std, 6),
                    "cv": _fmt_number(std / abs(mean) if mean != 0 else 0, 4),
                })

    # Regime
    regime = "simple"
    if reliability and reliability.get("level") == "fragile":
        regime = "double (validation fragile)"

    return {
        "spec": spec,
        "folds": folds,
        "reliability": reliability,
        "fold_variance": fold_variance,
        "regime": regime,
        "has_simulator": any(m["scope"] == "simulation" for m in metrics),
    }


def adapt_deliverables(card: dict, submissions: list[dict], project_path: str) -> dict:
    """Build deliverables screen (screen 07)."""
    # Checklist from competition facts
    checklist = []
    for fact in card.get("fields", {}).get("deliverables", []):
        if isinstance(fact, dict):
            checklist.append({
                "name": fact["value"],
                "status": fact["status"],
                "status_tag": _status_tag(fact["status"]),
            })

    # Submission files
    sub_files = []
    import os
    sub_dir = os.path.join(project_path, "submissions")
    if os.path.exists(sub_dir):
        for f in sorted(os.listdir(sub_dir)):
            if f.endswith(".csv"):
                sub_files.append(f)

    # Submission history
    history = []
    for s in submissions:
        history.append({
            "experiment_id": s["experiment_id"],
            "file": s["file_sha256"][:16] + "..." if s["file_sha256"] else "—",
            "date": s["submitted_at"],
            "public_score": _fmt_number(s["public_score"]) if s["public_score"] else "—",
            "note": s["note"] or "—",
        })

    # Candidates (from finalize)
    robust = {"id": "—", "method": "—", "reason": "Non calculé"}
    ambitious = {"id": "—", "method": "—", "reason": "Non calculé"}

    return {
        "checklist": checklist,
        "submission_files": sub_files,
        "history": history,
        "robust": robust,
        "ambitious": ambitious,
    }


def adapt_memory(knowledge: list[dict], evidence: list[dict],
                 calibration: list[dict]) -> dict:
    """Build memory screen (screen 08)."""
    # Knowledge items
    items = []
    for k in knowledge:
        items.append({
            "family": k["family"],
            "context": k["context_key"],
            "statement": k["statement"],
            "supporting": k["n_supporting"],
            "contradicting": k["n_contradicting"],
            "projects": k["n_projects"],
            "mean_delta": _fmt_number(k["mean_delta"], 5),
            "confidence": _fmt_number(k["confidence"], 2),
        })

    # Calibration bias
    bias = _empty_state("Pas encore d'historique de calibration.")
    if calibration:
        total = len(calibration)
        over = sum(1 for c in calibration if c["expected_gain"] > c["observed_gain"])
        under = sum(1 for c in calibration if c["expected_gain"] < c["observed_gain"])
        avg_expected = sum(c["expected_gain"] for c in calibration) / total
        avg_observed = sum(c["observed_gain"] for c in calibration) / total
        bias = {
            "total": total,
            "overestimated": over,
            "underestimated": under,
            "avg_expected": _fmt_number(avg_expected, 5),
            "avg_observed": _fmt_number(avg_observed, 5),
            "bias": _fmt_number(avg_expected - avg_observed, 5),
            "details": calibration[:20],
        }

    return {
        "knowledge": items,
        "calibration": bias,
    }


def adapt_log(execution_events: list[dict], network_log: list[dict],
              compliance_log: list[dict], config_history: list[dict]) -> dict:
    """Build log screen (screen 09)."""
    return {
        "execution_events": [
            {
                "id": e["id"],
                "exp_id": e["experiment_id"] or "—",
                "kind": e["kind"],
                "message": e["message"][:200],
                "at": e["at"],
            }
            for e in execution_events
        ],
        "network_log": [
            {
                "id": n["id"],
                "project": n["project_id"] or "global",
                "method": n["method"],
                "url": n["url"][:80],
                "purpose": n["purpose"],
                "status": n["status"],
                "bytes": n["bytes"],
                "at": n["at"],
            }
            for n in network_log
        ],
        "compliance_log": [
            {
                "id": c["id"],
                "item": c["item"],
                "justification": c["justification"][:200],
                "at": c["at"],
            }
            for c in compliance_log
        ],
        "config_history": [
            {
                "id": c["id"],
                "hash": c["config_hash"][:12],
                "at": c["recorded_at"],
            }
            for c in config_history
        ],
    }


# ---------- Small helpers ----------

def fact_value(card: dict, name: str, default=None):
    """Extract fact value from competition card."""
    fields = card.get("fields", {})
    fact = fields.get(name)
    if isinstance(fact, dict):
        return fact.get("value", default)
    if isinstance(fact, list) and fact:
        return fact[0].get("value", default)
    return default


def metric_direction(metric: str) -> tuple[str, str]:
    """Return (arrow, label) for metric direction."""
    if metric in METRICS:
        higher = METRICS[metric][0]
        if higher:
            return ("↑", "plus haut = mieux")
        else:
            return ("↓", "plus bas = mieux")
    return ("?", "inconnu")


def format_bytes(bytes_val: int) -> str:
    for unit in ["B", "Ko", "Mo", "Go"]:
        if bytes_val < 1024:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.1f} To"


def _protocol_view(protocol: dict | None, metrics: list[dict]) -> dict[str, Any]:
    if not protocol:
        return {"available": False, "task": {}, "metrics": [], "available_metrics": []}

    spec = _parse_json(protocol.get("spec_json")) or {}
    raw_task = spec.get("task", {})
    task = {
        key: raw_task[key]
        for key in ("problem_type", "target", "metric", "id_column", "group_column", "time_column")
        if key in raw_task
    }
    folds = _parse_json(protocol.get("folds_json")) or []
    reliability = _parse_json(protocol.get("reliability_json")) or {}
    groups: dict[tuple[str, str, str], list[float]] = {}
    for item in metrics:
        key = (item["experiment_id"], item["scope"], item["name"])
        groups.setdefault(key, []).append(float(item["value"]))
    metric_rows = []
    for (experiment_id, scope, name), values in groups.items():
        mean = sum(values) / len(values)
        std = (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5
        metric_rows.append({
            "experiment_id": experiment_id,
            "scope": scope,
            "name": name,
            "count": len(values),
            "mean": mean,
            "std": std,
        })

    return {
        "available": True,
        "id": protocol.get("id"),
        "task": task,
        "folds": len(folds) if isinstance(folds, list) else folds,
        "scheme": spec.get("scheme", "—"),
        "notes": spec.get("notes", []),
        "reliability": reliability,
        "metrics": metric_rows,
        "available_metrics": [
            {"name": name, "higher_is_better": details[0], "needs_probabilities": details[2]}
            for name, details in METRICS.items()
        ],
    }


def _column_profile(profile: dict | None, task: dict) -> dict[str, Any]:
    if not profile:
        return {
            "available": False, "rows": None, "features": None,
            "target": task.get("target"), "columns": [], "twins": {},
            "target_stats": None, "leak_hints": [],
        }

    target_stats = profile.get("target") or {}
    classes = target_stats.get("classes")
    safe_target_stats = {
        "missing": target_stats.get("missing"),
        "class_count": len(classes) if isinstance(classes, dict) else None,
        "imbalance_ratio": target_stats.get("imbalance_ratio"),
    }
    twins = profile.get("twins") or {}
    return {
        "available": True,
        "rows": profile.get("rows"),
        "features": profile.get("features"),
        "target": task.get("target"),
        "columns": [
            {key: column.get(key) for key in ("name", "kind", "missing_rate", "unique")}
            for column in profile.get("columns", [])
        ],
        "twins": {
            key: twins.get(key)
            for key in ("groups", "rows", "error_floor", "note")
            if key in twins
        },
        "target_stats": safe_target_stats,
        "leak_hints": profile.get("leak_hints", [])[:5],
    }


def build_viewmodel(workspace, project_id: str | None = None) -> dict[str, Any]:
    """Build the complete display-only state consumed by the local UI."""
    from pathlib import Path
    from urllib.parse import urlsplit

    from forge.ui import queries

    def platform_name(url: str | None) -> str:
        parsed = urlsplit(url or "")
        host = parsed.netloc.lower().removeprefix("www.")
        if host == "kaggle.com":
            return "kaggle"
        if host.endswith("zindi.africa"):
            return "zindi"
        if host.endswith("itu.int"):
            return "itu"
        return "generic"

    projects = queries.list_projects(workspace.global_db)
    selected_id = project_id or (projects[0]["id"] if projects else None)
    generated_at = datetime.now(timezone.utc).isoformat()
    empty_panels = {
        "overview": None,
        "campaign": None,
        "experiments": [],
        "hypotheses": [],
        "data": {
            "profile": _column_profile(None, {}),
            "sources": [],
            "facts": [],
        },
        "validation": _protocol_view(None, []),
        "deliverables": {"checklist": [], "wanted": [], "files": [], "submissions": []},
        "memory": {"knowledge": [], "calibration": [], "evidence": []},
        "journal": {"events": [], "network": [], "actions": [], "approvals": [], "compliance": []},
    }
    if selected_id is None:
        return {
            **empty_panels,
            "projects": projects,
            "selected_project": None,
            "generated_at": generated_at,
            "focus": {"screen": "overview", "phase": "create", "message": "Créez un projet ML Forge pour commencer."},
        }

    project = queries.get_project(workspace.global_db, selected_id)
    if project is None:
        raise ValueError(f"Projet inconnu : {selected_id}")
    project_db = workspace.projects_dir / selected_id / "project.db"
    experiments = queries.get_experiments(project_db)
    decisions = {item["experiment_id"]: item for item in queries.get_experiment_decisions(project_db)}
    protocol = queries.get_latest_protocol(project_db)
    protocol_spec = _parse_json(protocol.get("spec_json")) if protocol else {}
    protocol_spec = protocol_spec or {}
    task = protocol_spec.get("task", {})
    profile_row = None
    if protocol and protocol_spec.get("data_hash"):
        profile_row = queries.get_latest_data_profile(project_db, protocol_spec["data_hash"])
    raw_profile = _parse_json(profile_row.get("profile_json")) if profile_row else None
    profile = _column_profile(raw_profile, task)
    validation = _protocol_view(protocol, queries.get_all_experiment_metrics(project_db))

    facts = []
    facts_by_name: dict[str, Any] = {}
    for fact in queries.get_competition_facts(project_db):
        value = _parse_json(fact["value_json"])
        is_local = str(fact.get("source_url", "")).startswith("file:")
        if is_local:
            value = "donnée locale masquée"
        item = {
            "field": fact["field"],
            "value": value,
            "status": fact["status"],
            "source": None if is_local else fact.get("source_url"),
        }
        facts.append(item)
        facts_by_name[fact["field"]] = item

    experiment_rows = []
    for experiment in experiments:
        decision = decisions.get(experiment["id"])
        experiment_rows.append({
            "id": experiment["id"],
            "parent_id": experiment["parent_id"],
            "method": experiment["method"],
            "hypothesis": experiment["hypothesis"],
            "status": experiment["status"],
            "created_at": experiment["created_at"],
            "protocol_id": experiment["protocol_id"],
            "decision": ({
                "status": decision["status"],
                "metric_mean": decision["metric_mean"],
                "metric_std": decision["metric_std"],
                "delta": decision["delta"],
            } if decision else None),
        })

    raw_campaign = queries.get_active_campaign(project_db)
    if raw_campaign is None:
        campaigns = queries.get_campaigns(project_db)
        raw_campaign = campaigns[0] if campaigns else None
    campaign = None
    if raw_campaign:
        started_at = raw_campaign.get("started_at")
        try:
            elapsed_hours = max(0.0, (datetime.now(timezone.utc) -
                datetime.fromisoformat(started_at.replace("Z", "+00:00"))).total_seconds() / 3600)
        except (AttributeError, ValueError):
            elapsed_hours = 0.0
        campaign = {
            "id": raw_campaign["id"],
            "status": raw_campaign["status"],
            "active": raw_campaign["status"] in ("running", "paused", "waiting_user"),
            "driver": raw_campaign["driver"],
            "mode": raw_campaign["mode"],
            "started_at": started_at,
            "budget_hours": raw_campaign["budget_hours"],
            "hours_elapsed": elapsed_hours,
            "hours_remaining": max(0.0, raw_campaign["budget_hours"] - elapsed_hours),
            "experiments_used": len(experiments),
            "experiments_max": raw_campaign["max_experiments"],
            "patience": workspace.config.get("campaign", "patience") or 5,
            "reason": raw_campaign.get("reason"),
        }

    accepted = [item for item in experiment_rows if item["decision"] and item["decision"]["status"] in ("accepted", "baseline")]
    best = accepted[-1] if accepted else None
    platform = platform_name(project.get("competition_url"))
    problem_type = task.get("problem_type")
    support_level = "validated" if problem_type in ("binary", "multiclass", "regression") else "experimental"
    overview = {
        "project": {"id": selected_id, "name": project["name"]},
        "platform": platform,
        "problem_type": problem_type,
        "support_level": support_level,
        "campaign": campaign,
        "metric": task.get("metric"),
        "best": best,
        "experiment_counts": {
            "total": len(experiment_rows),
            "accepted": sum(1 for item in experiment_rows if item["decision"] and item["decision"]["status"] in ("accepted", "baseline")),
            "rejected": sum(1 for item in experiment_rows if item["decision"] and item["decision"]["status"] in ("rejected", "not_significant")),
            "running": sum(1 for item in experiment_rows if item["status"] == "running"),
        },
        "experiments_max": campaign["experiments_max"] if campaign else 40,
        "accepted_count": len(accepted),
        "open_actions": [item for item in queries.get_user_actions(project_db) if item["status"] == "open"],
        "pending_approvals": [item for item in queries.get_approval_requests(project_db) if item["status"] == "pending"],
        "deadline": facts_by_name.get("deadline", {}).get("value"),
        "recent_experiments": experiment_rows[-5:],
    }

    hypotheses = []
    for experiment in experiments:
        card = _parse_json(experiment.get("card_json")) or {}
        estimate = card.get("estimate", {})
        decision = decisions.get(experiment["id"])
        lane = "pending" if not decision else ("rejected" if decision["status"] in ("rejected", "not_significant") else "tested")
        hypotheses.append({
            "id": experiment["id"],
            "hypothesis": experiment["hypothesis"][:120],
            "method": experiment["method"],
            "lane": lane,
            "estimate": {
                key: estimate.get(key)
                for key in ("expected_gain", "success_probability", "cost_minutes", "risk")
            },
        })

    deliverable_path = workspace.projects_dir / selected_id / "deliverables"
    files = sorted(path.name for path in deliverable_path.iterdir() if path.is_file()) if deliverable_path.is_dir() else []
    deliverables_wanted = []
    for fact in facts:
        if fact["field"] == "deliverables":
            values = fact["value"] if isinstance(fact["value"], list) else [fact["value"]]
            deliverables_wanted.extend(str(value) for value in values)
    deliverables = {
        "wanted": deliverables_wanted,
        "files": files,
        "checklist": [
            {"kind": "file", "name": name, "required": True, "present": name in files}
            for name in deliverables_wanted
        ],
        "submissions": queries.get_submissions(project_db),
    }

    data = {
        "profile": profile,
        "sources": [
            {
                "name": Path(source["path"]).name,
                "role": source["role"],
                "size_bytes": source["size_bytes"],
                "license": source["license"],
                "sha256": source["sha256"],
            }
            for source in queries.get_data_sources(project_db)
        ],
        "facts": facts,
    }
    memory = {
        "knowledge": queries.get_knowledge_items(workspace.global_db),
        "calibration": queries.get_estimate_calibration(workspace.global_db, selected_id),
        "evidence": queries.get_knowledge_evidence(workspace.global_db, selected_id),
    }
    journal = {
        "events": [
            {"id": item["id"], "experiment_id": item["experiment_id"], "kind": item["kind"],
             "message": item["message"][:200], "at": item["at"]}
            for item in queries.get_execution_events(project_db)
        ],
        "network": queries.get_network_log(workspace.global_db, selected_id),
        "actions": queries.get_user_actions(project_db),
        "approvals": queries.get_approval_requests(project_db),
        "compliance": queries.get_compliance_log(project_db),
    }

    critical_missing = (
        not task.get("problem_type")
        or not task.get("target")
        or not task.get("metric")
        or not facts_by_name.get("submission_format", {}).get("value")
    )
    if critical_missing:
        phase, message = (
            "acquire",
            "Complétez les règles du challenge : type de problème, cible, métrique et format de soumission.",
        )
    elif not data["sources"]:
        phase, message = "inbox", "Ajoutez les fichiers officiels du challenge dans le dossier inbox du projet."
    elif not profile["available"]:
        phase, message = "profile", "Les données sont présentes ; lancez leur profilage avant la campagne."
    elif not validation["available"]:
        phase, message = "campaign_prepare", "Préparez le protocole de validation avant les expériences."
    elif campaign and campaign["active"]:
        phase, message = "experiment", f"Campagne {campaign['status']} : le tableau se met à jour automatiquement."
    else:
        phase, message = "campaign", "Le protocole est disponible ; vous pouvez démarrer une campagne."

    return {
        "projects": projects,
        "selected_project": selected_id,
        "generated_at": generated_at,
        "focus": {"screen": "overview", "phase": phase, "message": message},
        "overview": overview,
        "campaign": campaign,
        "experiments": experiment_rows,
        "hypotheses": hypotheses,
        "data": data,
        "validation": validation,
        "deliverables": deliverables,
        "memory": memory,
        "journal": journal,
    }