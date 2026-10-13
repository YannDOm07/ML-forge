"""Command-line interface (French). Usage: python -m forge <commande>."""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from logging.handlers import RotatingFileHandler
from pathlib import Path

from forge import __version__
from forge.backup import backup_global, backup_project, restore_project
from forge.config import DEFAULT_ROOT, SETTINGS, load_config, resolve_config_path, write_default_config
from forge.datasources import check_data_sources
from forge.errors import ForgeError
from forge.experiments import get_experiment, list_experiments
from forge.guardrails import ApprovalRequest
from forge.projects import archive_project, create_project, delete_project, list_projects, open_project
from forge.system import ML_MODULES, check_ml_stack, detect_hardware, hardware_warnings, prudent_settings
from forge.workspace import Workspace, disk_usage, open_workspace


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):  # a non-UTF-8 console must not crash on « Δ » or « ± »
            stream.reconfigure(errors="replace")
    args = _parser().parse_args(argv)
    try:
        if args.command == "init":
            return _cmd_init(args)
        workspace = open_workspace(load_config(args.config))
        _setup_logging(workspace)
        return COMMANDS[args.command](workspace, args)
    except ForgeError as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m forge", description=f"ML Forge {__version__}")
    parser.add_argument("--config", type=Path, help="fichier de configuration (défaut : config/mlforge.toml)")
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="créer la configuration et la racine des données")
    init.add_argument("--root", type=Path, default=DEFAULT_ROOT, help=f"racine des données (défaut : {DEFAULT_ROOT})")
    config = sub.add_parser("config", help="afficher ou vérifier la configuration")
    config.add_argument("action", choices=("show", "check"))
    create = sub.add_parser("create", help="créer un projet")
    create.add_argument("name")
    create.add_argument("--url", help="lien de la compétition")
    listing = sub.add_parser("list", help="lister les projets")
    listing.add_argument("--all", action="store_true", help="inclure les projets archivés")
    status = sub.add_parser("status", aliases=["open"], help="état d'un projet")
    status.add_argument("project_id")
    sub.add_parser("archive", help="archiver un projet (sans suppression)").add_argument("project_id")
    sub.add_parser("delete", help="supprimer un projet (accord explicite requis)").add_argument("project_id")
    backup = sub.add_parser("backup", help="sauvegarder un projet, ou tout si aucun identifiant")
    backup.add_argument("project_id", nargs="?")
    sub.add_parser("restore", help="restaurer un projet depuis un zip").add_argument("archive", type=Path)
    sub.add_parser("disk", help="espace occupé")
    sub.add_parser("history", help="historique des expériences d'un projet").add_argument("project_id")
    sub.add_parser("best", help="meilleur candidat d'un projet").add_argument("project_id")
    compare = sub.add_parser("compare", help="comparer deux expériences du même protocole")
    compare.add_argument("project_id")
    compare.add_argument("first")
    compare.add_argument("second")
    sub.add_parser("demo", help="démonstration reproductible sur une compétition synthétique")
    intake = sub.add_parser("intake", help="lire la compétition et construire sa fiche")
    intake.add_argument("project_id")
    intake.add_argument("--url", help="lien (défaut : celui du projet)")
    sub.add_parser("card", help="fiche de compétition et informations manquantes").add_argument("project_id")
    paste = sub.add_parser("paste", help="fournir le texte d'une page illisible (fichier texte UTF-8)")
    paste.add_argument("project_id")
    paste.add_argument("file", type=Path)
    fact = sub.add_parser("fact", help="enregistrer un fait cité (la citation doit figurer dans la source)")
    fact.add_argument("project_id")
    fact.add_argument("field")
    fact.add_argument("value")
    fact.add_argument("--source", required=True, help="URL lue ou user://texte-colle/N")
    fact.add_argument("--quote", required=True, help="passage exact de la source")
    sub.add_parser("acquire", help="télécharger les données officielles").add_argument("project_id")
    sub.add_parser("inbox", help="enregistrer les fichiers déposés dans inbox/").add_argument("project_id")
    campaign = sub.add_parser("campaign", help="campagne d'expériences : start, stop, resume, status")
    campaign.add_argument("action", choices=("start", "stop", "resume", "status"))
    campaign.add_argument("project_id")
    campaign.add_argument("--driver", choices=("builtin", "opencode"), default="builtin",
                          help="builtin : plan de référence sans LLM ; opencode : l'agent LLM")
    campaign.add_argument("--interactive", action="store_true", help="budget interactif au lieu du budget de nuit")
    campaign.add_argument("--attach", help="URL d'un serveur « opencode serve » déjà lancé")
    sub.add_parser("finalize", help="générer les livrables exigés").add_argument("project_id")
    sub.add_parser("approvals", help="accords en attente").add_argument("project_id")
    for verb, text in (("approve", "accorder"), ("refuse", "refuser")):
        answer = sub.add_parser(verb, help=f"{text} une demande en attente")
        answer.add_argument("project_id")
        answer.add_argument("request_id", type=int)
    submissions = sub.add_parser("submissions", help="plan de soumission ; --declare pour noter un dépôt")
    submissions.add_argument("project_id")
    submissions.add_argument("--declare", metavar="EXPERIENCE", help="expérience soumise par vous")
    submissions.add_argument("--score", type=float, help="score public obtenu (information seulement)")
    sub.add_parser("memory", help="leçons mesurées et calibration pour un projet").add_argument("project_id")
    opencode = sub.add_parser("setup-opencode", help="écrire opencode.json (serveur MCP et permissions)")
    opencode.add_argument("--docker", metavar="IMAGE", help="serveur MCP lancé dans l'image Docker (ex. mlforge:1.0.0)")
    ui = sub.add_parser("ui", help="ouvrir l'interface locale en lecture seule")
    ui.add_argument("--project", help="projet affiché au démarrage")
    ui.add_argument("--port", type=_port_number, default=8765, help="port local (défaut : 8765)")
    web = sub.add_parser("web-server", help="lancer l'interface web (dashboard)")
    web.add_argument("--host", default="127.0.0.1", help="adresse d'écoute (défaut: 127.0.0.1)")
    web.add_argument("--port", type=int, default=8080, help="port d'écoute (défaut: 8080)")
    web.add_argument("--reload", action="store_true", help="rechargement automatique (dev)")
    return parser


def _setup_logging(workspace: Workspace) -> None:
    workspace.logs_dir.mkdir(exist_ok=True)
    handler = RotatingFileHandler(workspace.logs_dir / "forge.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s : %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)


def _cmd_init(args: argparse.Namespace) -> int:
    path = resolve_config_path(args.config)
    root = args.root.resolve()
    config = write_default_config(path, root, prudent_settings(detect_hardware(root)))
    open_workspace(config)
    print(f"Configuration créée : {path}\nRacine des données : {config.root}")
    return 0


def _cmd_config(workspace: Workspace, args: argparse.Namespace) -> int:
    config = workspace.config
    if args.action == "show":
        print(f"Fichier : {config.source}\nroot = {config.root}")
        for section, settings in SETTINGS.items():
            print(f"[{section}]")
            for key in settings:
                print(f"  {key} = {config.get(section, key)}")
        return 0
    hardware = detect_hardware(config.root)
    gpus = ", ".join(f"{gpu.name} ({gpu.memory_gb} Go)" for gpu in hardware.gpus) or "aucun GPU NVIDIA détecté"
    print(f"Configuration valide : {config.source}")
    print(f"Matériel : {hardware.cpu_threads} threads, {hardware.ram_gb} Go de RAM, {gpus}")
    print(f"Espace libre : {hardware.free_disk_gb} Go")
    warnings = hardware_warnings(
        hardware, config.get("sandbox", "memory_limit_gb"), config.get("acquisition", "min_free_disk_gb")
    )
    stack = check_ml_stack(config.get("sandbox", "python") or sys.executable)
    for module, state in stack.items():
        print(f"  {module:<18} {state}")
        if module in ML_MODULES and state != "ok":
            warnings.append(f"{module} indisponible ({state}). Si Windows indique « Application Control », "
                            "Smart App Control bloque cette version : installez une version plus ancienne.")
    for warning in warnings:
        print(f"Attention : {warning}")
    return 1 if warnings else 0


def _cmd_create(workspace: Workspace, args: argparse.Namespace) -> int:
    project = create_project(workspace, args.name, args.url)
    print(f"Projet créé : {project.id}\nDossier : {project.path}")
    return 0


def _cmd_list(workspace: Workspace, args: argparse.Namespace) -> int:
    projects = list_projects(workspace, include_archived=args.all)
    if not projects:
        print("Aucun projet.")
    for project in projects:
        archived = " [archivé]" if project.status == "archived" else ""
        print(f"{project.id}  {project.name}{archived}  ({project.created_at})")
    return 0


def _cmd_status(workspace: Workspace, args: argparse.Namespace) -> int:
    project = open_project(workspace, args.project_id)
    print(f"Projet : {project.name} ({project.id}) — {project.status}")
    print(f"Compétition : {project.competition_url or 'non renseignée'}")
    counts = Counter(experiment.status for experiment in list_experiments(project.db_path))
    print("Expériences : " + (", ".join(f"{n} {status}" for status, n in sorted(counts.items())) or "aucune"))
    sources = check_data_sources(project.db_path)
    problems = [source for source in sources if source.state != "ok"]
    print(f"Données : {len(sources)} source(s), {len(problems)} problème(s)")
    for source in problems:
        print(f"  {source.state} : {source.path}")
    from forge.actions import open_actions

    for action in open_actions(project):
        print(f"À faire ({action['kind']}) : {action['message']}")
    return 1 if problems else 0


def _cmd_archive(workspace: Workspace, args: argparse.Namespace) -> int:
    archive_project(workspace, args.project_id)
    print(f"Projet archivé (rien n'a été supprimé) : {args.project_id}")
    return 0


def _cmd_delete(workspace: Workspace, args: argparse.Namespace) -> int:
    backup = delete_project(workspace, args.project_id, terminal_approver, decided_by="user:cli")
    print(f"Projet supprimé. Sauvegarde conservée : {backup}")
    return 0


def terminal_approver(request: ApprovalRequest) -> bool:
    print(f"Accord requis pour : {request.describe()}")
    if not sys.stdin.isatty():
        print("Mode non interactif : action refusée.")
        return False
    return input("Tapez « oui » pour confirmer : ").strip().lower() == "oui"


def _cmd_backup(workspace: Workspace, args: argparse.Namespace) -> int:
    if args.project_id:
        print(f"Sauvegarde : {backup_project(workspace, open_project(workspace, args.project_id))}")
        return 0
    print(f"Base globale : {backup_global(workspace)}")
    for project in list_projects(workspace, include_archived=True):
        print(f"Projet {project.id} : {backup_project(workspace, open_project(workspace, project.id))}")
    return 0


def _cmd_restore(workspace: Workspace, args: argparse.Namespace) -> int:
    project = restore_project(workspace, args.archive)
    print(f"Projet restauré : {project.id} ({project.path})")
    return 0


def _cmd_disk(workspace: Workspace, args: argparse.Namespace) -> int:
    usage = disk_usage(workspace)
    for area, size in usage.items():
        print(f"{size / 1024**2:10.1f} Mo  {area}")
    print(f"{sum(usage.values()) / 1024**2:10.1f} Mo  total ({workspace.root})")
    return 0


def _cmd_history(workspace: Workspace, args: argparse.Namespace) -> int:
    from forge.report import decisions_by_experiment

    project = open_project(workspace, args.project_id)
    decisions = decisions_by_experiment(project)
    experiments = list_experiments(project.db_path)
    if not experiments:
        print("Aucune expérience.")
    for experiment in experiments:
        decision = decisions.get(experiment.id)
        result = f"{decision['metric_mean']:.5f} ± {decision['metric_std']:.5f}  {decision['status']}" if decision else ""
        print(f"{experiment.id}  {experiment.status:<11} {experiment.method}  {result}")
    return 0


def _cmd_best(workspace: Workspace, args: argparse.Namespace) -> int:
    from forge.runner import current_best, latest_protocol

    project = open_project(workspace, args.project_id)
    protocol = latest_protocol(project)
    best = current_best(project, protocol.id) if protocol else None
    if best is None:
        print("Aucun candidat : exécutez d'abord une baseline.")
        return 1
    experiment = get_experiment(project.db_path, best)
    print(f"Meilleur candidat : {best} — {experiment.method} (protocole {protocol.id}, {protocol.task.metric})")
    print(f"Hypothèse : {experiment.hypothesis}")
    return 0


def _cmd_compare(workspace: Workspace, args: argparse.Namespace) -> int:
    from forge.runner import compare

    result = compare(open_project(workspace, args.project_id), args.first, args.second,
                     workspace.config.get("evaluation", "acceptance_std_factor"))
    for experiment_id in (args.first, args.second):
        folds = ", ".join(f"{value:.4f}" for value in result[experiment_id]["folds"])
        print(f"{experiment_id} : {result['metric']} = {result[experiment_id]['mean']:.5f}  [{folds}]")
    print(result["verdict"])
    return 0


def _cmd_demo(workspace: Workspace, args: argparse.Namespace) -> int:
    from forge.db import timestamp
    from forge.demo import run_demo

    project, outcomes, submission = run_demo(workspace, workspace.data_dir / f"demo-{timestamp()}")
    for outcome in outcomes:
        print(f"{outcome.experiment_id}  {outcome.status}  {outcome.decision}  "
              f"{outcome.metric_mean:.5f} ± {outcome.metric_std:.5f}  {outcome.message}")
    print(f"Projet : {project.id}")
    print(f"Rapport : {project.path / 'report.md'}")
    print(f"Soumission vérifiée : {submission}")
    return 0


def _print_card(card: dict) -> None:
    from forge.intake import FIELDS, questions

    print(f"Plateforme : {card['platform']} ({card['slug']}) — prise en charge : {card['support_level']}")
    for name, spec in FIELDS.items():
        value = card["fields"][name]
        facts = value if isinstance(value, list) else [value] if value else []
        shown = "; ".join(f"{f['value']} [{f['status']}]" for f in facts) or "—"
        print(f"  {spec.label:<38} {shown}")
    for question in questions(card):
        print(f"À fournir : {question}")


def _cmd_intake(workspace: Workspace, args: argparse.Namespace) -> int:
    from forge.intake import read_competition

    report = read_competition(workspace, open_project(workspace, args.project_id), args.url)
    print(f"{len(report.pages)} page(s) lue(s) : " + ", ".join(report.pages))
    for url, sentence in report.flags:
        print(f"Instruction ignorée ({url}) : {sentence}")
    for note in report.notes:
        print(f"Note : {note}")
    _print_card(report.card)
    return 1 if report.card["critical_unknowns"] else 0


def _cmd_card(workspace: Workspace, args: argparse.Namespace) -> int:
    from forge.intake import competition_card

    card = competition_card(open_project(workspace, args.project_id))
    _print_card(card)
    return 1 if card["critical_unknowns"] else 0


def _cmd_paste(workspace: Workspace, args: argparse.Namespace) -> int:
    from forge.intake import competition_card, extract_facts, paste_text

    project = open_project(workspace, args.project_id)
    url = paste_text(project, args.file.read_text(encoding="utf-8"))
    extract_facts(project)
    print(f"Texte enregistré comme source {url}.")
    _print_card(competition_card(project))
    return 0


def _cmd_fact(workspace: Workspace, args: argparse.Namespace) -> int:
    from forge.intake import record_fact

    fact = record_fact(open_project(workspace, args.project_id), args.field, args.value, args.source, args.quote,
                       "user:cli")
    print(f"Fait enregistré : {fact['field']} = {fact['value']} [{fact['status']}]")
    return 0


def _print_acquisition(result) -> int:
    print(result.message)
    for path in result.files:
        print(f"  {path}")
    return 0 if result.status == "done" else 1


def _cmd_acquire(workspace: Workspace, args: argparse.Namespace) -> int:
    from forge.acquisition import acquire_official

    return _print_acquisition(acquire_official(workspace, open_project(workspace, args.project_id)))


def _cmd_inbox(workspace: Workspace, args: argparse.Namespace) -> int:
    from forge.acquisition import ingest_inbox

    return _print_acquisition(ingest_inbox(workspace, open_project(workspace, args.project_id)))


def _cmd_campaign(workspace: Workspace, args: argparse.Namespace) -> int:
    from forge.campaign import BuiltinDriver, OpencodeDriver, active_campaign, request_stop, run_campaign

    project = open_project(workspace, args.project_id)
    if args.action == "stop":
        request_stop(project)
        print("Arrêt demandé : la campagne s'arrêtera à la fin du cycle en cours.")
        return 0
    if args.action == "status":
        campaign = active_campaign(project)
        print(campaign if campaign else "Aucune campagne active.")
        return 0
    driver = OpencodeDriver(attach=args.attach) if args.driver == "opencode" else BuiltinDriver()
    mode = "interactive" if args.interactive else "unattended"
    campaign = run_campaign(workspace, project, driver, mode)
    print(f"Campagne {campaign['id']} : {campaign['status']} — {campaign['reason']}")
    return 0 if campaign["status"] == "finished" else 1


def _cmd_finalize(workspace: Workspace, args: argparse.Namespace) -> int:
    from forge.deliverables import finalize

    result = finalize(workspace, open_project(workspace, args.project_id))
    print(result.summary)
    for path in result.files:
        print(f"  {path}")
    for problem in result.problems:
        print(f"Problème : {problem}")
    return 0 if not result.problems else 1


def _cmd_approvals(workspace: Workspace, args: argparse.Namespace) -> int:
    from forge.approvals import pending_requests

    requests = pending_requests(open_project(workspace, args.project_id))
    if not requests:
        print("Aucune demande en attente.")
    for request in requests:
        print(f"n° {request['id']} — {request['action']} : {request['target']}")
        for key, value in request["details"].items():
            print(f"    {key} : {value}")
    return 0


def _cmd_answer(workspace: Workspace, args: argparse.Namespace) -> int:
    from forge.approvals import answer_request

    approved = args.command == "approve"
    answer_request(open_project(workspace, args.project_id), args.request_id, approved)
    print(f"Demande n° {args.request_id} : {'accordée' if approved else 'refusée'}.")
    return 0


def _cmd_submissions(workspace: Workspace, args: argparse.Namespace) -> int:
    from forge.deliverables import declare_submission, submission_plan

    project = open_project(workspace, args.project_id)
    if args.declare:
        declare_submission(project, args.declare, args.score)
        print(f"Soumission de {args.declare} enregistrée.")
    print(submission_plan(project)["recommandation"])
    return 0


def _cmd_memory(workspace: Workspace, args: argparse.Namespace) -> int:
    import json

    from forge.knowledge import memory_digest

    print(json.dumps(memory_digest(workspace, open_project(workspace, args.project_id)), ensure_ascii=False, indent=1))
    return 0


def _cmd_setup_opencode(workspace: Workspace, args: argparse.Namespace) -> int:
    from forge.opencode_setup import write_opencode_config

    path = write_opencode_config(workspace, docker_image=args.docker)
    print(f"Configuration opencode écrite : {path}")
    return 0


def _cmd_web_server(workspace: Workspace, args: argparse.Namespace) -> int:
    import os
    os.environ["MLFORGE_CONFIG"] = str(workspace.config.source)
    from forge.web_server import main as web_main
    return web_main(["--host", args.host, "--port", str(args.port)] + (["--reload"] if args.reload else []))


def _cmd_ui(workspace: Workspace, args: argparse.Namespace) -> int:
    from forge.ui.server import serve

    serve(workspace, args.project, args.port)
    return 0


def _port_number(value: str) -> int:
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("le port doit être un entier") from exc
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("le port doit être compris entre 1 et 65535")
    return port


COMMANDS = {
    "config": _cmd_config,
    "create": _cmd_create,
    "list": _cmd_list,
    "status": _cmd_status,
    "open": _cmd_status,
    "archive": _cmd_archive,
    "delete": _cmd_delete,
    "backup": _cmd_backup,
    "restore": _cmd_restore,
    "disk": _cmd_disk,
    "history": _cmd_history,
    "best": _cmd_best,
    "compare": _cmd_compare,
    "demo": _cmd_demo,
    "intake": _cmd_intake,
    "card": _cmd_card,
    "paste": _cmd_paste,
    "fact": _cmd_fact,
    "acquire": _cmd_acquire,
    "inbox": _cmd_inbox,
    "campaign": _cmd_campaign,
    "finalize": _cmd_finalize,
    "approvals": _cmd_approvals,
    "approve": _cmd_answer,
    "refuse": _cmd_answer,
    "submissions": _cmd_submissions,
    "memory": _cmd_memory,
    "setup-opencode": _cmd_setup_opencode,
    "ui": _cmd_ui,
    "web-server": _cmd_web_server,
}
