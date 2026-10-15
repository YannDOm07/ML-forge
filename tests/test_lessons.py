"""Lessons register: what works or fails, in which context, and why — retrieved by context, never read whole."""

import pytest

from forge.lessons import LessonError, progress_dir, record, render_progress, scoped, search
from forge.mcp_server import Server
from forge.projects import create_project

AUDIO = {"modality": "audio", "problem_type": "regression", "metric": "rmse", "platform": "evalai", "size": "10k-100k"}
TABULAR = {"modality": "tabulaire", "problem_type": "binary", "metric": "roc_auc", "platform": "kaggle", "size": "1k-10k"}


def titles(lessons):
    return [lesson["titre"] for lesson in lessons]


def test_a_lesson_needs_a_reason_and_a_known_step(workspace):
    with pytest.raises(LessonError, match="why"):
        record(workspace, "experimenter", "technique", "Boosting sur descripteurs", "mieux", AUDIO, "marche")
    with pytest.raises(LessonError, match="Étape inconnue"):
        record(workspace, "deployer", "technique", "Boosting sur descripteurs", "Parce que les interactions comptent.", AUDIO)
    lesson = record(workspace, "Expérimenter", "technique", "Boosting sur descripteurs",
                    "Les interactions entre bandes portent l'information.", AUDIO, "marche")
    assert lesson["étape"] == "experimenter" and lesson["résultat"] == "marche"  # accents and case are tolerated


def test_the_same_lesson_is_strengthened_not_duplicated(workspace):
    why = "Une même phrase dans deux plis fait mémoriser la phrase au lieu d'apprendre."
    record(workspace, "valider", "technique", "Plis groupés par phrase", why, AUDIO, "marche", project_id="p1")
    again = record(workspace, "valider", "technique", "Plis  groupés par PHRASE", why, AUDIO, "marche", project_id="p2")
    assert again["vu"] == 2 and again["projets"] == 2
    same_evidence = record(workspace, "experimenter", "technique", "Linéaire", "Les descripteurs sont presque linéaires.",
                           AUDIO, "marche", project_id="p1", experiment_id="E0002", delta=0.05)
    replay = record(workspace, "experimenter", "technique", "Linéaire", "Les descripteurs sont presque linéaires.",
                    AUDIO, "marche", project_id="p1", experiment_id="E0002", delta=0.05)
    assert replay["vu"] == same_evidence["vu"] == 1  # one experiment never counts twice
    contradiction = record(workspace, "experimenter", "technique", "Linéaire", "Les descripteurs sont presque linéaires.",
                           AUDIO, "echoue", project_id="p3", experiment_id="E0004", delta=-0.01)
    assert contradiction["résultat"] == "neutre" and contradiction["vu"] == 2  # mixed evidence is not a rule


def test_search_returns_only_lessons_of_a_close_context(workspace):
    record(workspace, "experimenter", "technique", "Forêts sur descripteurs audio",
           "Aucun gain : le goulot est l'information des descripteurs.", AUDIO, "echoue")
    record(workspace, "experimenter", "technique", "Encodage cible des catégories rares",
           "Les catégories rares bruitent l'encodage one-hot.", TABULAR, "marche")
    record(workspace, "experimenter", "piege", "Score parfait en validation", "Presque toujours une fuite de la cible.",
           scoped(TABULAR, "general"))
    audio = titles(search(workspace, AUDIO, step="experimenter"))
    assert "Forêts sur descripteurs audio" in audio and "Score parfait en validation" in audio
    assert "Encodage cible des catégories rares" not in audio  # another modality never leaks in
    tabular = titles(search(workspace, TABULAR, step="experimenter", limit=50))
    assert "Forêts sur descripteurs audio" not in tabular
    assert tabular[0] == "Encodage cible des catégories rares"  # the exact context first
    assert len(search(workspace, TABULAR, limit=3)) == 3  # a few lessons, never the whole register
    assert titles(search(workspace, TABULAR, query="fuite", step="experimenter"))[0] == "Score parfait en validation"


def test_foundation_lessons_are_loaded_once(workspace):
    first = search(workspace, {"modality": "*"}, step="acquerir", limit=50)
    assert any("étiquettes" in lesson["titre"] for lesson in first)
    record(workspace, "acquerir", "technique", "Autre technique", "Une raison suffisamment longue.", AUDIO)
    assert len(search(workspace, {"modality": "*"}, step="acquerir", limit=50)) == len(first) + 1


def test_progress_files_are_an_index_and_one_file_per_modality(workspace):
    record(workspace, "calculer", "reussite", "Différence d'énergie dans les aigus",
           "La perte auditive retire surtout les hautes fréquences.", AUDIO)
    folder = render_progress(workspace)
    assert folder == progress_dir(workspace)
    index = (folder / "INDEX.md").read_text(encoding="utf-8")
    assert "Ne lisez pas tout" in index and "[audio.md](audio.md)" in index and "[general.md](general.md)" in index
    audio = (folder / "audio.md").read_text(encoding="utf-8")
    assert "## 5. Calculer" in audio and "Différence d'énergie dans les aigus" in audio
    assert "Différence d'énergie dans les aigus" not in (folder / "general.md").read_text(encoding="utf-8")


def test_the_agent_records_and_searches_through_mcp(workspace):
    project = create_project(workspace, "Agent", "https://www.kaggle.com/competitions/flows-cup")
    server = Server(workspace)
    saved = server.lesson_record(project.id, "comprendre", "erreur", "Métrique lue dans un forum au lieu du règlement",
                                 "Le forum citait l'ancienne métrique ; seule la page Evaluation fait foi.",
                                 scope="general")
    assert saved["contexte"] == "général"
    text = server.lessons_search(project.id, step="comprendre", kind="erreur")
    assert "Métrique lue dans un forum" in text and "seule la page Evaluation fait foi" in text
    with pytest.raises(LessonError):
        server.lesson_record(project.id, "comprendre", "astuce", "Titre correct", "Une raison assez longue ici.")
