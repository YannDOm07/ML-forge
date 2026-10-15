"""Test 6: a missing or modified dataset is detected; nothing is copied."""

import os

import pytest

from forge.datasources import check_data_sources, register_data_source, register_data_sources
from forge.errors import DataSourceError
from forge.projects import create_project


@pytest.fixture
def setup(workspace, tmp_path):
    project = create_project(workspace, "Données")
    data = tmp_path / "competition" / "train.csv"
    data.parent.mkdir()
    data.write_text("id,x,y\n1,0.5,0\n2,0.7,1\n", encoding="utf-8")
    register_data_source(project.db_path, data, "official", origin_url="https://example.org/train.csv")
    return project, data


def states(project, full=False):
    return [s.state for s in check_data_sources(project.db_path, full=full)]


def test_unchanged_source_is_ok_and_not_copied(setup, workspace):
    project, _ = setup
    assert states(project) == ["ok"]
    assert list(workspace.data_dir.iterdir()) == []


def test_missing_source_is_detected(setup):
    project, data = setup
    data.unlink()
    assert states(project) == ["missing"]


def test_modified_source_is_detected(setup):
    project, data = setup
    data.write_text("id,x,y\n1,0.5,1\n2,0.7,1\n", encoding="utf-8")  # same size, new content
    assert states(project) == ["modified"]


def test_content_change_with_restored_mtime_needs_full_check(setup):
    project, data = setup
    stat = data.stat()
    data.write_text("id,x,y\n1,0.5,1\n2,0.7,1\n", encoding="utf-8")
    os.utime(data, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert states(project) == ["ok"]
    assert states(project, full=True) == ["modified"]


def test_changed_file_cannot_be_silently_re_registered(setup):
    project, data = setup
    register_data_source(project.db_path, data, "official")  # unchanged: no-op
    data.write_text("autre contenu", encoding="utf-8")
    with pytest.raises(DataSourceError, match="a changé"):
        register_data_source(project.db_path, data, "official")


def test_invalid_role_and_missing_file_are_refused(setup, tmp_path):
    project, data = setup
    with pytest.raises(DataSourceError, match="Rôle"):
        register_data_source(project.db_path, data, "private_labels")
    with pytest.raises(DataSourceError, match="introuvable"):
        register_data_source(project.db_path, tmp_path / "absent.csv", "official")


def test_many_files_are_registered_at_once_with_the_same_guarantees(setup, tmp_path):
    project, data = setup
    clips = []
    for index in range(50):
        clip = tmp_path / "competition" / "audio" / f"clip{index}.flac"
        clip.parent.mkdir(exist_ok=True)
        clip.write_bytes(bytes([index]) * 100)
        clips.append(clip)
    assert register_data_sources(project.db_path, [data, *clips], "official") == 50  # train.csv was known
    assert register_data_sources(project.db_path, clips, "official") == 0  # untouched: nothing re-hashed
    clips[3].write_bytes(b"other content of another size")
    with pytest.raises(DataSourceError, match="a changé"):
        register_data_sources(project.db_path, clips, "official")
    assert states(project).count("ok") == 50 and "modified" in states(project)
