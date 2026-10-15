"""Steps 4 and 5 on a competition laid out like CLIP2: JSON metadata, audio files, two sets to
predict, a label column that only the training table has, and rules asking to split by prompt."""

import json
import wave
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forge.datasources import register_data_source
from forge.formalize import build_task, discover, infer, with_media_descriptors
from forge.intake import competition_card, infer_from_data, store_page
from forge.projects import create_project
from forge.runner import define_protocol, profile_data, run_experiment, submit_experiment
from forge.task import TaskSpec, read_table

DOC = "https://lyrics.example.org/docs/spec"
DOC_TEXT = (
    "correctness: Intelligibility score i.e. rate of correctly identified words (the target variable).\n"
    "phoneme_correctness: Intelligibility score in terms of phoneme accuracy, not used.\n"
    "Since each excerpt is presented three times, we recommend splitting by unique prompts first."
)


def _wav(path: Path, loudness: float, cutoff: float, seed: int) -> None:
    rate, rng = 16_000, np.random.default_rng(seed)
    noise = rng.normal(0, 1, rate)
    spectrum = np.fft.rfft(noise)
    spectrum[np.fft.rfftfreq(rate, 1 / rate) > cutoff] = 0  # a "hearing loss": high bands removed
    signal = np.fft.irfft(spectrum, rate) * loudness
    stereo = np.stack([signal, signal * 0.8], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes((np.clip(stereo, -1, 1) * 32767).astype("<i2").tobytes())


def make_competition(root: Path, n_prompts: int = 24) -> list[Path]:
    rng = np.random.default_rng(7)
    losses = {"No Loss": 7000.0, "Mild": 3000.0, "Moderate": 1200.0}
    tables = {"train": [], "valid": [], "eval": []}
    for split, count in (("train", n_prompts), ("valid", 6), ("eval", 6)):
        for p in range(count):
            for loss, cutoff in losses.items():
                signal = f"{split}{p:03d}{loss[:2].lower()}"
                _wav(root / f"clip.{split}" / split / "signals" / f"{signal}.wav", 0.2, cutoff, len(tables[split]))
                _wav(root / f"clip.{split}" / split / "unprocessed" / f"{signal}_unproc.wav", 0.2, 7000.0, 99)
                row = {"signal": signal, "prompt": f"prompt {split} {p}", "n_words": int(rng.integers(4, 10)),
                       "hearing_loss": loss}
                if split == "train":
                    row["correctness"] = float(np.clip(cutoff / 7000 + rng.normal(0, 0.05), 0, 1))
                    row["phoneme_correctness"] = row["correctness"]
                tables[split].append(row)
    paths = []
    for split, rows in tables.items():
        path = root / f"clip.{split}" / "metadata" / f"{split}_metadata.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows), encoding="utf-8")
        paths.append(path)
    (root / "clip.train" / "manifest").mkdir()
    (root / "clip.train" / "manifest" / "train.md5sum.json").write_text("[]", encoding="utf-8")
    return paths + sorted(p for p in root.rglob("*.wav"))


@pytest.fixture
def competition(tmp_path):
    return make_competition(tmp_path / "data")


def test_tables_of_any_format_are_read(tmp_path):
    frame = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    frame.to_csv(tmp_path / "t.tsv", sep="\t", index=False)
    frame.to_json(tmp_path / "t.jsonl", orient="records", lines=True)
    (tmp_path / "t.json").write_text(json.dumps({"data": frame.to_dict("records")}), encoding="utf-8")
    for name in ("t.tsv", "t.jsonl", "t.json"):
        assert read_table(tmp_path / name).to_dict("records") == frame.to_dict("records")
    assert list(read_table(tmp_path / "t.json", usecols=["b"]).columns) == ["b"]


def test_layout_target_leaks_id_and_groups_are_inferred(competition):
    layout = discover(competition)
    assert layout.train.name == "train_metadata.json"
    assert {name: path.name for name, path in layout.sets.items()} == {
        "valid": "valid_metadata.json", "eval": "eval_metadata.json"}
    assert layout.primary == "eval" and len(layout.media_dirs) == 3
    result = infer(layout, [{"url": DOC, "text": DOC_TEXT}])
    # Two columns only the training table has: the documentation names the target.
    assert result.target[0] == "correctness" and not result.target_sure
    assert set(result.train_only) == {"correctness", "phoneme_correctness"}
    assert result.id_column[0] == "signal"
    assert result.group_column[0] == "prompt"


def test_inferred_facts_are_cited_and_the_ambiguous_target_awaits_confirmation(workspace, competition):
    project = create_project(workspace, "Clip", "https://eval.example.org/web/challenges/challenge-page/9/overview")
    store_page(project, DOC, "web", DOC_TEXT)
    infer_from_data(project, competition)
    fields = competition_card(project)["fields"]
    assert fields["target"]["value"] == "correctness" and fields["target"]["status"] == "inferred"
    assert fields["id_column"]["value"] == "signal"
    assert fields["group_column"]["value"] == "prompt" and "splitting by unique prompts" in fields["group_column"]["quote"]
    assert fields["problem_type"]["value"] == "regression"


def test_media_descriptors_feed_the_whole_chain(workspace, competition):
    project = create_project(workspace, "Clip")
    for path in competition:
        register_data_source(project.db_path, path, "official")
    layout = discover(competition)
    task = build_task(layout, "regression", "correctness", "rmse", "signal", "prompt",
                      ["correctness", "phoneme_correctness"])
    assert task.train_only_columns == ("phoneme_correctness",)
    assert "phoneme_correctness" in task.excluded_columns()
    task = with_media_descriptors(workspace, project, task)
    train = read_table(task.train_path)
    assert "diff__band_2000_3000_db" in train.columns and train["signals__rms_db"].notna().all()
    # Heard signals lost their high bands relative to the reference, more so for "Moderate".
    by_loss = train.groupby("hearing_loss")["diff__band_4000_6000_db"].mean()
    assert by_loss["Moderate"] < by_loss["Mild"] < by_loss["No Loss"] + 1e-6
    again = with_media_descriptors(workspace, project, TaskSpec(**{**task.to_dict(), "train_path": str(layout.train),
                                                                    "test_path": str(layout.sets["eval"]),
                                                                    "prediction_sets": {k: str(v) for k, v in layout.sets.items()}}))
    assert read_table(again.train_path).equals(train)  # descriptors come from the cache, identical

    protocol = define_protocol(workspace, project, task)
    folds = pd.Series(protocol.folds)
    prompts = train["prompt"]
    assert all(folds[prompts == prompt].nunique() == 1 for prompt in prompts.unique())  # one prompt, one fold
    profile_data(workspace, project, task)
    from forge.demo import builtin_card
    from forge.runner import builtin_code

    baseline = submit_experiment(project, builtin_card(protocol.id, "baseline", "Plancher.", 0.0, "baseline"),
                                 builtin_code("baseline"))
    assert run_experiment(workspace, project, baseline.id).status == "completed"
    model = submit_experiment(project, builtin_card(protocol.id, "hist_gb", "Les descripteurs portent la perte auditive.", 0.05),
                              builtin_code("hist_gb"))
    outcome = run_experiment(workspace, project, model.id)
    assert outcome.status == "completed" and outcome.decision == "accepted", outcome.message
    folder = project.path / "experiments" / model.id
    eval_pred, valid_pred = pd.read_csv(folder / "test_pred.csv"), pd.read_csv(folder / "pred_valid.csv")
    assert list(eval_pred.columns) == ["signal", "prediction"] and len(eval_pred) == len(valid_pred) == 18
    info = json.loads((folder / "info.json").read_text(encoding="utf-8"))
    assert "phoneme_correctness" not in info["features"] and "correctness" not in info["features"]

    # Step 9: ML Forge records the measured outcome itself, in the project's context (audio, regression).
    from forge.lessons import project_context, search

    context = project_context(project)
    assert context["modality"] == "audio" and context["problem_type"] == "regression"
    measured = [l for l in search(workspace, context, step="experimenter", limit=50) if l["source"] == "ml-forge"]
    assert measured and measured[0]["résultat"] == "marche" and "hist_gb" in measured[0]["titre"]
    image = search(workspace, {**context, "modality": "image"}, step="experimenter", limit=50)
    assert all(l["source"] != "ml-forge" for l in image)  # an audio measurement never answers an image project

    # Step 8: no sample file; the rules describe one CSV per set, scores in percent.
    from forge.submission import SubmissionError, build_from_spec, parse_spec

    spec = parse_spec(json.dumps({"files": {"valid": "CUP_valid_{TEAM_ID}.csv", "eval": "CUP_eval_{TEAM_ID}.csv"},
                                  "columns": {"id": "signal_id", "prediction": "intelligibility_score"},
                                  "scale": 100, "clip": [0, 100]}))
    files = build_from_spec(project, model.id, spec, lambda request: False, "test")
    assert sorted(p.name for p in files) == ["CUP_eval_TEAM_ID.csv", "CUP_valid_TEAM_ID.csv"]
    written = pd.read_csv(files[0])
    assert list(written.columns) == ["signal_id", "intelligibility_score"]
    assert written["intelligibility_score"].between(0, 100).all() and written["intelligibility_score"].max() > 1
    with pytest.raises(SubmissionError):
        parse_spec('{"files": {}, "columns": {"id": "a", "prediction": "b"}}')
    with pytest.raises(SubmissionError, match="Jeux inconnus"):
        build_from_spec(project, model.id, {**spec, "files": {"private": "x.csv"}}, lambda request: False, "test")
