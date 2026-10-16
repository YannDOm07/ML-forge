"""Step 4 — formalise the task from the files themselves, whatever their layout.

Competitions rarely ship `train.csv` + `test.csv` + `sample_submission.csv`. Tables may be JSON
metadata next to thousands of audio or image files, there may be several sets to predict
(validation and evaluation), and the training table often holds columns the other sets do not
have (labels, annotations): those are the target candidates, and leaks for everything else.

Everything here is measured on the files; each conclusion is stored as cited evidence, and an
ambiguous one stays 'inferred' so that the agent or the user confirms it.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from forge.task import TABLE_FORMATS, TaskSpec, read_table

MEDIA_SUFFIXES = {
    ".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus",  # audio
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp",  # images
    ".mp4", ".avi", ".mov", ".mkv",  # video
    ".npy", ".npz", ".pt", ".h5", ".hdf5", ".mat", ".nii", ".dcm",  # arrays and scans
}
NOT_TABLES = re.compile(r"manifest|checksum|md5|sha\d*sum|readme|changelog|licen[cs]e|citation", re.IGNORECASE)
TRAIN_WORDS = {"train", "training"}
SET_WORDS = {"test": "test", "testing": "test", "eval": "eval", "evaluation": "eval", "valid": "valid",
             "validation": "valid", "dev": "valid", "public": "test", "private": "test", "holdout": "test"}
SAMPLE_WORDS = re.compile(r"sample|submission|example", re.IGNORECASE)
PRIMARY_ORDER = ("test", "eval", "valid")  # the set whose predictions are the main deliverable
TABLE_MAX_BYTES = 512 * 1024**2
TARGET_WORDS = re.compile(r"\btarget\b|\blabels?\b|ground[- ]truth|to predict|à prédire|\bcible\b", re.IGNORECASE)
GROUP_SENTENCE = re.compile(r"split(?:ting)?\s+(?:the\s+data\s+)?(?:by|on|per)\s+(?:unique\s+|the\s+)?([a-z_]+)",
                            re.IGNORECASE)


@dataclass
class Layout:
    train: Path | None = None
    sets: dict[str, Path] = field(default_factory=dict)
    sample: Path | None = None
    media_dirs: list[Path] = field(default_factory=list)
    columns: dict[Path, list[str]] = field(default_factory=dict)

    @property
    def primary(self) -> str | None:
        return next((name for name in PRIMARY_ORDER if name in self.sets), next(iter(self.sets), None))


def _words(path: Path, root: Path) -> list[str]:
    relative = path.relative_to(root) if path.is_relative_to(root) else Path(path.name)
    return [word for part in relative.with_suffix("").parts for word in re.split(r"[^a-z]+", part.lower()) if word]


def discover(paths: list[Path]) -> Layout:
    """Tables by role, and the directories of media files they refer to."""
    paths = [Path(path) for path in paths]
    if not paths:
        return Layout()
    root = Path(os.path.commonpath([str(p.parent) for p in paths])) if len(paths) > 1 else paths[0].parent
    layout = Layout()
    tables = [p for p in paths if p.suffix.lower() in TABLE_FORMATS and not NOT_TABLES.search(p.name)
              and p.stat().st_size <= TABLE_MAX_BYTES]
    # Shortest names first: "train.csv" beats "train_responses.json" for the same role.
    for path in sorted(tables, key=lambda p: (len(p.name), str(p))):
        words = _words(path, root)
        name_words = re.split(r"[^a-z]+", path.stem.lower())
        if SAMPLE_WORDS.search(path.stem):
            layout.sample = layout.sample or path
        elif TRAIN_WORDS & set(name_words) or (TRAIN_WORDS & set(words) and not set(SET_WORDS) & set(name_words)):
            layout.train = layout.train or path
        else:
            role = next((SET_WORDS[w] for w in name_words if w in SET_WORDS), None) or next(
                (SET_WORDS[w] for w in words if w in SET_WORDS), None)
            if role and role not in layout.sets:
                layout.sets[role] = path
    media = {p.parent for p in paths if p.suffix.lower() in MEDIA_SUFFIXES}
    tops = sorted({root / d.relative_to(root).parts[0] if d != root and d.is_relative_to(root) else d for d in media})
    layout.media_dirs = tops
    for path in [layout.train, layout.sample, *layout.sets.values()]:
        if path is not None:
            layout.columns[path] = [str(c) for c in read_table(path, nrows=50).columns]
    return layout


@dataclass
class Inference:
    """What the files say about the task, each with the evidence to cite."""
    target: tuple[str, str] | None = None  # (value, evidence)
    target_sure: bool = False
    id_column: tuple[str, str] | None = None
    group_column: tuple[str, str, str] | None = None  # (value, source url, quote)
    train_only: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def infer(layout: Layout, pages: list[dict]) -> Inference:
    result = Inference()
    if layout.train is None:
        result.notes.append("Aucune table d'entraînement reconnue parmi les fichiers.")
        return result
    train_columns = layout.columns[layout.train]
    set_columns = [set(layout.columns[path]) for path in layout.sets.values()]
    if set_columns:
        common = set.intersection(*set_columns)
        result.train_only = [c for c in train_columns if c not in common]
        shown = ", ".join(f"{name} : {path.name}" for name, path in layout.sets.items())
        evidence = (f"Table d'entraînement {layout.train.name} — colonnes : {', '.join(train_columns)}. "
                    f"Jeux à prédire ({shown}) — colonnes communes : {', '.join(c for c in train_columns if c in common)}. "
                    f"Colonnes propres à l'entraînement : {', '.join(result.train_only) or 'aucune'}.")
        candidates = result.train_only
        if len(candidates) == 1:
            result.target, result.target_sure = (candidates[0], evidence), True
        elif candidates:
            cited = _cited_target(candidates, pages)
            result.target = (cited or candidates[0], evidence)
            result.target_sure = False
            if cited:
                result.notes.append(f"Cible probable : {cited} (citée comme cible dans la documentation).")
        result.id_column = _id_column(layout, train_columns, common)
    result.group_column = _group_column(train_columns, pages)
    return result


def _cited_target(candidates: list[str], pages: list[dict]) -> str | None:
    """The candidate a documentation sentence calls the target ("correctness: … (the target variable)")."""
    hits = {}
    for page in pages:
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", page.get("text") or ""):
            if not TARGET_WORDS.search(sentence):
                continue
            for column in candidates:
                if re.search(rf"(?<![\w]){re.escape(column)}(?![\w])", sentence):
                    hits[column] = hits.get(column, 0) + 1
    ranked = sorted(hits.items(), key=lambda item: -item[1])
    if not ranked or (len(ranked) > 1 and ranked[0][1] == ranked[1][1]):
        return None  # no citation, or a tie: the choice is left to the agent or the user
    return ranked[0][0]


def _id_column(layout: Layout, train_columns: list[str], common: set[str]) -> tuple[str, str] | None:
    train = read_table(layout.train, usecols=[c for c in train_columns if c in common])
    for column in train.columns:
        values = train[column]
        if values.is_unique and not values.isna().any() and not str(values.dtype).startswith("float"):
            return column, (f"Colonne {column} : présente dans toutes les tables, une valeur distincte par ligne "
                            f"({len(values)} lignes dans {layout.train.name}).")
    return None


def _group_column(train_columns: list[str], pages: list[dict]) -> tuple[str, str, str] | None:
    """A column the rules ask to split by ("we recommend splitting by unique prompts")."""
    for page in pages:
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", page.get("text") or ""):
            match = GROUP_SENTENCE.search(sentence)
            if not match:
                continue
            word = match.group(1).lower()
            for column in train_columns:
                if column.lower() in {word, word.rstrip("s"), word.removesuffix("es")}:
                    return column, page["url"], sentence.strip()
    return None


def build_task(layout: Layout, problem_type: str, target: str, metric: str, id_column: str | None,
               group_column: str | None, train_only: list[str], positive_label: str | None = None) -> TaskSpec:
    primary = layout.primary
    sets = {name: str(path) for name, path in layout.sets.items()}
    return TaskSpec(
        problem_type=problem_type, target=target, metric=metric, train_path=str(layout.train),
        test_path=sets.get(primary) if primary else None, id_column=id_column, group_column=group_column,
        positive_label=positive_label,
        prediction_sets=sets if len(sets) > 1 else None,
        media_dirs=tuple(str(d) for d in layout.media_dirs),
        train_only_columns=tuple(c for c in train_only if c != target),
    )


def with_media_descriptors(workspace, project, task: TaskSpec, log=None) -> TaskSpec:
    """Join baseline descriptors of the media files to every table of the task (step 5).

    The derived tables live in the project (`data/derived/`), are registered as 'derived' data
    sources, and replace the original tables in the task; media folders stay readable."""
    import pandas as pd

    from forge.datasources import register_data_source
    from forge.media import MIN_COVERAGE, DescriptorCache, describe_rows, index_media

    if not task.media_dirs or not task.id_column:
        return task
    index = index_media([Path(d) for d in task.media_dirs])
    cache = DescriptorCache(workspace.data_dir / "derived" / "media-cache")
    target = project.path / "data" / "derived"
    target.mkdir(parents=True, exist_ok=True)
    replaced: dict[str, str] = {}
    for name, path in {"train": task.train_path, **task.sets()}.items():
        table = read_table(path)
        ids = table[task.id_column].astype(str).tolist()
        coverage = sum(1 for key in ids if key in index) / max(len(ids), 1)
        if coverage < MIN_COVERAGE:
            if log:
                log(f"{Path(path).name} : {coverage:.0%} des lignes ont un média ; descripteurs non joints.")
            return task

        def progress(done, total, seconds, table_name=Path(path).name):
            # Progress goes to the event log; the journal keeps one summary per table.
            if done % 2000 == 0:
                from forge.experiments import log_event

                log_event(project.db_path, None, "calcul", f"{table_name} : {done}/{total} fichiers décrits en {seconds:.0f} s.")

        rows, problems = describe_rows(ids, index, cache, progress=progress)
        descriptors = pd.DataFrame(rows, index=table.index)
        derived = target / f"{Path(path).stem}__media.csv"
        pd.concat([table, descriptors], axis=1).to_csv(derived, index=False)
        register_data_source(project.db_path, derived, "derived", origin_url=Path(path).resolve().as_uri())
        replaced[str(path)] = str(derived)
        if log:
            log(f"{Path(path).name} : {len(table)} lignes, {descriptors.shape[1]} descripteurs, couverture {coverage:.0%}"
                + (f", {len(problems)} fichier(s) illisible(s) : {'; '.join(problems[:3])}" if problems else "") + ".")
    sets = {name: replaced[path] for name, path in task.sets().items()}
    return TaskSpec(**{**task.to_dict(), "train_path": replaced[task.train_path],
                       "test_path": replaced[task.test_path] if task.test_path else None,
                       "prediction_sets": sets if task.prediction_sets else None})
