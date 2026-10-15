"""Step 5 — compute: baseline descriptors of the media files the tables refer to.

Many competitions are not tabular: each row points to audio clips, images or arrays. ML Forge
turns every media file into a fixed set of robust descriptors and joins them to the tables, so
that the whole tabular engine (protocol, models, decisions) works at once. These descriptors are
a floor, not a ceiling: experiments may read the media themselves (`task["media_dirs"]`).

A row finds its files by name: `<id>.<ext>` or `<id>_<suffix>.<ext>`, possibly in several
sub-folders ("signals/", "unprocessed/"): each sub-folder becomes a prefix of the descriptors,
and two files of the same row are also compared (level and band-energy differences).

Descriptors are cached per file (path, size, modification time) under the workspace, so that a
rerun or another project on the same data never decodes a file twice.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

AUDIO_SUFFIXES = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".opus", ".aiff", ".aif"}
ARRAY_SUFFIXES = {".npy"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
BANDS_HZ = (50, 125, 250, 500, 1000, 2000, 3000, 4000, 6000, 8000, 12000, 16000)
FRAME, HOP = 2048, 1024
MIN_COVERAGE = 0.5  # below this share of rows with media, the descriptors are not joined
CACHE_VERSION = 1


class MediaError(Exception):
    """A media file cannot be described."""


# --- descriptors ----------------------------------------------------------------------------

def _db(value: float) -> float:
    return float(10 * np.log10(max(value, 1e-12)))


def audio_descriptors(path: Path) -> dict[str, float]:
    """Level, dynamics, spectral shape, band energies and channel balance of one audio file."""
    try:
        import soundfile
    except ImportError as exc:  # declared dependency of the audio support
        raise MediaError("Le paquet soundfile est nécessaire pour lire l'audio (pip install soundfile).") from exc
    signal, rate = soundfile.read(str(path), dtype="float32", always_2d=True)
    if signal.size == 0:
        raise MediaError(f"{path.name} : fichier audio vide.")
    mono = signal.mean(axis=1)
    out = {"duration_s": len(mono) / rate, "rms_db": _db(float(np.mean(mono**2))),
           "peak_db": _db(float(np.max(np.abs(mono)) ** 2))}
    out["crest_db"] = out["peak_db"] - out["rms_db"]
    if signal.shape[1] >= 2:
        left, right = _db(float(np.mean(signal[:, 0] ** 2))), _db(float(np.mean(signal[:, 1] ** 2)))
        out.update(channel_left_db=left, channel_right_db=right, channel_best_db=max(left, right),
                   channel_diff_db=abs(left - right),
                   channel_corr=float(np.corrcoef(signal[:, 0], signal[:, 1])[0, 1]) if np.std(signal[:, 0]) * np.std(signal[:, 1]) > 0 else 0.0)
    out.update(_spectral(mono, rate))
    return out


def _spectral(mono: np.ndarray, rate: int) -> dict[str, float]:
    if len(mono) < FRAME:
        mono = np.pad(mono, (0, FRAME - len(mono)))
    count = 1 + (len(mono) - FRAME) // HOP
    index = np.arange(FRAME)[None, :] + HOP * np.arange(count)[:, None]
    frames = mono[index] * np.hanning(FRAME)[None, :]
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2 + 1e-12
    freqs = np.fft.rfftfreq(FRAME, 1 / rate)
    total = power.sum(axis=1)
    frame_db = 10 * np.log10(total / FRAME)
    centroid = (power * freqs).sum(axis=1) / total
    cumulative = np.cumsum(power, axis=1) / total[:, None]
    rolloff = freqs[np.argmax(cumulative >= 0.85, axis=1)]
    flatness = np.exp(np.mean(np.log(power), axis=1)) / np.mean(power, axis=1)
    signs = np.signbit(mono)
    out = {
        "dynamic_range_db": float(np.percentile(frame_db, 95) - np.percentile(frame_db, 5)),
        "level_std_db": float(np.std(frame_db)),
        "silence_share": float(np.mean(frame_db < np.max(frame_db) - 40)),
        "centroid_hz": float(np.mean(centroid)), "centroid_std_hz": float(np.std(centroid)),
        "rolloff_hz": float(np.mean(rolloff)), "flatness": float(np.mean(flatness)),
        "zcr": float(np.mean(signs[1:] != signs[:-1])),
        # Speed of level changes (syllabic rhythm, between 2 and 8 Hz for sung lyrics).
        "modulation_4hz": _modulation(frame_db, rate / HOP),
    }
    edges = [b for b in BANDS_HZ if b < rate / 2] + [rate / 2]
    spectrum = power.mean(axis=0)
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (freqs >= low) & (freqs < high)
        if mask.any():
            out[f"band_{int(low)}_{int(high)}_db"] = _db(float(spectrum[mask].sum()))
    return out


def _modulation(frame_db: np.ndarray, frame_rate: float) -> float:
    envelope = frame_db - frame_db.mean()
    if len(envelope) < 8:
        return 0.0
    spectrum = np.abs(np.fft.rfft(envelope)) ** 2
    freqs = np.fft.rfftfreq(len(envelope), 1 / frame_rate)
    band = (freqs >= 2) & (freqs <= 8)
    return float(spectrum[band].sum() / (spectrum[1:].sum() + 1e-12))


def array_descriptors(path: Path) -> dict[str, float]:
    values = np.load(path, allow_pickle=False)
    flat = np.asarray(values, dtype=float).ravel()
    return {"size": float(flat.size), "mean": float(flat.mean()), "std": float(flat.std()),
            "min": float(flat.min()), "max": float(flat.max()), "ndim": float(values.ndim)}


def image_descriptors(path: Path) -> dict[str, float]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise MediaError("Le paquet pillow est nécessaire pour lire les images.") from exc
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGB"), dtype=float) / 255.0
    gray = pixels.mean(axis=2)
    laplacian = np.abs(gray[1:-1, 1:-1] * 4 - gray[:-2, 1:-1] - gray[2:, 1:-1] - gray[1:-1, :-2] - gray[1:-1, 2:])
    out = {"width": float(pixels.shape[1]), "height": float(pixels.shape[0]),
           "brightness": float(gray.mean()), "contrast": float(gray.std()), "sharpness": float(laplacian.var())}
    for index, channel in enumerate("rgb"):
        out[f"mean_{channel}"] = float(pixels[:, :, index].mean())
    return out


def describe_file(path: Path) -> dict[str, float]:
    suffix = path.suffix.lower()
    if suffix in AUDIO_SUFFIXES:
        return audio_descriptors(path)
    if suffix in ARRAY_SUFFIXES:
        return array_descriptors(path)
    if suffix in IMAGE_SUFFIXES:
        return image_descriptors(path)
    raise MediaError(f"Type de média non décrit : {suffix}")


# --- rows to files --------------------------------------------------------------------------

def index_media(media_dirs: list[Path]) -> dict[str, dict[str, Path]]:
    """{id: {role: file}} where the role is the file's sub-folder ("signals", "unprocessed"…)."""
    index: dict[str, dict[str, Path]] = {}
    suffixes = AUDIO_SUFFIXES | ARRAY_SUFFIXES | IMAGE_SUFFIXES
    for directory in media_dirs:
        for path in Path(directory).rglob("*"):
            if path.suffix.lower() not in suffixes or not path.is_file():
                continue
            key = re.split(r"[_.-](?=[a-zA-Z]+$)", path.stem)[0]  # "<id>_unproc" -> "<id>"
            role = re.sub(r"\W+", "_", path.parent.name.lower()) or "media"
            index.setdefault(key, {}).setdefault(role, path)
            if key != path.stem:
                index.setdefault(path.stem, {}).setdefault(role, path)
    return index


# --- cache and derivation -------------------------------------------------------------------

class DescriptorCache:
    def __init__(self, directory: Path) -> None:
        self.path = directory / f"descripteurs-v{CACHE_VERSION}.jsonl"
        directory.mkdir(parents=True, exist_ok=True)
        self.values: dict[str, dict] = {}
        if self.path.is_file():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                entry = json.loads(line)
                self.values[entry["key"]] = entry["values"]
        self._new: list[str] = []

    @staticmethod
    def key(path: Path) -> str:
        stat = path.stat()
        return hashlib.sha256(f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode()).hexdigest()[:24]

    def get(self, path: Path) -> dict | None:
        return self.values.get(self.key(path))

    def put(self, path: Path, values: dict) -> None:
        key = self.key(path)
        self.values[key] = values
        self._new.append(json.dumps({"key": key, "file": path.name, "values": values}))

    def flush(self) -> None:
        if self._new:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write("\n".join(self._new) + "\n")
            self._new = []


def describe_rows(ids: list[str], index: dict[str, dict[str, Path]], cache: DescriptorCache,
                  workers: int | None = None, progress=None) -> tuple[list[dict], list[str]]:
    """Descriptors of each row (in order) and the problems met; a row without media gets {}."""
    files = sorted({path for key in ids for path in index.get(str(key), {}).values()})
    todo = [path for path in files if cache.get(path) is None]
    problems: list[str] = []

    def work(path: Path):
        try:
            return path, describe_file(path), None
        except Exception as exc:  # one unreadable file never stops the others
            return path, None, f"{path.name} : {exc}"

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers or max(1, (os.cpu_count() or 2) - 1)) as pool:
        for done, (path, values, error) in enumerate(pool.map(work, todo), start=1):
            if error:
                problems.append(error)
            else:
                cache.put(path, values)
            if done % 500 == 0:
                cache.flush()
                if progress:
                    progress(done, len(todo), time.monotonic() - started)
    cache.flush()
    rows = []
    for key in ids:
        roles = index.get(str(key), {})
        row: dict[str, float] = {}
        described = {role: cache.get(path) for role, path in roles.items()}
        for role, values in described.items():
            row.update({f"{role}__{name}": value for name, value in (values or {}).items()})
        _pair_features(row, described)
        rows.append(row)
    return rows, problems


def _pair_features(row: dict, described: dict[str, dict | None]) -> None:
    """Two files of one row (heard signal vs reference): what the first lost relative to the second."""
    roles = [role for role, values in described.items() if values]
    if len(roles) != 2:
        return
    first, second = sorted(roles)
    a, b = described[first], described[second]
    for name in a:
        if name in b and (name.endswith("_db") or name in {"centroid_hz", "rolloff_hz", "flatness"}):
            row[f"diff__{name}"] = a[name] - b[name]
