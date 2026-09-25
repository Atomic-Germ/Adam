#!/usr/bin/env python3
"""Embedding-driven memory for the Mind daemon.

Brings Arthur's memory capability into the Mind body:

  * embedder — Arthur's `HashingEmbedder` (blake2b bag-of-words, 384-dim,
    pure numpy, deterministic) by default; optional sentence-transformers via
    `MIND_EMBED_BACKEND=st` with automatic hash fallback (`MIND_EMBED_MODEL`).
  * store — the Cathedral's engine shape: a readable `memory.json` of records
    plus a parallel `embeddings.npy` matrix; cosine similarity for recall.
  * chunking — Arthur's paragraph/word chunker (chunk/overlap configurable).
  * relevance selection — mixed semantic + lexical + recency scoring (the
    Bubble `_select_relevant_memory` pattern) so recall works even with the
    embedder disabled.

Records are tagged by `source` — "instruction" (seeded, e.g. the letter),
"user", or "mind" — so the mind can always tell who said what, mirroring
Bubble's user/self distinction.

The store degrades gracefully: without numpy (or with the backend "off"),
selection falls back to lexical + recency only.
"""

from __future__ import annotations

import glob
import hashlib
import json
import logging
import math
import os
import queue
import random
import re
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger("mind-memory")

_EMB_DIM = 384
_TOKEN_RE = re.compile(r"[a-z0-9']+", re.I)

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:  # pragma: no cover
    np = None
    HAS_NUMPY = False


# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------
DEFAULT_MEMORY_DIR = Path.home() / ".local" / "share" / "mind"
DEFAULT_EMBED_BACKEND = "hash"          # hash | st | off
DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 6
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 80
DEFAULT_BACKUP_LIMIT = 50
SEED_DIR_NAME = "original_memory"
_SOURCE_LABELS = {"instruction": "instruction", "user": "user", "mind": "mind"}


def _int_env(key: str, default: int, lo: int, hi: int) -> int:
    raw = os.environ.get(key, "").strip()
    if raw.isdigit():
        v = int(raw)
        if lo <= v <= hi:
            return v
    return default


def _float_env(key: str, default: float, lo: float, hi: float) -> float:
    raw = os.environ.get(key, "").strip()
    try:
        v = float(raw)
    except ValueError:
        return default
    if lo <= v <= hi:
        return v
    return default


def _getenv_bool(name: str, default: bool = False) -> bool:
    return os.environ.get(name, "1" if default else "0").strip().lower() in {
        "1", "true", "yes", "on",
    }


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------
class HashingEmbedder:
    """Deterministic bag-of-words hashing vectorizer (no neural net)."""

    def __init__(self, dim: int = _EMB_DIM):
        self.dim = dim

    def encode(self, texts, show_progress_bar=False, convert_to_numpy=True):
        if isinstance(texts, str):
            texts = [texts]
        if HAS_NUMPY:
            return np.vstack([self._one(t) for t in texts])
        return [self._one_list(t) for t in texts]

    def _one(self, text: str):
        vec = np.zeros(self.dim, dtype=np.float32)
        for tok in _TOKEN_RE.findall((text or "").lower()):
            h = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(h[:4], "little") % self.dim
            sign = 1.0 if h[4] % 2 == 0 else -1.0
            vec[idx] += sign
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec

    def _one_list(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in _TOKEN_RE.findall((text or "").lower()):
            h = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(h[:4], "little") % self.dim
            sign = 1.0 if h[4] % 2 == 0 else -1.0
            vec[idx] += sign
        norm = (sum(v * v for v in vec) ** 0.5) or 1.0
        return [v / norm for v in vec]


class _SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        getter = getattr(self._model, "get_embedding_dimension",
                         self._model.get_sentence_embedding_dimension)
        self.dim = getter()

    def encode(self, texts, show_progress_bar=False, convert_to_numpy=True):
        if isinstance(texts, str):
            texts = [texts]
        vecs = self._model.encode(
            texts, show_progress_bar=show_progress_bar,
            convert_to_numpy=(HAS_NUMPY or False), normalize_embeddings=True,
        )
        if HAS_NUMPY:
            return np.asarray(vecs, dtype=np.float32)
        return [list(v) for v in vecs]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into chunks (Arthur's paragraph/word chunker)."""
    text = text.strip()
    if not text:
        return []
    paragraphs = re.split(r"\n\s*\n", text)
    chunks: list[str] = []
    current = ""

    def flush():
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
        current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(para) <= chunk_size:
            if len(current) + len(para) + 1 <= chunk_size:
                current = f"{current}\n\n{para}".strip()
            else:
                flush()
                current = para
        else:
            flush()
            words = para.split()
            buf: list[str] = []
            size = 0
            for w in words:
                if size + len(w) + 1 > chunk_size and buf:
                    chunks.append(" ".join(buf))
                    overlap_words = []
                    osize = 0
                    for ow in reversed(buf):
                        if osize + len(ow) + 1 > overlap:
                            break
                        overlap_words.insert(0, ow)
                        osize += len(ow) + 1
                    buf = overlap_words + [w]
                    size = sum(len(x) + 1 for x in buf)
                else:
                    buf.append(w)
                    size += len(w) + 1
            if buf:
                chunks.append(" ".join(buf))
            current = ""
    flush()
    return chunks


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------
class MindMemory:
    """Records + embeddings matrix, Cathedral-style, with atomic persistence."""

    def __init__(
        self,
        memory_dir: Optional[Path] = None,
        backend: str = "",
        model: str = "",
        top_k: int = 0,
        chunk_size: int = 0,
        chunk_overlap: int = 0,
    ) -> None:
        env_dir = os.environ.get("MIND_MEMORY_DIR", "").strip()
        self.memory_dir = Path(env_dir) if env_dir else (memory_dir or DEFAULT_MEMORY_DIR)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.records_path = self.memory_dir / "memory.json"
        self.embeddings_path = self.memory_dir / "embeddings.npy"
        self.backup_dir = self.memory_dir / "memory-backups"

        self.backend = (backend or os.environ.get("MIND_EMBED_BACKEND", "").strip()
                        or DEFAULT_EMBED_BACKEND).lower()
        if self.backend not in {"hash", "st", "off"}:
            self.backend = DEFAULT_EMBED_BACKEND
        self.model = model or os.environ.get("MIND_EMBED_MODEL", "").strip() or DEFAULT_EMBED_MODEL
        self.top_k = top_k or _int_env("MIND_MEMORY_TOP_K", DEFAULT_TOP_K, 1, 64)
        self.chunk_size = chunk_size or _int_env("MIND_MEMORY_CHUNK_SIZE", DEFAULT_CHUNK_SIZE, 64, 8192)
        self.chunk_overlap = chunk_overlap or _int_env(
            "MIND_MEMORY_CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP, 0, 1024,
        )
        self.sync_index = _getenv_bool("MIND_MEMORY_SYNC_INDEX", False)
        self.fuzzy = _getenv_bool("MIND_MEMORY_FUZZY", True)
        self.fuzzy_temp = _float_env("MIND_MEMORY_FUZZY_TEMP", 2.0, 0.1, 10.0)
        self.fuzzy_hops = _int_env("MIND_MEMORY_FUZZY_HOPS", 1, 0, 4)
        self.fuzzy_seed = os.environ.get("MIND_MEMORY_FUZZY_SEED", "").strip()

        self._lock = threading.RLock()
        self._embedder = None
        self._embed_error: Optional[str] = None
        self._embed_cache: dict[str, list[float]] = {}
        self._records: list[dict] = []
        self._matrix: Optional[list] = None        # list-of-lists if not numpy
        self._matrix_np = None                     # np.ndarray if numpy
        self._jet: "queue.Queue[Optional[dict]]" = queue.Queue()
        self._pending: set[str] = set()
        self._worker: Optional[threading.Thread] = None
        self._last_change: float = 0.0

        self._load_or_create()

    # ------------------------------------------------------------------
    # Loading / persistence
    # ------------------------------------------------------------------
    def _load_or_create(self) -> None:
        if self.records_path.exists():
            try:
                with open(self.records_path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                self._records = data.get("records", [])
                log.info("Memory loaded: %d records from %s",
                         len(self._records), self.records_path)
            except Exception as exc:  # noqa: BLE001
                log.warning("Could not read %s (%s); starting empty",
                            self.records_path, exc)
                self._records = []
        self._load_embeddings()
        self._start_worker()

    def _load_embeddings(self) -> None:
        if not HAS_NUMPY:
            return
        if self.embeddings_path.exists():
            try:
                arr = np.load(self.embeddings_path)
                if arr.shape[0] == len(self._records):
                    self._matrix_np = arr
                    return
                log.warning("Embeddings matrix (%s) out of sync with records (%d); "
                            "will re-embed", arr.shape, len(self._records))
            except Exception as exc:  # noqa: BLE001
                log.warning("Could not load %s (%s)", self.embeddings_path, exc)

    def save(self) -> None:
        with self._lock:
            self._backup_existing()
            tmp = self.records_path.with_suffix(".json.tmp")
            tmp.write_text(
                json.dumps({"version": 1, "records": self._records},
                           indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            tmp.replace(self.records_path)
            if HAS_NUMPY and self._matrix_np is not None:
                npy_tmp = self.embeddings_path.with_suffix(".npy.tmp")
                np.save(npy_tmp, self._matrix_np)
                npy_tmp.replace(self.embeddings_path)
            self._last_change = time.time()

    def _backup_existing(self) -> None:
        if not self.records_path.exists():
            return
        stamp = time.strftime("%Y%m%d")
        marker = self.backup_dir / f".daily-{stamp}"
        if marker.exists():
            return
        try:
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.records_path, self.backup_dir / f"memory-{stamp}.json")
            marker.touch()
            backups = sorted(self.backup_dir.glob("memory-*.json"))
            for old in backups[:-DEFAULT_BACKUP_LIMIT]:
                old.unlink(missing_ok=True)
        except Exception as exc:  # noqa: BLE001
            log.warning("Memory backup failed: %s", exc)

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------
    def _get_embedder(self):
        if self._embedder is not None:
            return self._embedder
        if self.backend == "off":
            return None
        with self._lock:
            if self._embedder is not None:
                return self._embedder
            if not HAS_NUMPY:
                log.warning("numpy unavailable — semantic recall disabled")
                self.backend = "off"
                return None
            try:
                if self.backend == "st":
                    self._embedder = _SentenceTransformerEmbedder(self.model)
                    log.info("Embedder: sentence-transformers %s", self.model)
                else:
                    self._embedder = HashingEmbedder(_EMB_DIM)
                    log.info("Embedder: hashing (dim=%s)", _EMB_DIM)
            except Exception as exc:  # noqa: BLE001
                self._embed_error = str(exc)
                log.warning("Embedder %r failed (%s); falling back to hashing",
                            self.backend, exc)
                try:
                    self._embedder = HashingEmbedder(_EMB_DIM)
                except Exception:  # pragma: no cover
                    self._embedder = None
            return self._embedder

    def _embed_texts(self, texts: list[str]):
        embedder = self._get_embedder()
        if embedder is None:
            return None
        if HAS_NUMPY:
            return embedder.encode(texts)
        return embedder.encode(texts)

    def _embed_one(self, text: str):
        clean = (text or "").strip()
        if not clean:
            return None
        key = hashlib.sha1(clean.encode("utf-8", "ignore")).hexdigest()
        with self._lock:
            cached = self._embed_cache.get(key)
        if cached is not None:
            return cached
        vec = self._embed_texts([clean])
        if vec is None:
            return None
        row = vec[0]
        if HAS_NUMPY:
            row = np.asarray(row, dtype=np.float32)
        with self._lock:
            self._embed_cache[key] = row
        return row

    def _ensure_matrix(self) -> None:
        """Ensure embeddings for every record exist (in-line with the file)."""
        if self.backend == "off":
            return
        if not HAS_NUMPY:
            return
        if self._matrix_np is not None and self._matrix_np.shape[0] == len(self._records):
            return
        rows = []
        for rec in self._records:
            vec = self._embed_one(rec.get("text", ""))
            if vec is None:
                rows.append(np.zeros(_EMB_DIM, dtype=np.float32))
            else:
                rows.append(vec)
        if rows:
            self._matrix_np = np.vstack(rows).astype(np.float32)
        else:
            self._matrix_np = np.zeros((0, _EMB_DIM), dtype=np.float32)

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------
    def add_chunked(self, text: str, source: str, kind: str = "experience") -> int:
        """Chunk + store long text as records; returns count added."""
        added = 0
        for chunk in chunk_text(text, self.chunk_size, self.chunk_overlap):
            self._record(chunk, source=source, kind=kind)
            added += 1
        if self.sync_index:
            self.save()
        return added

    def _record(self, text: str, source: str, kind: str = "experience") -> int:
        with self._lock:
            rec = {
                "id": (self._records[-1]["id"] + 1) if self._records else 0,
                "text": text,
                "source": source,
                "kind": kind,
                "created_at": int(time.time()),
                "updated_at": int(time.time()),
            }
            self._records.append(rec)
            self._matrix_np = None  # matrix is now stale; rebuild lazily
            self._last_change = time.time()
            return rec["id"]

    def _index_job(self, text: str, source: str, kind: str) -> bool:
        key = hashlib.sha1(
            f"{source}|{kind}|{text}".encode("utf-8", "ignore")
        ).hexdigest()
        with self._lock:
            if key in self._pending:
                return False
            self._pending.add(key)
        self._jet.put({"text": text, "source": source, "kind": kind, "key": key})
        return True

    def _start_worker(self) -> None:
        if self.sync_index:
            return
        if self._worker and self._worker.is_alive():
            return
        self._worker = threading.Thread(
            target=self._worker_loop, daemon=True, name="mind-memory-index",
        )
        self._worker.start()

    def _worker_loop(self) -> None:
        while True:
            job = self._jet.get()
            if job is None:
                break
            try:
                self.add_chunked(job["text"], source=job["source"], kind=job["kind"])
                self.save()
                log.info("Indexed %s chunk from '%s' (%d chars)",
                         job["source"], job["kind"], len(job["text"]))
            except Exception as exc:  # noqa: BLE001
                log.warning("Indexing failed: %s", exc)
            finally:
                with self._lock:
                    self._pending.discard(job["key"])
                self._jet.task_done()

    def add_experience(self, text: str, source: str) -> None:
        """Queue a conversation text for indexing (background by default)."""
        if not (text or "").strip():
            return
        if self.sync_index:
            self.add_chunked(text, source=source, kind="experience")
            return
        if self._index_job(text, source, "experience"):
            log.debug("Queued %s experience (%d chars)", source, len(text))

    def flush(self, timeout: float = 30.0) -> None:
        """Wait for queued index jobs to drain (tests / clean shutdown)."""
        if self.sync_index:
            return
        deadline = time.monotonic() + timeout
        while self._jet.unfinished_tasks > 0:
            if time.monotonic() >= deadline:
                break
            time.sleep(0.05)

    def add_instruction(self, text: str) -> int:
        for chunk in chunk_text(text, self.chunk_size, self.chunk_overlap):
            self._record(chunk, source="instruction", kind="instruction")
        if self.sync_index:
            self.save()
        return len(self._records)

    def add_dream(self, summary: str) -> int:
        """Store a previous dream in the same mental space as everything else."""
        if not (summary or "").strip():
            return 0
        n = self.add_chunked(summary, source="mind", kind="dream")
        self.save()
        return n

    def has_dream(self) -> bool:
        return any(r.get("kind") == "dream" for r in self._records)

    def instruction_texts(self) -> list[str]:
        """The seeded instruction records (the mind's first memory)."""
        return [
            r.get("text", "")
            for r in self._records
            if r.get("kind") == "instruction" and (r.get("text") or "")
        ]

    # ------------------------------------------------------------------
    # Seeding
    # ------------------------------------------------------------------
    def seed_dir(self, seed_dir: Optional[Path]) -> int:
        """Seed text files from a directory as instruction records (once)."""
        if not seed_dir or not seed_dir.is_dir():
            return 0
        if self._records:
            return 0
        seeded = 0
        for path in sorted(seed_dir.iterdir()):
            if not path.is_file() or path.name.startswith("."):
                continue
            if path.suffix.lower() not in {".txt", ".md"}:
                log.info("Skip seed %s (not plain text)", path.name)
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace").strip()
            except Exception as exc:  # noqa: BLE001
                log.warning("Could not read seed %s: %s", path.name, exc)
                continue
            if text:
                added = self.add_instruction(text)
                seeded += 1
                log.info("Seeded %s (%d chunks)", path.name, added)
        if seeded:
            self.save()
            self._ensure_matrix()
        return seeded

    # ------------------------------------------------------------------
    # Read / recall
    # ------------------------------------------------------------------
    def count(self) -> int:
        return len(self._records)

    @staticmethod
    def _relevance_terms(text: str) -> set[str]:
        terms = set()
        for tok in _TOKEN_RE.findall((text or "").lower()):
            if len(tok) >= 3:
                terms.add(tok)
        return terms

    def select(self, query: str, extra: str = "", top_k: Optional[int] = None) -> list[dict]:
        """Surfaced memory for a query.

        The anchor core (a large share of the slots) is the direct, relevant
        match. The remaining slots are deliberately fuzzy: temperature-weighted
        sampling over the embedding space with a live seed (GPU temperature,
        model hash, context hash, a coarse time bucket), plus near-seed hops so
        tangential-but-true connections can come to the mind. The seed is not
        broadly reproducible: a similar situation tends to surface a similar
        set — never the exact same one, deliberately.
        """
        records = self._records
        if not records:
            return []
        k = top_k or self.top_k

        query_terms = self._relevance_terms(query) | self._relevance_terms(extra)
        now = int(time.time())

        self._ensure_matrix()

        scored: list[tuple[float, float, int, dict]] = []
        for idx, rec in enumerate(records):
            text = rec.get("text", "")
            score = 0.0
            semantic = 0.0
            if query_terms:
                ltext = text.lower()
                for t in query_terms:
                    if t in ltext:
                        score += 3.0
            if self._matrix_np is not None and query_terms:
                qv = self._embed_one(query + "\n" + extra)
                if qv is not None:
                    row = self._matrix_np[idx]
                    try:
                        denom = (float(np.linalg.norm(row)) *
                                 float(np.linalg.norm(qv))) or 1.0
                        semantic = float(np.dot(row, qv)) / denom
                    except Exception:  # noqa: BLE001
                        semantic = 0.0
                    if semantic > 0:
                        score += semantic * 6.0
                    else:
                        semantic = 0.0
            recency = 0.0
            try:
                age_hours = max(0, (now - int(rec.get("created_at", 0))) / 3600.0)
                recency = max(0.0, 4.0 - age_hours / 24.0)
            except Exception:  # noqa: BLE001
                recency = 0.0
            if query_terms and score > 0:
                score += recency
            scored.append((score, semantic, -idx, rec))

        scored.sort(reverse=True)
        matched = [tup for tup in scored if tup[0] > 0]
        if not matched:
            # Default to recency ordering when nothing matched lexically.
            return [rec for _s, _sem, _neg_idx, rec in scored[:k]]

        if not self.fuzzy or k <= 1:
            return [rec for _s, _sem, _neg_idx, rec in matched[:k]]

        # Anchor core: the direct connection is not left to chance.
        anchor_n = max(1, int(k * 0.6))
        core = matched[:anchor_n]
        pool = matched[anchor_n:]
        if not pool:
            return [rec for _s, _sem, _neg_idx, rec in core[:k]]

        rng = random.Random(self._live_seed(query, extra))
        slots = k - len(core)
        picks: list[tuple[float, float, int, dict]] = []
        remaining = [t for t in pool if t[2] not in {t[2] for t in core}]
        for _ in range(slots):
            if not remaining:
                break
            max_score = max(t[0] for t in remaining) or 1.0
            weights = [
                math.exp((t[0] - max_score) / self.fuzzy_temp)
                for t in remaining
            ]
            total = sum(weights) or 1.0
            roll = rng.random() * total
            acc = 0.0
            stop = 0
            for i, w in enumerate(weights):
                acc += w
                if acc >= roll:
                    stop = i
                    break
            picks.append(remaining.pop(stop))

        # Near-seed geometric hops: pull in a tangential-but-true neighbor of
        # a surfaced node from the embedding space (the shape of the memory).
        hops = 0
        try:
            while hops < self.fuzzy_hops and picks:
                t = picks[hops] if hops < len(picks) else picks[-1]
                pos = -t[2]
                nidx = self._nearest_embedding(
                    pos, {(-p[2]) for p in core} | {(-p[2]) for p in picks}
                )
                if nidx is None:
                    break
                picks[hops] = (t[0] * 0.5, 0.0, -nidx, records[nidx])
                hops += 1
        except Exception:  # noqa: BLE001
            pass

        selected = core + picks
        selected.sort(reverse=True)
        return [rec for _s, _sem, _neg_idx, rec in selected[:k]]

    def _nearest_embedding(self, idx: int, exclude: set[int]) -> Optional[int]:
        """Nearest memory node (by cosine) not already surfaced; the shape hop."""
        if self._matrix_np is None or self._matrix_np.shape[0] != len(self._records):
            return None
        row = self._matrix_np[idx]
        best: Optional[int] = None
        best_sim = 0.35
        for j in range(len(self._records)):
            if j == idx or j in exclude:
                continue
            rj = self._matrix_np[j]
            denom = (float(np.linalg.norm(rj)) * float(np.linalg.norm(row))) or 1.0
            sim = float(np.dot(rj, row)) / denom
            if sim > best_sim:
                best_sim = sim
                best = j
        return best

    def _live_seed(self, query: str, extra: str = "") -> int:
        """A seed that is *mostly* contextual but never deliberately fixed.

        Similar situations feed similar material (model, query, coarse time
        bucket, GPU temperature) into the same hash, so a related memory tends
        to resurface — but the exact draw is not reproducible on purpose.
        An explicit MIND_MEMORY_FUZZY_SEED overrides for tests.
        """
        if self.fuzzy_seed:
            try:
                return int(self.fuzzy_seed, 0)
            except ValueError:
                pass
        temps = "|".join(self._read_gpu_temps()) or "?"
        material = "|".join([
            temps,
            self.model,
            self.backend,
            hashlib.sha256((query + "\n" + extra).encode("utf-8", "ignore")).hexdigest(),
            str(int(time.time()) // 300),
        ])
        digest = hashlib.blake2b(material.encode("utf-8", "ignore"), digest_size=8)
        return int.from_bytes(digest.digest(), "little")

    @staticmethod
    def _read_gpu_temps() -> list[str]:
        """Live GPU package temperatures (AMD hwmon); best-effort on a battery."""
        temps: list[str] = []
        try:
            for path in glob.glob("/sys/class/drm/card*/device/hwmon/hwmon*/temp*_input"):
                with open(path, "r", encoding="utf-8") as fh:
                    temps.append(fh.read().strip())
            if not temps:
                for path in glob.glob("/sys/class/thermal/thermal_zone*/temp"):
                    with open(path, "r", encoding="utf-8") as fh:
                        temps.append(fh.read().strip())
        except Exception:  # noqa: BLE001
            pass
        return temps[:4]

    def build_memory_block(
        self, query: str = "", extra: str = "", top_k: Optional[int] = None
    ) -> Optional[str]:
        """Surfaced memory block for the system prompt.

        The selection machinery never enters the context window: no "we looked
        up N nodes and filtered to K" framing, no relevance counts. The block
        simply presents what came to mind — the model does the judgment of
        what holds.
        """
        records = self._records
        if not records:
            return None
        selected = self.select(query, extra, top_k=top_k)
        if not selected:
            return None
        lines = ["--- Memory ---"]
        for rec in selected:
            source = _SOURCE_LABELS.get(rec.get("source", ""), "unknown")
            text = rec.get("text", "").strip().replace("\n", " ").strip()
            if len(text) > 500:
                text = text[:497].rstrip() + "…"
            lines.append(f"[{source}] {text}")
        return "\n".join(lines)

    def stats(self) -> dict:
        return {
            "mem_nodes": len(self._records),
            "backend": self.backend,
            "embed_error": self._embed_error,
        }

    def shutdown(self) -> None:
        try:
            self._jet.put(None)
        except Exception:  # pragma: no cover
            pass


# ---------------------------------------------------------------------------
# Module-level singleton helpers (keep the daemon's word small)
# ---------------------------------------------------------------------------
def get_memory(
    memory_dir: Optional[Path] = None,
    **kwargs,
) -> MindMemory:
    return MindMemory(memory_dir=memory_dir, **kwargs)