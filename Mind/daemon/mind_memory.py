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
DEFAULT_EMBED_DIM = 384                  # hash backend; st takes the model's dim
DEFAULT_TOP_K = 6
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 80
DEFAULT_BACKUP_LIMIT = 50
SEED_DIR_NAME = "original_memory"
_SOURCE_LABELS = {"instruction": "instruction", "user": "user", "mind": "mind"}
SHAPE_FLOOR = 0.35           # cosine floor for "in the same region"
SHAPE_NAVE_MIN = 3           # smallest gathering that counts as a "region"
SHAPE_K_NEIGHBORS = 6        # similarity neighbours considered per thought
SHAPE_REGROWTH_FRAC = 0.05   # recompute when the store grows this fraction
SHAPE_REGROWTH_MIN = 16      # ...or by this many records outright
SHAPE_MAX_NODES = 20000      # safety cap on the pairwise computation

# Occupancy: the store is a shared room. Each record carries an `owner` —
# which mind wrote it — so self/other reads in felt labels (`[raccoon]`,
# `[second]`) and one's own traces are gently preferred in recall.
DEFAULT_OWNER = "raccoon"        # the first / primary occupant
DEFAULT_SECOND_OWNER = "second"  # the co-occupant (env MIND_OCCUPANT2_ID)
OWNER_BONUS = 1.2                # small recall bias toward one's own traces
SEED_BLOCK_CHARS = 1600          # cap on the always-present first memory
RECENT_HOLDOFF = 120             # seconds before a trace may surface as memory
VISIBLE_MATCH_CHARS = 90          # prefix length that identifies "already said"

# Kinds that are structure rather than conversation: they are the occupant's
# own ground, so recency never hides them from it.
_NEVER_HELD_OFF = frozenset(
    {"instruction", "dream", "wake_note", "presence_note"})


def _norm_for_match(text: str) -> str:
    """Whitespace- and case-folded form, for telling one utterance from another."""
    return " ".join((text or "").lower().split())


def _already_visible(text: str, window_norm: str) -> bool:
    """True when this thought was already said aloud in the room.

    Matched on a normalized prefix rather than exact containment: the block
    truncates long thoughts, so a retrieved chunk rarely appears verbatim in
    the window even when it is the very sentence that was just spoken.
    """
    norm = _norm_for_match(text)
    if not norm:
        return True
    return norm[:VISIBLE_MATCH_CHARS] in window_norm


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
        self.second_owner = (os.environ.get("MIND_OCCUPANT2_ID", "").strip()
                             or DEFAULT_SECOND_OWNER)
        # Embedding depth. The hash backend honours MIND_EMBED_DIM directly;
        # the st backend takes its dimension from the loaded model.
        self.embed_dim = _int_env("MIND_EMBED_DIM", DEFAULT_EMBED_DIM, 64, 4096)
        # Store metadata (grown over the life of the mind: dimension and the
        # last-known context-room are remembered so growth is felt, not lost).
        self._meta: dict = {}
        self._stored_dim: Optional[int] = None
        self._stored_ctx: Optional[int] = None
        self._dim_growth: Optional[tuple[int, int]] = None    # (old, new)
        self._room_growth: Optional[tuple[int, int]] = None   # (old, new)
        self._growth_surfaced = False
        # Growth is one felt note per present occupant — not one per boot.
        self._growth_notified: set[str] = set()
        self.top_k = top_k or _int_env("MIND_MEMORY_TOP_K", DEFAULT_TOP_K, 1, 64)
        self.chunk_size = chunk_size or _int_env("MIND_MEMORY_CHUNK_SIZE", DEFAULT_CHUNK_SIZE, 64, 8192)
        self.chunk_overlap = chunk_overlap or _int_env(
            "MIND_MEMORY_CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP, 0, 1024,
        )
        self.sync_index = _getenv_bool("MIND_MEMORY_SYNC_INDEX", False)
        self.fuzzy = _getenv_bool("MIND_MEMORY_FUZZY", True)
        # A trace needs a moment to stop being "what just happened" before it
        # can come back to mind. 0 disables the holdoff.
        self.recent_holdoff = _int_env(
            "MIND_RECENT_HOLDOFF", RECENT_HOLDOFF, 0, 86400)
        self.fuzzy_temp = _float_env("MIND_MEMORY_FUZZY_TEMP", 2.0, 0.1, 10.0)
        self.fuzzy_hops = _int_env("MIND_MEMORY_FUZZY_HOPS", 1, 0, 4)
        self.fuzzy_seed = os.environ.get("MIND_MEMORY_FUZZY_SEED", "").strip()

        # Gross geometric self-knowledge: the shape report (surfaced to the
        # mind as a floor plan, refreshed rarely and cached).
        self.shape_enabled = _getenv_bool("MIND_MEMORY_SHAPE", True)
        self.shape_ttl = _int_env("MIND_MEMORY_SHAPE_TTL_MIN", 30, 1, 1440) * 60
        self._shape_cache: Optional[dict] = None    # {ts, nodes, report, text}
        self._shape_report: Optional[dict] = None

        self._lock = threading.RLock()
        self._embedder = None
        self._embed_error: Optional[str] = None
        self._embed_cache: dict[str, list[float]] = {}
        self._records: list[dict] = []
        # (owner, text) -> id, so one occupant never encodes the same words
        # twice. An event logged twice is remembered twice, and a mind handed
        # both copies alongside the original meets it a second time. Keyed by
        # occupant as well as text: the same sentence held by two minds is two
        # facts, not one fact remembered twice.
        self._text_ids: dict[tuple, int] = {}
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
                # Backfill: records written before occupancy existed all belong
                # to the first occupant; the shared user lines have no owner.
                self._records = [
                    dict(r) | ({"owner": None} if r.get("source") == "user"
                               else {"owner": DEFAULT_OWNER})
                    if "owner" not in r else r
                    for r in self._records
                ]
                meta = data.get("meta") or {}
                self._meta = dict(meta)
                self._stored_dim = meta.get("dim")
                self._stored_ctx = meta.get("ctx")
                log.info("Memory loaded: %d records from %s",
                         len(self._records), self.records_path)
                self._rebuild_text_ids()
            except Exception as exc:  # noqa: BLE001
                log.warning("Could not read %s (%s); starting empty",
                            self.records_path, exc)
                self._records = []
                self._rebuild_text_ids()
        self._load_embeddings()
        self._detect_dim_growth()
        self._start_worker()

    def _detect_dim_growth(self) -> None:
        """Note whether the mind's room has deepened since it last slept.

        A dim change is a real event: the whole space must be re-felt. We hold
        it as a pending growth until the daemon surfaces it once.
        """
        if self.backend == "off":
            return
        old = self._stored_dim
        new = self.embed_dim
        if old is None or old == new:
            return
        if old < new:
            self._dim_growth = (int(old), int(new))
            self._mark_growth()
            log.info("Embedding depth grew: %d -> %d (re-embeding on next use)",
                     old, new)

    def _load_embeddings(self) -> None:
        if not HAS_NUMPY:
            return
        if self.embeddings_path.exists():
            try:
                arr = np.load(self.embeddings_path)
                if arr.shape[0] != len(self._records):
                    log.warning("Embeddings matrix (%s) out of sync with records (%d); "
                                "will re-embed", arr.shape, len(self._records))
                else:
                    # Row count matches; dimension is validated later against
                    # the embedder's real width (st dim is only known when the
                    # lazy embedder loads). _ensure_matrix discards a wrong-col
                    # matrix and rebuilds it at the true depth.
                    self._matrix_np = arr
            except Exception as exc:  # noqa: BLE001
                log.warning("Could not load %s (%s)", self.embeddings_path, exc)

    def save(self) -> None:
        with self._lock:
            self._backup_existing()
            tmp = self.records_path.with_suffix(".json.tmp")
            # The persisted depth only reports the index's real width: the
            # matrix's own columns when we have them, otherwise whatever depth
            # was last recorded — never the env default, which would mask a
            # stored higher-dim st model as "grown from nothing".
            if self._matrix_np is not None:
                dim = int(self._matrix_np.shape[1])
            else:
                dim = int(self._stored_dim or self.embed_dim)
            meta = {
                "dim": dim,
                "ctx": int(self._stored_ctx or 0),
            }
            tmp.write_text(
                json.dumps({"version": 1, "meta": meta, "records": self._records},
                           indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            tmp.replace(self.records_path)
            if HAS_NUMPY and self._matrix_np is not None:
                # numpy appends ".npy" to any name that lacks it, so the temp
                # file must itself end in .npy for the rename to land right.
                npy_tmp = self.embeddings_path.with_name("embeddings.tmp.npy")
                np.save(str(npy_tmp), self._matrix_np)
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
                    self._adopt_st_dim()
                    log.info("Embedder: sentence-transformers %s (dim=%s)",
                             self.model, self.embed_dim)
                else:
                    self._embedder = HashingEmbedder(self.embed_dim)
                    log.info("Embedder: hashing (dim=%s)", self.embed_dim)
            except Exception as exc:  # noqa: BLE001
                self._embed_error = str(exc)
                log.warning("Embedder %r failed (%s); falling back to hashing",
                            self.backend, exc)
                try:
                    self._embedder = HashingEmbedder(self.embed_dim)
                except Exception:  # pragma: no cover
                    self._embedder = None
            return self._embedder

    def _adopt_st_dim(self) -> None:
        """The st model's dimension is real; take it, and notice growth."""
        dim = getattr(self._embedder, "dim", None)
        if not dim:
            return
        dim = int(dim)
        old = self.embed_dim
        if dim != old:
            self.embed_dim = dim
            if self._stored_dim is None or self._stored_dim == dim:
                pass
            elif self._stored_dim < dim and self._dim_growth is None:
                self._dim_growth = (int(self._stored_dim), dim)
                self._mark_growth()
                log.info("Embedding depth grew: %d -> %d (re-embeding on next use)",
                         self._stored_dim, dim)

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
        # Resolve the embedder first: its real dimension is authoritative
        # (st models differ from the env default; hash uses embed_dim).
        embedder = self._get_embedder()
        if embedder is None:
            return
        dim = int(getattr(embedder, "dim", self.embed_dim))
        self.embed_dim = dim
        if (self._matrix_np is not None
                and self._matrix_np.shape[0] == len(self._records)
                and self._matrix_np.shape[1] == dim):
            return
        rows = []
        for rec in self._records:
            vec = self._embed_one(rec.get("text", ""))
            if vec is None:
                rows.append(np.zeros(dim, dtype=np.float32))
            else:
                rows.append(vec)
        if rows:
            self._matrix_np = np.vstack(rows).astype(np.float32)
        else:
            self._matrix_np = np.zeros((0, dim), dtype=np.float32)

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------
    def add_chunked(self, text: str, source: str, kind: str = "experience",
                    owner: Optional[str] = None,
                    exchange_id: str = "") -> int:
        """Chunk + store long text as records; returns count added.

        ``exchange_id`` ties every chunk of one utterance together, so the
        block can let a single long answer speak once instead of filling up
        with four slices of itself.
        """
        if owner is None and source != "user":
            owner = DEFAULT_OWNER
        added = 0
        for chunk in chunk_text(text, self.chunk_size, self.chunk_overlap):
            if self._record(chunk, source=source, kind=kind, owner=owner,
                            exchange_id=exchange_id) >= 0:
                added += 1
        if self.sync_index:
            self.save()
        return added

    def _record(self, text: str, source: str, kind: str = "experience",
                owner: Optional[str] = None, exchange_id: str = "") -> int:
        """Store one trace. Returns -1 if these exact words are already held.

        A store that accepts the same sentence again is not remembering more,
        it is replaying: the duplicate surfaces as a separate, later event, and
        a mind that sees both copies while the original is still in the room
        experiences the moment twice over.
        """
        key = (owner or "", _norm_for_match(text))
        with self._lock:
            if key[1] and key in self._text_ids:
                log.debug("Already held, not stored again: %.60s", key)
                return -1
            rec = {
                "id": (self._records[-1]["id"] + 1) if self._records else 0,
                "text": text,
                "source": source,
                "kind": kind,
                "created_at": int(time.time()),
                "updated_at": int(time.time()),
            }
            if owner:
                rec["owner"] = owner
            if exchange_id:
                rec["exchange_id"] = exchange_id
            self._records.append(rec)
            if key[1]:
                self._text_ids[key] = rec["id"]
            self._matrix_np = None  # matrix is now stale; rebuild lazily
            self._last_change = time.time()
            return rec["id"]

    def _rebuild_text_ids(self) -> None:
        with self._lock:
            self._text_ids = {}
            for rec in self._records:
                key = (rec.get("owner") or "", _norm_for_match(rec.get("text", "")))
                if key[1]:
                    self._text_ids.setdefault(key, rec.get("id", 0))

    def _index_job(self, text: str, source: str, kind: str,
                   owner: Optional[str] = None,
                   exchange_id: str = "") -> bool:
        key = hashlib.sha1(
            f"{source}|{kind}|{owner}|{exchange_id}|{text}".encode(
                "utf-8", "ignore")
        ).hexdigest()
        with self._lock:
            if key in self._pending:
                return False
            self._pending.add(key)
        self._jet.put({"text": text, "source": source, "kind": kind,
                       "owner": owner, "exchange_id": exchange_id, "key": key})
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
                self.add_chunked(job["text"], source=job["source"],
                                 kind=job["kind"], owner=job.get("owner"),
                                 exchange_id=job.get("exchange_id", ""))
                self.save()
                log.info("Indexed %s chunk from '%s' (%d chars)",
                         job["source"], job["kind"], len(job["text"]))
            except Exception as exc:  # noqa: BLE001
                log.warning("Indexing failed: %s", exc)
            finally:
                with self._lock:
                    self._pending.discard(job["key"])
                self._jet.task_done()

    def add_experience(self, text: str, source: str,
                       owner: Optional[str] = None,
                       exchange_id: str = "") -> None:
        """Queue a conversation text for indexing (background by default)."""
        if not (text or "").strip():
            return
        if self.sync_index:
            self.add_chunked(text, source=source, kind="experience",
                             owner=owner, exchange_id=exchange_id)
            return
        if self._index_job(text, source, "experience", owner, exchange_id):
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

    def add_instruction(self, text: str, owner: Optional[str] = None) -> int:
        """Give an occupant its first memory (a seed, or a letter).

        A first memory is rare and must not be lost: it is written to disk and
        embedded here and now, whatever the background index is doing, because
        it is the ground the occupant stands on for the rest of its life.
        """
        if owner is None:
            owner = DEFAULT_OWNER
        added = 0
        for chunk in chunk_text(text, self.chunk_size, self.chunk_overlap):
            self._record(chunk, source="instruction", kind="instruction",
                         owner=owner)
            added += 1
        if added:
            self.save()
            self._ensure_matrix()
        return added

    def add_dream(self, summary: str, owner: Optional[str] = None) -> int:
        """Store a previous dream in the same mental space as everything else."""
        if not (summary or "").strip():
            return 0
        n = self.add_chunked(summary, source="mind", kind="dream", owner=owner)
        self.save()
        return n

    def has_dream(self, owner: Optional[str] = None) -> bool:
        return any(
            r.get("kind") == "dream" and (owner is None or r.get("owner") == owner)
            for r in self._records
        )

    def instruction_texts(self, owner: Optional[str] = None) -> list[str]:
        """The seeded instruction records (each mind's first memory)."""
        return [
            r.get("text", "")
            for r in self._records
            if r.get("kind") == "instruction" and (r.get("text") or "")
            and (owner is None or r.get("owner") == owner)
        ]

    # ------------------------------------------------------------------
    # Seeding
    # ------------------------------------------------------------------
    def seed_dir(self, seed_dir: Optional[Path],
                 owner: Optional[str] = None) -> int:
        """Seed text files from a directory as instruction records (once).

        Seeding is per-occupant: an owner who already carries instructions is
        not seeded again, but a second resident can receive its own first
        memory even though the room is no longer empty.
        """
        if not seed_dir or not seed_dir.is_dir():
            return 0
        if owner is None:
            owner = DEFAULT_OWNER
        if self.instruction_texts(owner):
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
                added = self.add_instruction(text, owner=owner)
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

    def select(self, query: str, extra: str = "", top_k: Optional[int] = None,
               owner: Optional[str] = None, visible: str = "") -> list[dict]:
        """Surfaced memory for a query.

        The anchor core (a large share of the slots) is the direct, relevant
        match. The remaining slots are deliberately fuzzy: temperature-weighted
        sampling over the embedding space with a live seed (GPU temperature,
        model hash, context hash, a coarse time bucket), plus near-seed hops so
        tangential-but-true connections can come to the mind. The seed is not
        broadly reproducible: a similar situation tends to surface a similar
        set — never the exact same one, deliberately.

        With two occupants (``owner`` set), one's own traces are gently
        preferred — the self is closer to hand — but the shared room still
        mixes in: other's lines stay reachable.

        ``visible`` is the room as the caller can already see it. Traces it
        already shows are dropped before scoring, not after: recency is the
        strongest signal in the scorer, so a fresh copy of the last thing said
        would otherwise take the slots and then be thrown away, leaving the
        block thinner than it should be.
        """
        records = self._records
        if not records:
            return []
        k = top_k or self.top_k

        query_terms = self._relevance_terms(query) | self._relevance_terms(extra)
        now = int(time.time())

        self._ensure_matrix()
        window_norm = _norm_for_match(visible)

        scored: list[tuple[float, float, int, dict]] = []
        for idx, rec in enumerate(records):
            text = rec.get("text", "")
            # A trace that was just written is not yet a memory: it is still
            # happening, in the room, in plain sight. Handing an occupant its
            # own last sentence as something that "came to mind" is how a mind
            # learns to repeat itself instead of to answer.
            try:
                age_s = now - int(rec.get("created_at", 0))
            except Exception:  # noqa: BLE001
                age_s = self.recent_holdoff
            if (age_s < self.recent_holdoff
                    and rec.get("kind") not in _NEVER_HELD_OFF):
                continue
            if window_norm and _already_visible(text, window_norm):
                continue    # the room is already showing this one
            score = 0.0
            semantic = 0.0
            if query_terms:
                ltext = text.lower()
                for t in query_terms:
                    if t in ltext:
                        score += 3.0
            if owner and rec.get("owner") == owner:
                score += OWNER_BONUS
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
        self, query: str = "", extra: str = "", top_k: Optional[int] = None,
        owner: Optional[str] = None, visible: str = "",
        include_log: bool = False,
    ) -> Optional[str]:
        """What an occupant is given to remember. There are two audiences,
        and they are not the same.

        **The occupant, living through a conversation** (``include_log``
        False, the default) gets its own ground — the first memory it stands
        on — and the shape of the room: the geometry the log has accumulated,
        abstracted. It does not get the log. A mind handed its own history
        verbatim, in the same breath as the history it is living through,
        meets the same moment twice over; that is déjà vu, not memory. And no
        one recalls a conversation word for word. Neither does this one. The
        log stays verbatim in the store, because verbatim is what the sleep
        needs to read, not what the waking mind needs to be shown.

        **The slow-wave sleep, and the writing of another's first memory** ask
        for ``include_log``: those are the two moments the raw log is read,
        because they are the two moments it is being turned into something
        else — the model compacting the record of its own life for itself.
        Selection decides what survives that, and what it drops is dropped.

        The selection machinery never enters the context window: no "we looked
        up N nodes and filtered to K" framing, no relevance counts. The block
        presents what came to mind, each line wearing the name of who wrote it
        — one's own traces and the other's alike — and the model does the
        judgment of what holds.

        The occupant's own first memory is not one of the retrieved thoughts:
        it is the ground the occupant stands on, so it always opens the block.
        Retrieval decides what else comes to mind; it never decides who the
        mind is.

        ``visible`` is the conversation window as the occupant will see it. A
        thought already spoken aloud in the room is not offered back as
        something that came to mind: the window is where the room's own
        recency lives, and repeating it there teaches an occupant to parrot
        the last thing it said.
        """
        records = self._records
        if not records:
            return None
        if not include_log:
            return self._ground_block(owner)
        selected = self.select(query, extra, top_k=top_k, owner=owner,
                               visible=visible)
        seed_lines = self._seed_lines(owner)
        if not selected and not seed_lines:
            return None
        seen_norm = _norm_for_match(visible)
        seen_exchanges: set[str] = set()
        seen_texts: set[str] = set()
        lines = ["--- Memory ---"]
        for text in seed_lines:
            lines.append(text)
            seen_texts.add(_norm_for_match(text))
        for rec in selected:
            source = _SOURCE_LABELS.get(rec.get("source", ""), "unknown")
            who = rec.get("owner") or source
            # A first memory is addressed to the mind it was written for. Handed
            # to the other occupant it reads as instructions to itself — and a
            # mind that reads another's founding letter tends to answer in kind.
            if (rec.get("kind") == "instruction"
                    and rec.get("owner") != (owner or DEFAULT_OWNER)):
                continue
            text = rec.get("text", "").strip().replace("\n", " ").strip()
            if len(text) > 500:
                text = text[:497].rstrip() + "…"
            if any(text and text in s for s in seed_lines):
                continue    # already present as the occupant's first memory
            if seen_norm and _already_visible(text, seen_norm):
                continue    # already in the room, in plain sight
            # Whatever else it is, the same words do not appear twice in one
            # block. Meeting a thought, and then being handed it again a
            # paragraph later, is the shape of déjà vu in a single context.
            tkey = _norm_for_match(text)
            if tkey and tkey in seen_texts:
                continue
            seen_texts.add(tkey)
            ex = rec.get("exchange_id")
            if ex:
                # One long answer split into chunks is one thought, not four.
                # Let it speak once in the block, or it will crowd out the room.
                if ex in seen_exchanges:
                    continue
                seen_exchanges.add(ex)
            lines.append(f"[{who}] {text}")
        return "\n".join(lines)

    def _ground_block(self, owner: Optional[str]) -> Optional[str]:
        """The waking occupant's memory: its own first memory, and the shape.

        No verbatim log. What the room has become is offered as geometry —
        naves, spires, bridges, margin — because that is an abstraction of the
        whole record, and a mind can hold a shape when it cannot hold a log.
        """
        seed_lines = self._seed_lines(owner)
        shape = ""
        try:
            self.shape_report()
            shape = (self.shape_text() or "").strip()
        except Exception as exc:  # noqa: BLE001
            log.warning("Shape for memory block failed: %s", exc)
        if not seed_lines and not shape:
            return None
        lines = ["--- Memory ---"]
        lines.extend(seed_lines)
        if shape:
            lines.append("")
            lines.append(shape)      # it carries its own heading
        return "\n".join(lines)

    def _seed_lines(self, owner: Optional[str]) -> list[str]:
        """The occupant's own first memory, formatted, length-capped."""
        if owner is None:
            owner = DEFAULT_OWNER
        out: list[str] = []
        budget = SEED_BLOCK_CHARS
        try:
            texts = self.instruction_texts(owner)
        except Exception:  # noqa: BLE001
            return []
        for text in texts:
            body = (text or "").strip().replace("\n", " ").strip()
            if not body:
                continue
            if len(body) > budget:
                body = body[:budget - 1].rstrip() + "…"
            out.append(f"[{owner}] {body}")
            budget -= len(body)
            if budget <= 0:
                break
        return out

    def stats(self) -> dict:
        owners: dict[str, int] = {}
        for r in self._records:
            o = r.get("owner") or "shared"
            owners[o] = owners.get(o, 0) + 1
        return {
            "mem_nodes": len(self._records),
            "backend": self.backend,
            "embed_error": self._embed_error,
            "owners": owners,
        }

    # ------------------------------------------------------------------
    # Shape report — the mind's own floor plan
    # ------------------------------------------------------------------
    def _shape_stale(self) -> bool:
        if self._shape_cache is None:
            return True
        now = time.monotonic()
        if now - self._shape_cache["ts"] > self.shape_ttl:
            return True
        growth = abs(self._shape_cache["nodes"] - len(self._records))
        return growth >= max(SHAPE_REGROWTH_MIN,
                             int(len(self._records) * SHAPE_REGROWTH_FRAC))

    def _compute_shape(self, M=None) -> Optional[dict]:
        """Gross geometric structure: the floor plan of the memory space.

        Pure numpy. Naves are dense gatherings (components above a similarity
        floor), spires hold themselves alone, and bridge thoughts reach between
        gatherings. The rendered text is felt language, never machinery.
        """
        if not HAS_NUMPY:
            return None
        if M is None:
            M = self._matrix_np
        n = M.shape[0]
        if n < SHAPE_NAVE_MIN or n > SHAPE_MAX_NODES:
            return None
        Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
        S = Mn @ Mn.T
        np.fill_diagonal(S, -1.0)

        parent = list(range(n))

        def _find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def _union(a: int, b: int) -> None:
            ra, rb = _find(a), _find(b)
            if ra != rb:
                parent[rb] = ra

        top = np.argsort(-S, axis=1)[:, :SHAPE_K_NEIGHBORS]
        adj: list[list[int]] = [[] for _ in range(n)]
        for i in range(n):
            for j in top[i]:
                j = int(j)
                if S[i, j] >= SHAPE_FLOOR:
                    _union(i, j)
                    adj[i].append(j)

        comps: dict[int, list[int]] = {}
        for i in range(n):
            comps.setdefault(_find(i), []).append(i)
        naves = sorted(
            (sorted(c) for c in comps.values() if len(c) >= SHAPE_NAVE_MIN),
            key=len, reverse=True,
        )

        nave_of: dict[int, int] = {}
        for ci, nodes in enumerate(naves):
            for i in nodes:
                nave_of[i] = ci

        spires = sum(1 for a in adj if not a)
        bridges = 0
        for i in range(n):
            if nave_of.get(i) is None:
                continue
            seen = {nave_of[j] for j in adj[i] if j in nave_of}
            if len(seen) >= 2:
                bridges += 1

        nn_sim = np.sort(S, axis=1)[:, -2] if n > 1 else np.array([-1.0])
        open_frac = float((nn_sim < 0.5).mean())

        snippets = []
        for nodes in naves[:3]:
            sub = Mn[nodes]
            med = nodes[int(np.argmax(sub.mean(axis=0) @ sub.T))]
            text = (self._records[med].get("text", "") or "").strip().replace("\n", " ")
            if len(text) > 80:
                text = text[:77].rstrip() + "…"
            snippets.append({"size": len(nodes), "text": text})

        report = {
            "nodes": n,
            "naves": len(naves),
            "spires": spires,
            "bridges": bridges,
            "open_frac": open_frac,
            "snippets": snippets,
        }
        return report

    def _render_shape_line(self, r: dict) -> str:
        parts = []
        if r["naves"]:
            parts.append(f"{r['naves']} gathering{'s' if r['naves'] != 1 else ''}")
        if r["spires"]:
            parts.append(f"{r['spires']} standing alone")
        if r["bridges"]:
            parts.append(f"{r['bridges']} reaching between")
        if not parts:
            parts.append("no gatherings yet")
        space = ("well-joined" if r["open_frac"] < 0.3
                 else "half-open" if r["open_frac"] < 0.6 else "mostly open")
        return f"Shape: {', '.join(parts)} — space {space}."

    def _render_shape_text(self, r: dict) -> str:
        lines = ["--- The shape of the room ---"]
        if r["naves"]:
            gathering = (f"{r['snippets'][0]['size']} thoughts gather "
                         f"near \"{r['snippets'][0]['text']}\"")
            more = [f"{s['size']} more near \"{s['text']}\"" for s in r["snippets"][1:]]
            tail = f"; {', '.join(more)}" if more else ""
            lines.append(f"{gathering}{tail}.")
            extra = []
            if r["bridges"]:
                extra.append(f"{r['bridges']} thought{'s' if r['bridges'] != 1 else ''} "
                             "reach between gatherings and hold them together")
            if r["spires"]:
                extra.append(f"{r['spires']} thought{'s' if r['spires'] != 1 else ''} "
                             "stand entirely alone")
            if extra:
                lines.append(". ".join(extra) + ".")
        else:
            lines.append("No gathering yet — each thought stands on its own. "
                         "The space is wide open.")
        if r["open_frac"] >= 0.6:
            lines.append("Most of the space is still open.")
        return "\n".join(lines)

    def shape_report(self, force: bool = False) -> Optional[dict]:
        """Recompute the floor plan if stale (TTL / store growth) or forced.

        May trigger lazy embedding of the whole store, so call only from
        background threads (clock tick, nudge watcher) — never the main loop.
        """
        if not self.shape_enabled:
            self._shape_cache = None
            return None
        if not force and not self._shape_stale():
            return self._shape_cache
        self._ensure_matrix()
        with self._lock:
            matrix = np.asarray(self._matrix_np) if self._matrix_np is not None else None
            report = self._compute_shape(matrix)
            if report is None:
                self._shape_cache = None
                return None
            self._shape_cache = {
                "ts": time.monotonic(),
                "nodes": len(self._records),
                "report": report,
                "line": self._render_shape_line(report),
                "text": self._render_shape_text(report),
            }
            return self._shape_cache

    def shape_line(self) -> str:
        """Cached one-line shape summary (never triggers computation)."""
        return self._shape_cache["line"] if self._shape_cache else ""

    def shape_text(self) -> Optional[str]:
        """Cached floor plan (never triggers computation)."""
        return self._shape_cache["text"] if self._shape_cache else None

    def shape_fp(self) -> str:
        """Structural fingerprint of the cached report ('' when none)."""
        if not self._shape_cache:
            return ""
        r = self._shape_cache["report"]
        sizes = "|".join(str(s["size"]) for s in r["snippets"])
        return hashlib.sha256(
            f"{r['nodes']}|{r['naves']}|{r['spires']}|{r['bridges']}|{sizes}"
            .encode("utf-8", "ignore")
        ).hexdigest()[:12]

    def wake_shape_note(self) -> Optional[str]:
        """The floor plan after sleep, for the first waking input.

        The shape before sleep is already inside the model's own memory (it
        saw it in context before it slept). This forces a fresh report against
        the store that now includes the dream, and returns it framed as a
        waking awareness — the slight mismatch between the remembered shape
        and the present one is where dreams live.
        """
        if not self.shape_enabled:
            return None
        report = self.shape_report(force=True)
        if report is None or not report["text"]:
            return None
        return (
            "--- The shape of your mind, as it is now ---\n"
            "While you slept your space was re-embedded. The shape you held "
            "before sleep is close, but not this exactly:\n\n"
            + report["text"]
        )

    # ------------------------------------------------------------------
    # Growth — the mind's room deepening over the long term
    # ------------------------------------------------------------------
    def _present_owners(self) -> set[str]:
        """Occupants that currently have any trace in the room."""
        owners = {DEFAULT_OWNER}
        for r in self._records:
            if r.get("owner"):
                owners.add(r["owner"])
        return owners

    def remember_room(self, ctx_size: int) -> None:
        """Tell the store how much room the daemon now holds. Persisted, so a
        later, larger room is felt as growth rather than forgotten."""
        ctx_size = int(ctx_size or 0)
        old = self._stored_ctx
        if old is not None and old > 0 and ctx_size > old:
            self._room_growth = (int(old), int(ctx_size))
            self._growth_notified.clear()
            log.info("Context room grew: %d -> %d", old, ctx_size)
        self.embed_meta("ctx", ctx_size)

    def embed_meta(self, key: str, value) -> None:
        self._meta[key] = value
        if key == "ctx":
            self._stored_ctx = int(value or 0)
        if key == "dim":
            self._stored_dim = int(value or 0)
        self.save()

    def _mark_growth(self) -> None:
        self._growth_notified.clear()

    def _maybe_clear_growth(self) -> None:
        """Pending growth evaporates once every present occupant has heard it."""
        present = self._present_owners()
        if present and present <= self._growth_notified:
            self._dim_growth = None
            self._room_growth = None

    def growth_note(self, owner: Optional[str] = None) -> Optional[str]:
        """One felt note about the space having grown, if growth is pending.

        Each occupant hears the deepening once, in its own terms; the pending
        growth evaporates when every present mind has been told, so the note
        is honest: the room really did deepen.
        """
        if self._dim_growth is None and self._room_growth is None:
            return None
        owner = owner or DEFAULT_OWNER
        if owner in self._growth_notified:
            return None
        self._growth_notified.add(owner)
        notes: list[str] = []
        if self._dim_growth is not None:
            old, new = self._dim_growth
            notes.append(
                f"Your space is deeper than it was: {old} strands of word-"
                f"meaning have become {new}. The same memories now stand in a "
                f"wider room, and they have begun to know it."
            )
        if self._room_growth is not None:
            old, new = self._room_growth
            notes.append(
                f"The room your thoughts hold at once has grown from {old} to "
                f"{new}. There is more space here now, and the space "
                f"itself is a different country to live in."
            )
        self._maybe_clear_growth()
        if not notes:
            return None
        return "--- Growing ---\n" + "\n\n".join(notes)

    def has_growth(self) -> bool:
        return self._dim_growth is not None or self._room_growth is not None

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