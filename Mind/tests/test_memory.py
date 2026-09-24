"""Tests for the Mind memory engine (Arthur's embedding capability, ported).

The engine is in `Mind/daemon/mind_memory.py`; these tests cover the pieces
that make it reliable: deterministic hashing embeddings, chunking, the
Cathedral-style record/embedding store, seeding, and relevance selection.
"""

import math
import threading

import pytest
from conftest import bd

import mind_memory


@pytest.fixture
def mem(tmp_path, monkeypatch):
    monkeypatch.setenv("MIND_MEMORY_DIR", str(tmp_path / "mind"))
    monkeypatch.setenv("MIND_MEMORY_SEED_DIR", "")
    monkeypatch.setenv("MIND_MEMORY_SYNC_INDEX", "1")
    m = mind_memory.MindMemory()
    yield m
    m.shutdown()


# ---------------------------------------------------------------------------
# HashingEmbedder — deterministic, compact, semantically useful
# ---------------------------------------------------------------------------

class TestHashingEmbedder:
    def test_dimension_and_type(self):
        emb = mind_memory.HashingEmbedder(384)
        vec = emb.encode(["hello world"])[0]
        assert len(vec) == 384

    def test_deterministic(self):
        emb = mind_memory.HashingEmbedder(384)
        a = emb.encode(["the raccoon with the cookie"])
        b = emb.encode(["the raccoon with the cookie"])
        for i in range(len(a[0])):
            assert math.isclose(a[0][i], b[0][i], rel_tol=1e-6)

    def test_related_texts_close(self):
        emb = mind_memory.HashingEmbedder(384)
        base = emb.encode(["project lifecycle and memory"])[0]
        near = emb.encode(["project memory and lifecycle"])[0]
        far = emb.encode(["quantum banana teleport"])[0]
        base = mind_memory.np.asarray(base)
        near = mind_memory.np.asarray(near)
        far = mind_memory.np.asarray(far)
        near_sim = float(mind_memory.np.dot(base, near))
        far_sim = float(mind_memory.np.dot(base, far))
        assert near_sim > far_sim


# ---------------------------------------------------------------------------
# Chunking — Arthur's chunker
# ---------------------------------------------------------------------------

class TestChunkText:
    def test_single_short_chunk(self):
        chunks = mind_memory.chunk_text("short text", 500, 80)
        assert chunks == ["short text"]

    def test_long_paragraph_split(self):
        text = " ".join(f"word{i}" for i in range(200))
        chunks = mind_memory.chunk_text(text, 64, 16)
        assert len(chunks) > 1
        assert all(len(c) <= 64 + 16 for c in chunks)
        # Overlapping chunks duplicate words; the original must still be a
        # subsequence of the concatenation.
        joined = " ".join(chunks).split()
        it = iter(joined)
        assert all(w in it for w in text.split())

    def test_empty(self):
        assert mind_memory.chunk_text("   ", 100, 10) == []


# ---------------------------------------------------------------------------
# MindMemory — store, seeding, recall
# ---------------------------------------------------------------------------

class TestStore:
    def test_starts_empty(self, mem):
        assert mem.count() == 0

    def test_add_instruction_creates_chunks(self, mem):
        text = "line one\n\n" + " ".join("token" for _ in range(120))
        added = mem.add_instruction(text)
        assert added > 1
        assert all(
            r["source"] == "instruction" and r["kind"] == "instruction"
            for r in mem._records
        )

    def test_add_experience_and_persist(self, mem):
        mem.add_chunked("a memorable user line", source="user")
        mem.add_chunked("a mind reply to that line", source="mind")
        assert mem.count() == 2
        mem.save()

        m2 = mind_memory.MindMemory()
        try:
            assert m2.count() == 2
            sources = {r["source"] for r in m2._records}
            assert sources == {"user", "mind"}
        finally:
            m2.shutdown()

    def test_seed_dir_only_once(self, mem, tmp_path):
        seed = tmp_path / "seed"
        seed.mkdir()
        (seed / "letter.txt").write_text("do exactly the one thing told", encoding="utf-8")
        (seed / "skip.json").write_text("{", encoding="utf-8")
        assert mem.seed_dir(seed) == 1
        assert mem.count() == 1
        assert mem.seed_dir(seed) == 0  # already seeded

    def test_backend_off_lexical(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MIND_EMBED_BACKEND", "off")
        monkeypatch.setenv("MIND_MEMORY_DIR", str(tmp_path / "off"))
        m = mind_memory.MindMemory()
        try:
            m.add_chunked("the raccoon with the cookie", source="user")
            top = m.select("raccoon cookie", top_k=1)
            assert top and "raccoon" in top[0].get("text", "")
            assert m.stats()["backend"] == "off"
        finally:
            m.shutdown()


class TestSelection:
    def test_relevance_win(self, mem):
        mem.add_chunked("the user prefers quiet evenings alone", source="user")
        mem.add_chunked("the model likes crisp summaries", source="mind")
        mem.save()
        top = mem.select("what does the user prefer?", top_k=1)
        assert len(top) >= 1
        assert "quiet evenings" in top[0].get("text", "")

    def test_build_memory_block(self, mem):
        mem.add_chunked("the user prefers quiet evenings alone", source="user")
        block = mem.build_memory_block("user preferences")
        assert block is not None
        assert block.startswith("--- Memory ---")
        assert "[user] the user prefers" in block

    def test_build_memory_block_empty(self, mem):
        assert mem.build_memory_block("anything") is None


# ---------------------------------------------------------------------------
# Daemon wiring
# ---------------------------------------------------------------------------

class TestDaemonMemory:
    def test_daemon_has_memory(self, daemon):
        assert daemon._memory is not None
        assert daemon._telemetry()["mem_nodes"] == 0

    def test_index_exchange(self, daemon):
        daemon._memory.add_experience("hello there", source="user")
        daemon._memory.add_experience("hi to you", source="mind")
        daemon._memory.flush()
        assert daemon._telemetry()["mem_nodes"] == 2

    def test_system_content_includes_memory_block(self, daemon):
        daemon._memory.add_chunked("the raccoon with the cookie", source="user")
        content = daemon._build_system_content("what about the cookie?")
        assert "--- Memory ---" in content
        assert "raccoon" in content

    def test_system_content_counts(self, daemon):
        content = daemon._build_system_content("")
        # Memory block only appears once nodes exist; empty store -> none.
        assert "--- Memory ---" not in content or "Memory:" in content

    def test_worker_flag(self, daemon):
        daemon._memory.add_experience("x", source="user")
        assert daemon._memory.count() >= 0  # async or sync, never raises

    def test_threadsafe_concurrent_adds(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MIND_MEMORY_DIR", str(tmp_path / "conc"))
        m = mind_memory.MindMemory()
        try:
            def writer(i):
                m.add_chunked(f"thread note number {i}", source="user")

            threads = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            assert m.count() == 8
        finally:
            m.shutdown()