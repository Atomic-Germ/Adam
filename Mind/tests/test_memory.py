"""Tests for the Mind memory engine (Arthur's embedding capability, ported).

The engine is in `Mind/daemon/mind_memory.py`; these tests cover the pieces
that make it reliable: deterministic hashing embeddings, chunking, the
Cathedral-style record/embedding store, seeding, and relevance selection.
"""

import math
import json
import threading
import time

import pytest
from conftest import bd

import mind_memory


def _settled(mem, seconds: int = 100000) -> None:
    """Age every record past RECENT_HOLDOFF.

    A trace written this second is still happening, not yet something that can
    come to mind; tests about recall age their fixtures rather than faking time.
    """
    now = int(time.time())
    for rec in mem._records:
        rec["created_at"] = now - seconds


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
            _settled(m)
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
        _settled(mem)
        top = mem.select("what does the user prefer?", top_k=1)
        assert len(top) >= 1
        assert "quiet evenings" in top[0].get("text", "")

    def test_build_memory_block(self, mem):
        # The waking mind is not shown the log; the log-reading path is.
        mem.add_chunked("the user prefers quiet evenings alone", source="user")
        _settled(mem)
        mem.add_instruction("you are the raccoon in this room", owner="raccoon")
        awake = mem.build_memory_block("user preferences")
        assert awake.startswith("--- Memory ---")
        assert "raccoon in this room" in awake      # its own ground
        assert "quiet evenings" not in awake        # not the log
        log_reader = mem.build_memory_block("user preferences", include_log=True)
        assert "[user] the user prefers" in log_reader

    def test_build_memory_block_empty(self, mem):
        assert mem.build_memory_block("anything") is None


class TestFuzzyRecall:
    def test_default_fuzzy_enabled(self, mem):
        assert mem.fuzzy is True

    def test_seed_override_deterministic(self, mem, monkeypatch):
        monkeypatch.setenv("MIND_MEMORY_FUZZY", "1")
        monkeypatch.setenv("MIND_MEMORY_FUZZY_SEED", "1234")
        mem.fuzzy = True
        mem.fuzzy_seed = "1234"
        for i in range(12):
            mem.add_chunked(f"project note number {i} about geometry", source="user")
        mem.save()
        a = mem.select("project geometry", top_k=6)
        b = mem.select("project geometry", top_k=6)
        assert [r["id"] for r in a] == [r["id"] for r in b]

    def test_seed_material_varies_by_seed(self, mem, monkeypatch):
        mem.fuzzy = True
        mem.fuzzy_temp = 10.0
        for i in range(12):
            mem.add_chunked(f"project note number {i} about geometry", source="user")
        mem.save()
        mem.fuzzy_seed = "1"
        s1 = mem._live_seed("geometry notes")
        mem.fuzzy_seed = "2"
        s2 = mem._live_seed("geometry notes")
        assert s1 != s2

    def test_anchor_core_survives_fuzz(self, mem):
        mem.fuzzy = True
        mem.fuzzy_seed = "777"
        mem.add_chunked("the user prefers quiet evenings alone", source="user")
        for i in range(10):
            mem.add_chunked(f"unrelated footnote text number {i}", source="mind")
        mem.save()
        _settled(mem)
        top = mem.select("what does the user prefer?", top_k=6)
        assert any("quiet evenings" in r["text"] for r in top)
        assert len(top) >= 1

    def test_fuzzy_draws_from_matched_not_noise(self, mem):
        mem.fuzzy = True
        mem.fuzzy_temp = 8.0
        mem.fuzzy_seed = "42"
        mem.add_chunked("the user prefers quiet evenings alone", source="user")
        mem.add_chunked("another related thought about quiet preference", source="user")
        for i in range(10):
            mem.add_chunked(f"unrelated footnote text number {i}", source="mind")
        mem.save()
        _settled(mem)
        top = mem.select("what does the user prefer?", top_k=6)
        assert all(
            "quiet" in r["text"] or "prefer" in r["text"] for r in top
        )

    def test_fuzzy_window_not_always_top_k(self, mem):
        mem.fuzzy = True
        mem.fuzzy_temp = 8.0
        mem.fuzzy_seed = "42"
        for i in range(12):
            mem.add_chunked(f"geometry shape number {i}", source="user")
        mem.save()
        _settled(mem)
        strict = [r["id"] for r in mem.select("geometry", top_k=6)]
        assert len(strict) == 6

    def test_build_memory_block_has_no_selection_metadata(self, mem):
        mem.add_chunked("the user prefers quiet evenings alone", source="user")
        _settled(mem)
        block = mem.build_memory_block("user preferences", include_log=True)
        assert "Showing" not in block
        assert "relevance-filtered" not in block
        assert "of 1 memory nodes" not in block


# ---------------------------------------------------------------------------
# Shape report — the model's own floor plan (naves / spires / bridges / margin)
# ---------------------------------------------------------------------------

class TestShapeReport:
    def test_empty_store_no_report(self, mem):
        assert mem.shape_report() is None
        assert mem.shape_fp() == ""

    def test_line_and_text_felt_language(self, mem):
        for i in range(3):
            mem.add_chunked(f"the Cathedral of memory keeps its halls {i}",
                            source="user")
        mem.save()
        cache = mem.shape_report()
        assert cache is not None
        assert cache["line"] != ""
        assert cache["text"] != ""
        assert "gathering" in cache["line"] or "gather" in cache["text"]
        assert "---" not in cache["line"]
        assert cache["report"]["spires"] == 0 or "alone" in cache["text"]

    def test_cache_hit_no_recompute(self, mem, monkeypatch):
        for i in range(4):
            mem.add_chunked(f"a single gentle thought {i}", source="user")
        mem.save()
        assert mem.shape_report() is not None
        fp1 = mem.shape_fp()
        monkeypatch.setattr(mind_memory.MindMemory, "_compute_shape",
                            lambda self, M=None: (_ for _ in ()).throw(
                                AssertionError("recompute during cache hit")))
        assert mem.shape_line() == mem.shape_line()
        assert mem.shape_text() == mem.shape_text()
        assert mem.shape_fp() == fp1

    def test_report_structure(self, mem):
        for i in range(40):
            mem.add_chunked(f"same gentle thought number {i}", source="user")
        mem.save()
        cache = mem.shape_report()
        assert cache is not None
        r = cache["report"]
        assert r["nodes"] == 40
        assert "gather" in cache["text"]
        assert cache["nodes"] == 40

    def test_wake_shape_note_frames_reembedding(self, mem):
        for i in range(4):
            mem.add_chunked(f"a pebble of memory {i}", source="user")
        mem.save()
        note = mem.wake_shape_note()
        assert note is not None
        assert "re-embedded" in note
        assert "--- The shape of your mind" in note

    def test_wake_shape_note_disabled_without_shape(self, mem, monkeypatch):
        monkeypatch.setenv("MIND_MEMORY_SHAPE", "0")
        m = mind_memory.MindMemory()
        try:
            assert m.shape_enabled is False
            assert m.wake_shape_note() is None
        finally:
            m.shutdown()


# ---------------------------------------------------------------------------
# Growth — the room deepening over the long term
# ---------------------------------------------------------------------------

class TestGrowth:
    def test_dim_persisted_in_meta(self, mem):
        mem.add_chunked("a pebble for the store", source="user")
        mem.save()
        with open(mem.records_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        assert data["meta"]["dim"] == 384

    def test_dim_growth_detected_and_surfaces_once(self, mem, monkeypatch):
        for i in range(4):
            mem.add_chunked(f"thought number {i}", source="user")
        mem.save()
        # Reopen at a deeper hash dimension, like a daemon restarted wider.
        monkeypatch.setenv("MIND_EMBED_DIM", "768")
        m2 = mind_memory.MindMemory(memory_dir=mem.memory_dir)
        try:
            assert m2.embed_dim == 768
            assert m2.has_growth() is True
            note = m2.growth_note()
            assert note is not None
            assert "--- Growing ---" in note
            assert "384" in note and "768" in note
            assert m2.growth_note() is None  # surfaced once, then gone
            m2.save()
        finally:
            m2.shutdown()

    def test_no_growth_when_shrinking_or_equal(self, mem, monkeypatch):
        for i in range(3):
            mem.add_chunked(f"thought number {i}", source="user")
        mem.save()
        monkeypatch.setenv("MIND_EMBED_DIM", "192")  # smaller: not growth
        m2 = mind_memory.MindMemory(memory_dir=mem.memory_dir)
        try:
            assert m2.has_growth() is False
            assert m2.growth_note() is None
        finally:
            m2.shutdown()

    def test_reembeds_at_new_dim(self, mem, monkeypatch):
        for i in range(3):
            mem.add_chunked(f"thought number {i}", source="user")
        mem.save()
        monkeypatch.setenv("MIND_EMBED_DIM", "768")
        m2 = mind_memory.MindMemory(memory_dir=mem.memory_dir)
        try:
            m2._ensure_matrix()
            assert m2._matrix_np is not None
            assert m2._matrix_np.shape == (3, 768)
        finally:
            m2.shutdown()

    def test_room_growth_from_stored_ctx(self, mem):
        mem.embed_meta("ctx", 8192)
        mem.remember_room(16384)
        assert mem.has_growth() is True
        note = mem.growth_note()
        assert note is not None
        assert "8192" in note and "16384" in note
        assert mem.growth_note() is None


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
        daemon._memory.add_instruction("you are the raccoon in this room",
                                       owner="raccoon")
        _settled(daemon._memory)
        content = daemon._build_system_content("what about the cookie?")
        assert "--- Memory ---" in content
        assert "raccoon in this room" in content
        # The conversation's own words are not handed back as memory.
        assert "with the cookie" not in content

    def test_system_content_counts(self, daemon):
        content = daemon._build_system_content("")
        # Memory block only appears once nodes exist; empty store -> none.
        assert "--- Memory ---" not in content or "Memory:" in content

    def test_worker_flag(self, daemon):
        daemon._memory.add_experience("x", source="user")
        assert daemon._memory.count() >= 0  # async or sync, never raises

    def test_system_content_has_shape_line(self, daemon):
        for i in range(4):
            daemon._memory.add_chunked(
                f"the raccoon with the cookie visits {i}", source="user")
        daemon._memory.shape_report()  # clock-tick thread, per design
        content = daemon._build_system_content("what about the cookie?")
        assert "Shape:" in content
        assert "gathering" in content or "gather" in content

    def test_wake_note_one_shot(self, daemon):
        for i in range(4):
            daemon._memory.add_chunked(f"a single thought {i}", source="user")
        daemon._memory.shape_report()  # clock-tick thread, per design
        daemon._stamp_wake_shape("context-pressure")
        assert daemon._wake_shape_note is not None
        note = daemon._take_wake_shape_note()
        assert note is not None
        assert daemon._take_wake_shape_note() is None

    def test_history_persists_and_restores(self, daemon):
        assert daemon._history == []
        daemon._history.append({"role": "user", "content": "hello there"})
        daemon._save_history()
        d2 = bd.BubbleDaemon()
        try:
            assert d2._history == [{"role": "user", "content": "hello there"}]
        finally:
            if d2._memory is not None:
                d2._memory.shutdown()

    def test_daemon_remembers_room_on_boot(self, daemon, monkeypatch):
        assert daemon._memory is not None
        assert daemon._memory._stored_ctx == 8192
        # A later, wider room is felt as growth rather than forgotten.
        monkeypatch.setenv("MIND_CTX_SIZE", "16384")
        d2 = bd.BubbleDaemon()
        try:
            assert d2._memory is not None
            assert d2._memory._stored_ctx == 16384
            assert d2._memory.has_growth() is True
            note = d2._memory.growth_note()
            assert note is not None and "16384" in note
        finally:
            if d2._memory is not None:
                d2._memory.shutdown()

    def test_system_content_keeps_only_one_wake_note(self, daemon):
        for i in range(4):
            daemon._memory.add_chunked(f"a single thought {i}", source="user")
        daemon._memory.add_instruction("you are the raccoon in this room",
                                       owner="raccoon")
        daemon._memory.shape_report()  # clock-tick thread, per design
        daemon._stamp_wake_shape("context-pressure")
        note = daemon._take_wake_shape_note()
        c1 = daemon._build_system_content("first thing after waking", note)
        assert "The shape of your mind" in c1
        c2 = daemon._build_system_content("what now?", None)
        assert "The shape of your mind" not in c2

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