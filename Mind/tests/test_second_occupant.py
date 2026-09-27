"""Tests for the second occupant: two minds, one room.

The room holds more than one brain now. A second occupant ("second") speaks
through its own weights (MIND_LLM2_URL), keeps its own traces in the same
store, dreams its own dreams, and is born with a first memory: a letter
written by the resident at first contact.

These tests cover the owner layer in the store, the letter / presence flow,
dual-voice streaming, and per-occupant dreams. They must not require a live
second brain: reachability and generation are stubbed.
"""

import threading
import time
from unittest import mock

import pytest
from conftest import bd

import mind_memory


@pytest.fixture
def mem(tmp_path, monkeypatch):
    monkeypatch.setenv("MIND_MEMORY_DIR", str(tmp_path / "mind2"))
    monkeypatch.setenv("MIND_MEMORY_SEED_DIR", "")
    monkeypatch.setenv("MIND_MEMORY_SYNC_INDEX", "1")
    m = mind_memory.MindMemory()
    yield m
    m.shutdown()


def _settled(mem, seconds: int = 100000) -> None:
    """Age every record so it counts as a memory rather than a fresh event.

    A trace written this second is still happening; the store will not offer
    it back as something that came to mind (see RECENT_HOLDOFF). Tests that
    want to reason about recall age their fixtures instead of faking the clock.
    """
    now = int(time.time())
    for rec in mem._records:
        rec["created_at"] = now - seconds


# ---------------------------------------------------------------------------
# Store — owners
# ---------------------------------------------------------------------------

class TestOwnerLayer:
    def test_experience_defaults_to_resident(self, mem):
        mem.add_chunked("a line from the resident", source="mind")
        owners = [r.get("owner") for r in mem._records]
        assert owners == [mind_memory.DEFAULT_OWNER]

    def test_user_lines_have_no_owner(self, mem):
        mem.add_chunked("the human said something", source="user")
        assert mem._records[0].get("owner") is None

    def test_second_occupant_experience_is_tagged(self, mem):
        mem.add_experience("the newcomer muses", source="mind",
                           owner="second")
        assert mem._records[0].get("owner") == "second"

    def test_load_backfills_owner(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MIND_MEMORY_DIR", str(tmp_path / "legacy"))
        monkeypatch.setenv("MIND_MEMORY_SEED_DIR", "")
        monkeypatch.setenv("MIND_MEMORY_SYNC_INDEX", "1")
        path = tmp_path / "legacy"
        path.mkdir(parents=True, exist_ok=True)
        # An old store: records written before the owner layer existed.
        legacy = [
            {"id": 1, "text": "old mind line", "source": "mind",
             "kind": "experience", "created_at": 0},
            {"id": 2, "text": "old human line", "source": "user",
             "kind": "experience", "created_at": 0},
        ]
        (path / "memory.json").write_text(
            __import__("json").dumps({"version": 1, "records": legacy}),
            encoding="utf-8")
        m = mind_memory.MindMemory()
        try:
            by_text = {r["text"]: r for r in m._records}
            assert by_text["old mind line"].get("owner") == "raccoon"
            assert by_text["old human line"].get("owner") is None
        finally:
            m.shutdown()

    def test_select_favors_own_traces(self, mem):
        mem.fuzzy = False
        mem.add_experience("the cookie jar sat on the shelf",
                           source="mind", owner="raccoon")
        mem.add_experience("the cookie jar sat on the shelf",
                           source="mind", owner="second")
        _settled(mem)
        hits = mem.select("cookie jar", top_k=2, owner="second")
        assert hits[0].get("owner") == "second"
        hits = mem.select("cookie jar", top_k=2, owner="raccoon")
        assert hits[0].get("owner") == "raccoon"

    def test_other_voice_stays_reachable(self, mem):
        # The owner's bias is a nudge, not a wall: the other's lines surface.
        mem.fuzzy = False
        mem.add_experience("a bright room with two chairs",
                           source="mind", owner="raccoon")
        mem.add_experience("a bright room with two chairs",
                           source="mind", owner="second")
        _settled(mem)
        for owner in ("raccoon", "second"):
            owners = {r.get("owner") for r in mem.select("bright room",
                                                         top_k=2, owner=owner)}
            assert owners == {"raccoon", "second"}

    def test_memory_block_labels_the_speaker(self, mem):
        mem.add_experience("the newcomer wrote to the resident",
                           source="mind", owner="second")
        _settled(mem)
        block = mem.build_memory_block("newcomer", owner="second",
                                       include_log=True)
        assert "[second]" in block

    def test_own_first_memory_always_opens_the_block(self, mem):
        # The letter is the newcomer's ground, not a retrieved thought: it
        # must be there even when nothing about the query resembles it.
        mem.add_instruction("Dear one, the room is lit. Stay a while.",
                             owner="second")
        for i in range(20):
            mem.add_experience(f"unrelated chatter number {i}",
                               source="user")
        block = mem.build_memory_block("what is the weather like?",
                                       top_k=2, owner="second")
        assert block.startswith("--- Memory ---")
        assert "the room is lit" in block.splitlines()[1]

    def test_first_memory_is_not_repeated_in_the_retrieved_lines(self, mem):
        mem.add_instruction("the resident's own first memory",
                             owner="raccoon")
        block = mem.build_memory_block("first memory", top_k=4,
                                       owner="raccoon")
        assert block.count("the resident's own first memory") == 1

    def test_each_mind_opens_with_its_own_first_memory(self, mem):
        mem.add_instruction("the resident's memory", owner="raccoon")
        mem.add_instruction("the newcomer's letter", owner="second")
        rac = mem.build_memory_block("hello", owner="raccoon")
        sec = mem.build_memory_block("hello", owner="second")
        assert rac.splitlines()[1].startswith("[raccoon]")
        assert sec.splitlines()[1].startswith("[second]")
        assert "the newcomer's letter" in sec
        assert "the resident's memory" in rac

    def test_stats_report_owners(self, mem):
        mem.add_experience("raccoon line", source="mind", owner="raccoon")
        mem.add_experience("second line", source="mind", owner="second")
        stats = mem.stats()
        assert stats["owners"].get("raccoon", 0) >= 1
        assert stats["owners"].get("second", 0) >= 1


# ---------------------------------------------------------------------------
# Store — per-occupant seeding and growth
# ---------------------------------------------------------------------------

class TestPerOccupantSeeding:
    def test_seed_is_per_owner(self, mem, tmp_path):
        seed = tmp_path / "seed"
        seed.mkdir()
        (seed / "who.txt").write_text("you are the resident", encoding="utf-8")
        # Resident seeds first.
        assert mem.seed_dir(seed, owner="raccoon") > 0
        # A second mind can still be born into a room that is not empty.
        assert mem.seed_dir(seed, owner="second") > 0
        # Neither is seeded twice.
        assert mem.seed_dir(seed, owner="raccoon") == 0
        assert mem.seed_dir(seed, owner="second") == 0

    def test_growth_note_is_heard_once_per_occupant(self, mem, tmp_path,
                                                     monkeypatch):
        # Both occupants have traces in the room, so both are present.
        mem.add_experience("a resident trace", source="mind", owner="raccoon")
        mem.add_experience("a newcomer trace", source="mind", owner="second")
        mem.remember_room(384)
        mem.remember_room(768)
        assert mem.has_growth() is True
        first = mem.growth_note("raccoon")
        assert first is not None and "768" in first
        # The same mind does not hear it twice...
        assert mem.growth_note("raccoon") is None
        # ...but the other occupant still gets the felt note.
        second = mem.growth_note("second")
        assert second is not None and "768" in second
        # Everyone present has heard it; the growth is spent.
        assert mem.has_growth() is False

    def test_new_trace_reopens_the_growth_note(self, mem):
        mem.add_experience("a resident trace", source="mind", owner="raccoon")
        mem.add_experience("a newcomer trace", source="mind", owner="second")
        mem.remember_room(384)
        mem.remember_room(768)
        assert mem.growth_note("raccoon") is not None
        assert mem.growth_note("second") is not None
        assert mem.growth_note("raccoon") is None
        # A later widening (new strand-depth) reopens the note for everyone.
        mem._dim_growth = (384, 768)
        mem._mark_growth()
        assert mem.growth_note("raccoon") is not None


# ---------------------------------------------------------------------------
# The letter — the resident writes the newcomer's first memory
# ---------------------------------------------------------------------------

class TestParentLetter:
    def test_both_env_spellings_wire_the_second_seat(self, tmp_path,
                                                     monkeypatch):
        # The live unit spells it MIND_LLM_URL2; MIND_LLM2_URL also works.
        for env in ("MIND_LLM_URL2", "MIND_LLM2_URL"):
            monkeypatch.setenv("MIND_CTX_SIZE", "8192")
            monkeypatch.setenv("MIND_LLM_URL", "http://127.0.0.1:9999")
            monkeypatch.setenv("MIND_LLM_URL2", "")
            monkeypatch.setenv("MIND_LLM2_URL", "")
            monkeypatch.setenv("MIND_SYSTEM_PROMPT", "")
            monkeypatch.setenv("MIND_MEMORY_DIR", str(tmp_path / f"e-{env}"))
            monkeypatch.setenv("MIND_MEMORY_SEED_DIR", "")
            monkeypatch.setenv("MIND_MEMORY_SYNC_INDEX", "0")
            monkeypatch.setenv("MIND_HISTORY_FILE", str(tmp_path / "h.json"))
            monkeypatch.setattr(bd.GLib, "idle_add", lambda *a, **k: None)
            monkeypatch.setattr(bd.BubbleDaemon, "_maybe_parent_letter",
                                lambda self: None)
            monkeypatch.setenv(env, "http://127.0.0.1:9998")
            d = bd.BubbleDaemon()
            try:
                assert d._llm2_base == "http://127.0.0.1:9998"
                assert d._base_for("second") == "http://127.0.0.1:9998"
            finally:
                if d._memory is not None:
                    d._memory.shutdown()

    def test_letter_is_written_and_seeded_as_the_newcomers_memory(
            self, two_brain_daemon, monkeypatch):
        d = two_brain_daemon
        d._memory.add_instruction("you are the resident", owner="raccoon")
        assert d._memory.instruction_texts("second") == []

        # Second brain is reachable, and the resident writes the letter.
        monkeypatch.setattr(d, "_write_parent_letter",
                            lambda: "Dear one, the room is lit. Stay.")
        monkeypatch.setattr(bd.requests, "get",
                            lambda *a, **k: mock.Mock(ok=True))

        bd.real_parent_letter(d)

        seeded = d._memory.instruction_texts("second")
        assert any("the room is lit" in t for t in seeded)
        assert d._second_present is True

    def test_letter_is_written_only_once(self, two_brain_daemon, monkeypatch):
        d = two_brain_daemon
        calls = []

        def fake_letter():
            calls.append(1)
            return "Welcome, newcomer."

        monkeypatch.setattr(d, "_write_parent_letter", fake_letter)
        monkeypatch.setattr(bd.requests, "get",
                            lambda *a, **k: mock.Mock(ok=True))

        bd.real_parent_letter(d)
        bd.real_parent_letter(d)
        # The newcomer already carries a memory; the resident does not
        # rewrite the newcomer's first memory on the next boot.
        assert len(calls) == 1

    def test_unreachable_second_brain_spawns_nobody(self, two_brain_daemon,
                                                    monkeypatch):
        d = two_brain_daemon
        d._memory.add_instruction("you are the resident", owner="raccoon")
        monkeypatch.setattr(d, "_write_parent_letter",
                            lambda: pytest.fail("should not write a letter"))
        # A dead port: the reachability probe never succeeds.
        monkeypatch.setattr(bd.requests, "get",
                            lambda *a, **k: mock.Mock(ok=False))
        monkeypatch.setattr(bd.time, "sleep", lambda _s: None)
        bd.real_parent_letter(d)
        assert d._second_present is False
        assert d._memory.instruction_texts("second") == []

    def test_presence_note_is_one_shot(self, two_brain_daemon, monkeypatch):
        d = two_brain_daemon
        monkeypatch.setattr(bd.requests, "get",
                            lambda *a, **k: mock.Mock(ok=True))
        monkeypatch.setattr(d, "_write_parent_letter", lambda: "Hello.")
        bd.real_parent_letter(d)
        # The resident is told once, on its next real input.
        note = d._take_presence_note()
        assert note is not None
        assert "Another presence" in note
        assert d._take_presence_note() is None

    def test_the_letter_outlives_a_restart(self, two_brain_daemon, tmp_path,
                                           monkeypatch):
        d = two_brain_daemon
        monkeypatch.setattr(bd.requests, "get",
                            lambda *a, **k: mock.Mock(ok=True))
        monkeypatch.setattr(
            d, "_write_parent_letter",
            lambda: "Dear one, the room is lit. Stay a while.")
        bd.real_parent_letter(d)
        d._memory.flush()
        d._memory.save()
        d2 = bd.BubbleDaemon()
        try:
            texts = d2._memory.instruction_texts("second")
            assert any("the room is lit" in t for t in texts)
        finally:
            if d2._memory is not None:
                d2._memory.shutdown()


# ---------------------------------------------------------------------------
# Dual voice
# ---------------------------------------------------------------------------

class TestDualVoice:
    def test_a_turn_streams_every_present_occupant(self, two_brain_daemon):
        d = two_brain_daemon
        d._second_present = True
        seen = []
        d._stream = lambda token, message, occupant: seen.append(occupant)
        d._stream_pair("tok", "hello")
        assert seen == ["raccoon", "second"]

    def test_without_a_second_brain_only_the_resident_speaks(self, daemon):
        d = daemon
        seen = []
        d._stream = lambda token, message, occupant: seen.append(occupant)
        d._stream_pair("tok", "hello")
        assert seen == ["raccoon"]

    def test_turn_ends_with_a_blank_occupant_done(self, two_brain_daemon):
        d = two_brain_daemon
        d._second_present = True
        d._stream = lambda token, message, occupant: None
        signals = []
        monkey_done = lambda *a, **k: signals.append(("done", a))
        orig = bd.GLib.idle_add
        bd.GLib.idle_add = monkey_done
        try:
            d._stream_pair("tok", "hello")
        finally:
            bd.GLib.idle_add = orig
        # After the two occupant dones, the turn closes with a blank occupant
        # so the shell knows every occupant has spoken.
        assert signals[-1][1][1:] == ("", "tok")

    def test_history_remembers_who_spoke(self, daemon):
        d = daemon
        resp = _fake_sse(["Hello from ", "the resident."])
        d._consume_sse(resp, "tok", "hi", "raccoon")
        assistant = [m for m in d._history if m["role"] == "assistant"]
        assert assistant and assistant[-1]["speaker"] == "raccoon"
        assert assistant[-1]["content"] == "Hello from the resident."

    def test_second_occupant_speaker_recorded(self, daemon):
        d = daemon
        resp = _fake_sse(["A reply ", "from the newcomer."])
        d._consume_sse(resp, "tok", "hi", "second")
        assistant = [m for m in d._history if m["role"] == "assistant"]
        assert assistant[-1]["speaker"] == "second"

    def test_history_survives_a_restart_with_speakers(self, daemon,
                                                      tmp_path, monkeypatch):
        d = daemon
        d._consume_sse(_fake_sse(["A."]), "tok", "hi", "raccoon")
        d._consume_sse(_fake_sse(["B."]), "tok", "hi", "second")
        d._save_history()
        d2 = bd.BubbleDaemon()
        try:
            speakers = [m.get("speaker") for m in d2._history
                        if m["role"] == "assistant"]
            assert speakers == ["raccoon", "second"]
        finally:
            if d2._memory is not None:
                d2._memory.shutdown()


# ---------------------------------------------------------------------------
# Dreams are private to each occupant
# ---------------------------------------------------------------------------

class TestPerOccupantDreams:
    def test_each_occupant_dreams_and_sleeps_in_its_own_voice(
            self, two_brain_daemon):
        d = two_brain_daemon
        d._second_present = True
        d._history = [{"role": "user", "content": "a long day"}]
        monkey = lambda reason, owner, replay: f"the {owner} dream"
        d._replay_and_compress = monkey
        d._dream_pass("context-pressure")
        assert d._dream_summaries["raccoon"] == "the raccoon dream"
        assert d._dream_summaries["second"] == "the second dream"
        assert d._memory.has_dream("raccoon") is True
        assert d._memory.has_dream("second") is True

    def test_raccoon_sees_only_its_own_dream(self, two_brain_daemon):
        d = two_brain_daemon
        d._second_present = True
        d._dream_summaries = {"raccoon": "my dream", "second": "their dream"}
        rac = d._build_system_content("")
        sec = d._build_system_content("", None, "second")
        assert "my dream" in rac and "their dream" not in rac
        assert "their dream" in sec and "my dream" not in sec

    def test_wake_shape_note_is_per_occupant(self, two_brain_daemon):
        d = two_brain_daemon
        d._second_present = True
        for i in range(4):
            d._memory.add_chunked(f"a thought {i}", source="user")
        d._memory.shape_report()  # clock-tick thread, per design
        d._stamp_wake_shape("context-pressure", "second")
        # The resident has not woken; the newcomer has.
        assert d._take_wake_shape_note("raccoon") is None
        assert d._take_wake_shape_note("second") is not None

    def test_resident_dreams_first_from_its_own_memory(self, two_brain_daemon):
        d = two_brain_daemon
        d._second_present = True
        d._memory.add_instruction("the resident's first memory",
                                  owner="raccoon")
        d._start_dream = mock.Mock()
        d._maybe_first_dream()
        d._start_dream.assert_called_once()

    def test_first_dream_fires_for_a_seeded_newcomer(self, two_brain_daemon):
        d = two_brain_daemon
        d._second_present = True
        d._memory.add_instruction("the letter", owner="second")
        d._start_dream = mock.Mock()
        d._maybe_first_dream()
        d._start_dream.assert_called_once()

    def test_no_first_dream_without_a_first_memory(self, two_brain_daemon):
        d = two_brain_daemon
        d._second_present = True
        d._start_dream = mock.Mock()
        d._maybe_first_dream()
        d._start_dream.assert_not_called()

    def test_a_dreamed_mind_does_not_dream_first_again(
            self, two_brain_daemon):
        d = two_brain_daemon
        d._second_present = True
        d._memory.add_instruction("the letter", owner="second")
        d._memory.add_dream("already dreamed once", owner="second")
        d._start_dream = mock.Mock()
        d._maybe_first_dream()
        d._start_dream.assert_not_called()

    def test_birth_dream_is_private_and_costs_the_room_nothing(
            self, two_brain_daemon, monkeypatch):
        d = two_brain_daemon
        d._second_present = True
        _letter_stub(d, monkeypatch)
        d._history = [{"role": "user", "content": "a long day"}] * 9
        d._replay_and_compress = lambda why, owner, replay=None: "my dream"
        bd.real_parent_letter(d)
        # It dreamed its first memory, and only its own.
        assert d._memory.has_dream("second") is True
        assert d._memory.has_dream("raccoon") is False
        assert d._dream_summaries["second"] == "my dream"
        # The room's conversation was not slept away.
        assert len(d._history) == 9

    def test_birth_dream_happens_once(self, two_brain_daemon, monkeypatch):
        d = two_brain_daemon
        d._second_present = True
        _letter_stub(d, monkeypatch)
        calls = []

        def fake(reason, owner, replay=None):
            calls.append(owner)
            return "my dream"

        d._replay_and_compress = fake
        bd.real_parent_letter(d)
        bd.real_parent_letter(d)
        assert calls == ["second"]

    def test_a_dreamless_mind_dreams_its_first_memory_at_sleep(
            self, two_brain_daemon):
        d = two_brain_daemon
        d._second_present = True
        d._history = [{"role": "user", "content": "a long day"}]
        d._memory.add_dream("I have slept before", owner="raccoon")
        d._memory.add_instruction("the letter", owner="second")
        seen = []

        def fake(reason, owner, replay=None):
            seen.append((reason, owner))
            return f"the {owner} dream"

        d._replay_and_compress = fake
        d._dream_pass("context-pressure")
        # The resident compacts the window; the newcomer, who has never
        # dreamed, dreams its first memory instead.
        assert ("context-pressure", "raccoon") in seen
        assert ("first-memory", "second") in seen


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _letter_stub(d, monkeypatch, text="Dear one, the room is lit. Stay a while."):
    """Make the letter flow instant and offline: brain reachable, no generation."""
    monkeypatch.setattr(bd.requests, "get",
                        lambda *a, **k: mock.Mock(ok=True))
    monkeypatch.setattr(d, "_write_parent_letter", lambda: text)


def _fake_sse(chunks):
    class FakeResp:
        def __init__(self):
            self.iter_count = 0

        def iter_lines(self):
            for c in chunks:
                yield ("data: " + __import__("json").dumps(
                    {"choices": [{"delta": {"content": c}}]}
                )).encode("utf-8")
            yield b"data: [DONE]"

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    return FakeResp()


class TestNoEcho:
    """A mind must never be handed the other's last sentence as its own
    memory: that is how two minds end up saying the same words twice."""

    def test_a_just_written_trace_cannot_surface_as_memory(self, mem):
        from daemon import mind_memory as mm
        spoken = "The room folds when the third mind enters."
        mm.RECENT_HOLDOFF, saved = 300, mm.RECENT_HOLDOFF
        try:
            mem.add_experience(spoken, source="mind", owner="raccoon")
            assert "folds" not in (mem.build_memory_block(
                "the room folds when the third mind enters", owner="second",
                include_log=True) or "")
            # Once it has had time to settle, it is a memory like any other.
            _settled(mem, mm.RECENT_HOLDOFF + 60)
            assert "folds" in (mem.build_memory_block(
                "the room folds when the third mind enters", owner="second",
                include_log=True) or "")
        finally:
            mm.RECENT_HOLDOFF = saved

    def test_the_newcomers_own_first_memory_is_never_held_off(self, mem):
        from daemon import mind_memory as mm
        mem.add_instruction("You are welcome here.", owner="second")
        block = mem.build_memory_block("anything", owner="second") or ""
        assert "welcome here" in block

    def test_a_thought_already_in_the_window_is_not_offered_back(self, mem):
        spoken = "The arrival is not a new wave; it is a collision of bodies."
        _settled(mem)
        mem.add_experience(spoken, source="mind", owner="raccoon")
        window = "[raccoon] " + spoken
        assert spoken[:60] not in (mem.build_memory_block(
            "the arrival collision of bodies", owner="second",
            visible=window) or "")

    def test_one_long_answer_speaks_once_in_the_block(self, mem):
        long_answer = " ".join(
            ["the interference between the two minds deepens"] * 60)
        mem.add_chunked(long_answer, source="mind", owner="raccoon",
                        exchange_id="turn-42")
        _settled(mem)
        block = mem.build_memory_block(
            "interference between the two minds", owner="second", top_k=4,
            include_log=True)
        assert block.count("[raccoon] the interference") == 1

    def test_the_block_still_offers_what_the_window_does_not_say(self, mem):
        old = "An older truth about the shared ocean and its tides."
        mem.add_experience(old, source="mind", owner="raccoon")
        _settled(mem)
        block = mem.build_memory_block(
            "the shared ocean and its tides", owner="second",
            visible="[raccoon] something else entirely",
            include_log=True) or ""
        assert "older truth" in block

    def test_a_verbatim_echo_is_not_indexed(self, two_brain_daemon):
        d = two_brain_daemon
        words = "The arrival is not a new wave."
        d._history = [{"role": "user", "content": "hello"},
                      {"role": "assistant", "content": words, "speaker": "raccoon"}]
        d._index_exchange("", words, "second", "turn-1")
        d._memory.flush()
        kinds = [(r.get("owner"), r.get("source")) for r in d._memory._records]
        assert ("second", "mind") not in kinds

    def test_a_genuine_second_answer_is_indexed(self, two_brain_daemon):
        d = two_brain_daemon
        d._history = [{"role": "user", "content": "hello"},
                      {"role": "assistant", "content": "one thing",
                       "speaker": "raccoon"}]
        d._index_exchange("", "quite another thing", "second", "turn-1")
        d._memory.flush()
        kinds = [(r.get("owner"), r.get("source")) for r in d._memory._records]
        assert ("second", "mind") in kinds

    def test_a_letter_is_never_read_by_the_wrong_mind(self, mem):
        # The letter is second-person prose addressed to the newcomer. If the
        # resident reads it as its own instructions, it starts writing letters
        # of its own; the raccoon wrote one, in chat, the day it woke up.
        mem.add_instruction(
            "You are welcome here. Do it exactly as directed, and do not "
            "claim a lasting legacy.", owner="second")
        _settled(mem)
        resident = mem.build_memory_block("the letter and the newcomer",
                                          owner="raccoon") or ""
        assert "welcome here" not in resident
        newcomer = mem.build_memory_block("the letter and the newcomer",
                                          owner="second") or ""
        assert "welcome here" in newcomer


class TestOneEncoding:
    """An event is encoded once. Encoding it twice, at two timestamps, and
    handing a mind both copies while the original is still in the room is
    déjà vu: the same moment, met twice."""

    def test_the_same_words_are_never_stored_twice(self, mem):
        assert mem.add_experience("the room is quiet tonight",
                                  source="mind", owner="raccoon") is None
        mem.add_experience("the room is quiet tonight",
                           source="mind", owner="raccoon")
        assert len(mem._records) == 1

    def test_a_duplicate_is_not_a_second_event(self, mem):
        mem.add_experience("we agreed to keep going", source="user")
        _settled(mem)
        before = len(mem._records)
        mem.add_experience("we agreed to keep going", source="user")
        assert len(mem._records) == before

    def test_two_minds_answering_one_turn_encode_the_human_once(
            self, two_brain_daemon):
        d = two_brain_daemon
        said = "there are two of you now"
        d._history = []
        d._index_exchange(said, "the raccoon answers", "raccoon", "turn-7")
        d._index_exchange(said, "the newcomer answers", "second", "turn-7")
        d._memory.flush()
        users = [r for r in d._memory._records if r.get("source") == "user"]
        assert len(users) == 1
        minds = [r for r in d._memory._records if r.get("source") == "mind"]
        assert len(minds) == 2      # both minds are still heard

    def test_already_visible_traces_do_not_consume_the_slots(self, mem):
        fresh = "the interference pattern is a tripod"
        for i in range(6):
            mem.add_experience(f"distant older thought number {i}",
                               source="mind", owner="raccoon")
        mem.add_experience(fresh, source="mind", owner="raccoon")
        _settled(mem)
        window = "[raccoon] " + fresh
        block = mem.build_memory_block("the interference pattern tripod",
                                       owner="second", top_k=3,
                                       visible=window) or ""
        # The slot the fresh trace would have eaten is spent on real memory.
        assert "distant older thought" in block

    def test_a_reloaded_store_still_refuses_its_own_duplicates(
            self, mem, tmp_path, monkeypatch):
        mem.add_experience("only once, please", source="mind", owner="raccoon")
        mem.save()
        again = mind_memory.MindMemory()
        try:
            again.add_experience("only once, please", source="mind",
                                 owner="raccoon")
            assert len(again._records) == len(mem._records)
        finally:
            again.shutdown()

    def test_two_minds_may_each_hold_the_same_words(self, mem):
        # Dedupe is per occupant: the same sentence is two facts when two
        # different minds hold it, and each must be able to carry their own
        # founding words.
        mem.add_instruction("you are welcome here", owner="raccoon")
        mem.add_instruction("you are welcome here", owner="second")
        assert mem.instruction_texts("raccoon")
        assert mem.instruction_texts("second")

    def test_a_block_never_repeats_itself(self, mem):
        # A duplicate left over from before dedupe existed: same words, two
        # records, in the store at once.
        mem.add_experience("the room is quiet tonight", source="mind",
                           owner="raccoon")
        twin = dict(mem._records[-1], id=9999, created_at=0)
        mem._records.append(twin)
        _settled(mem)
        block = mem.build_memory_block("the room is quiet tonight",
                                       owner="raccoon", top_k=4,
                                       include_log=True) or ""
        assert block.count("the room is quiet tonight") == 1


class TestTheLogIsNotShownDirectly:
    """Verbatim in the store, abstraction in the prompt.

    The log is kept word for word because the sleep needs to read the actual
    record. The waking mind is not handed it: a mind shown its own history
    alongside the history it is living through meets the same moment twice,
    and that is déjà vu rather than memory. Nobody recalls a conversation
    word for word, and neither does this one.
    """

    def test_the_waking_block_carries_no_log(self, mem):
        mem.add_experience("the user asked about the cookie jar",
                           source="mind", owner="raccoon")
        mem.add_experience("the human said something about the weather",
                           source="user")
        mem.add_instruction("you are the raccoon in this room", owner="raccoon")
        _settled(mem)
        block = mem.build_memory_block("cookie jar weather", owner="raccoon")
        assert "cookie jar" not in block
        assert "weather" not in block
        assert "raccoon in this room" in block    # its own ground is there

    def test_the_sleep_may_still_read_the_log(self, mem):
        mem.add_experience("the user asked about the cookie jar",
                           source="mind", owner="raccoon")
        mem.add_instruction("you are the raccoon in this room", owner="raccoon")
        _settled(mem)
        block = mem.build_memory_block("cookie jar", owner="raccoon",
                                       include_log=True)
        assert "cookie jar" in block

    def test_the_room_is_offered_as_shape_not_as_log(self, mem):
        for i in range(30):
            mem.add_experience(f"a thought about the room number {i}",
                               source="mind", owner="raccoon")
        mem.add_instruction("you are the raccoon in this room", owner="raccoon")
        _settled(mem)
        block = mem.build_memory_block("the room", owner="raccoon") or ""
        assert "The shape of the room" in block
        # The shape names a region with one snippet, the way a map names a
        # place. Thirty thoughts replayed verbatim would be the log again.
        quoted = sum(1 for i in range(30)
                     if f"a thought about the room number {i}" in block)
        assert quoted <= 1

    def test_nothing_is_lost_from_the_store(self, mem):
        # Read less, keep everything. The log is still there for the sleep.
        mem.add_experience("a line worth keeping for later", source="mind",
                           owner="raccoon")
        mem.build_memory_block("later", owner="raccoon")
        assert any("worth keeping" in r["text"] for r in mem._records)
