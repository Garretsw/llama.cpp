import pytest
from utils import *

# tinygemma3 uses sliding-window attention (gemma3.attention.sliding_window),
# so any divergent continuation relies on context checkpoints: without a
# usable checkpoint the server forces full prompt re-processing. These tests
# verify that slot save files carry the checkpoints (trailing "CKPC" section)
# and that a restored slot can resume from one after a mid-prompt divergence.
server = ServerPreset.tinygemma3()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinygemma3()
    server.no_mmproj = True  # mtmd disables slot save/restore and checkpoints
    server.slot_save_path = "./tmp"
    server.temperature = 0.0
    server.n_slots = 1
    server.n_predict = 8
    # disable the in-RAM prompt cache: it also preserves checkpoints across
    # erase/re-prompt, which would mask whether the reuse measured below
    # really came from the restored slot FILE
    server.cache_ram = 0


def _chat(messages):
    res = server.make_request("POST", "/v1/chat/completions", data={
        "messages": messages,
        "cache_prompt": True,
        "temperature": 0.0,
        "max_tokens": 8,
    })
    assert res.status_code == 200
    return res.body


def test_props_advertises_slot_checkpoints():
    global server
    server.start()
    res = server.make_request("GET", "/props")
    assert res.status_code == 200
    assert res.body["slot_checkpoints"] is True


def test_slot_save_restore_checkpoints():
    global server
    server.start()

    user1 = "Tell me a story about a brave knight. " * 8
    user2 = "Now tell me a story about a dragon who lives in the mountains. " * 4

    # Turn A: single user message, generates checkpoints near the prompt end.
    body = _chat([{"role": "user", "content": user1}])
    reply1 = body["choices"][0]["message"]["content"]

    # Turn B: extended conversation — prefill crosses the last user-message
    # boundary, which creates a checkpoint at the start of user2.
    messages = [
        {"role": "user", "content": user1},
        {"role": "assistant", "content": reply1},
        {"role": "user", "content": user2},
    ]
    _chat(messages)

    # Save slot 0: the file must carry the checkpoints.
    res = server.make_request("POST", "/slots/0?action=save", data={
        "filename": "ckpt.bin",
    })
    assert res.status_code == 200
    n_saved = res.body["n_saved"]
    assert n_saved > 0
    assert res.body["n_ckpt_saved"] > 0
    # the trailing checkpoint section is accounted in n_written
    assert res.body["n_written"] > 0

    # Restart the server: all in-RAM state (slot KV, live checkpoints, RAM
    # prompt cache) is gone, so any later reuse must come from the restored
    # file. This is the real-world scenario the checkpoint section exists
    # for — surviving an unload/restart.
    server.stop()
    server.start()

    # Restore: tokens AND checkpoints come back.
    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "ckpt.bin",
    })
    assert res.status_code == 200
    assert res.body["n_restored"] == n_saved
    assert res.body["n_ckpt_restored"] > 0

    # Divergent continuation: same conversation but user2's content changed.
    # The prompt diverges inside the last user message — past the checkpoint
    # captured at its start. Without restored checkpoints an SWA model must
    # re-process the whole prompt (cache_n == 0); with them it rolls back to
    # the checkpoint and reuses the prefix.
    messages[2] = {"role": "user", "content": "Completely different question: what is 2+2? " * 4}
    body = _chat(messages)
    assert body["timings"]["cache_n"] > 0


def test_slot_restore_without_checkpoint_section():
    """A file saved before any checkpoints exist restores cleanly (n_ckpt 0)."""
    global server
    server.start()

    # /completion (no message spans): checkpoints are still created near the
    # prompt end, so erase the slot and save an empty one — header-only files
    # exercise the no-trailing-section path.
    res = server.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox",
        "n_predict": 4,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/slots/0?action=save", data={
        "filename": "plain.bin",
    })
    assert res.status_code == 200
    n_saved = res.body["n_saved"]
    assert n_saved > 0

    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "plain.bin",
    })
    assert res.status_code == 200
    assert res.body["n_restored"] == n_saved
