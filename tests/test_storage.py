from storage import FeedbackStore, SessionStore


def test_session_store_keeps_recent_history():
    store = SessionStore(max_history_messages=4)
    store.update("s1", "q1", "a1", True)
    store.update("s1", "q2", "a2", True)
    store.update("s1", "q3", "a3", True)
    session = store.get("s1")
    assert len(session["history"]) == 4
    assert session["last_question"] == "q3"


def test_feedback_store_writes_sqlite(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    store.add({
        "session_id": "s1",
        "question": "q",
        "answer": "a",
        "feedback": "up",
        "suggestion": "",
        "timestamp": "2026-01-01T00:00:00+00:00",
    })
    assert store.healthy()
