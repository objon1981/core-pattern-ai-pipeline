import os
import pytest
from src.feedback_db import FeedbackDB


@pytest.fixture
def temp_db():
    db_path = "tests/temp_feedback.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    db = FeedbackDB(db_path=db_path)
    yield db
    db.close()
    if os.path.exists(db_path):
        os.remove(db_path)


def test_insert_and_get_feedback(temp_db):
    temp_db.insert_feedback(user_id="user1", explanation_id="exp1", vote=1, refinement="Great!")
    temp_db.insert_feedback(user_id="user2", explanation_id="exp1", vote=0)
    feedbacks = temp_db.get_feedback(explanation_id="exp1")
    assert len(feedbacks) == 2
    assert feedbacks[0]["user_id"] == "user1"
    assert feedbacks[1]["vote"] == 0


def test_get_feedback_filter_by_user(temp_db):
    temp_db.insert_feedback(user_id="user1", explanation_id="exp2", vote=1)
    temp_db.insert_feedback(user_id="user2", explanation_id="exp2", vote=0)
    feedbacks = temp_db.get_feedback(user_id="user2")
    assert len(feedbacks) == 1
    assert feedbacks[0]["user_id"] == "user2"


def test_get_vote_stats(temp_db):
    temp_db.insert_feedback(user_id="u1", explanation_id="exp3", vote=1)
    temp_db.insert_feedback(user_id="u2", explanation_id="exp3", vote=1)
    temp_db.insert_feedback(user_id="u3", explanation_id="exp3", vote=0)
    stats = temp_db.get_vote_stats("exp3")
    assert stats["helpful"] == 2
    assert stats["not_helpful"] == 1
    assert stats["total"] == 3


def test_empty_stats(temp_db):
    stats = temp_db.get_vote_stats("nonexistent")
    assert stats == {"helpful": 0, "not_helpful": 0, "total": 0}
