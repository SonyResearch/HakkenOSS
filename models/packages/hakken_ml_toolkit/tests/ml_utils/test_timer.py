import time

import pytest

from hakken_ml_toolkit.ml_utils import TimersCollection


@pytest.fixture
def timer() -> TimersCollection:
    return TimersCollection()


def test_tic_toc(timer: TimersCollection) -> None:
    timer.tic("test")
    time.sleep(0.1)
    elapsed = timer.toc("test")
    assert 0.1 <= elapsed < 0.2
    assert "test" not in timer.timer_dict


def test_multiple_timers(timer: TimersCollection) -> None:
    timer.tic("timer1")
    time.sleep(0.1)
    timer.tic("timer2")
    time.sleep(0.1)
    elapsed2 = timer.toc("timer2")
    time.sleep(0.1)
    elapsed1 = timer.toc("timer1")
    assert 0.1 <= elapsed2 < 0.2
    assert 0.3 <= elapsed1 < 0.4


def test_toc_without_tic(timer: TimersCollection) -> None:
    with pytest.raises(AssertionError):
        timer.toc("nonexistent")


def test_pause_resume(timer: TimersCollection) -> None:
    timer.tic("test")
    time.sleep(0.1)
    timer.pause("test")
    time.sleep(0.1)
    timer.resume("test")
    time.sleep(0.1)
    elapsed = timer.toc("test")
    assert 0.2 <= elapsed < 0.3


def test_pause_resume_multiple(timer: TimersCollection) -> None:
    timer.tic("test")
    time.sleep(0.1)
    timer.pause("test")
    time.sleep(0.1)
    timer.resume("test")
    time.sleep(0.1)
    timer.pause("test")
    time.sleep(0.1)
    timer.resume("test")
    elapsed = timer.toc("test")
    assert 0.2 <= elapsed < 0.3


def test_resume_without_pause(timer: TimersCollection) -> None:
    timer.tic("test")
    with pytest.raises(KeyError):
        timer.resume("test")
    elapsed = timer.toc("test")
    assert elapsed >= 0


def test_resume_without_tic(timer: TimersCollection) -> None:
    timer.pause("test")
    timer.resume("test")
    assert "test" not in timer.timer_dict
    assert "test" not in timer.pause_dict
