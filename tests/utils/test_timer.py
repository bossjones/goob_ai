from __future__ import annotations

import time

from goob_ai.utils.timer import Timer, TimerCollection

import pytest


def test_timer_start_stop(mocker):
    timer = Timer()
    mock_time = mocker.patch("time.time", side_effect=[1, 2])
    timer.start()
    timer.stop()
    assert timer.duration() == 1.0


def test_timer_reset(mocker):
    timer = Timer()
    mock_time = mocker.patch("time.time", side_effect=[1, 2, 3])
    timer.start()
    timer.stop()
    timer.reset()
    assert timer.duration() == 0.0


def test_timer_duration_running(mocker):
    timer = Timer()
    mock_time = mocker.patch("time.time", side_effect=[1, 2, 3])
    timer.start()
    assert timer.duration() == 1.0
    timer.stop()
    assert timer.duration() == 2.0


def test_timer_collection_reset(mocker):
    tc = TimerCollection()
    mock_time = mocker.patch("time.time", side_effect=[1, 2, 3, 4])
    tc.start("Timer 1")
    tc.stop("Timer 1")
    tc.reset("Timer 1")
    assert tc.duration("Timer 1") == 0.0


def test_timer_collection_reset_all(mocker):
    tc = TimerCollection()
    mock_time = mocker.patch("time.time", side_effect=[1, 2, 3, 4, 5, 6])
    tc.start("Timer 1")
    tc.start("Timer 2")
    tc.stop("Timer 1")
    tc.stop("Timer 2")
    tc.reset_all()
    assert tc.duration("Timer 1") == 0.0
    assert tc.duration("Timer 2") == 0.0


def test_timer_collection_names():
    tc = TimerCollection()
    tc.start("Timer 1")
    tc.start("Timer 2")
    assert set(tc.names()) == {"Timer 1", "Timer 2"}


def test_timer_collection_nonexistent_timer():
    tc = TimerCollection()
    with pytest.raises(KeyError):
        tc.stop("Nonexistent Timer")
