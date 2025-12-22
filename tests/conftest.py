import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def stub_frames():
    return ["frame0", "frame1", "frame2"]


@pytest.fixture
def stub_player_tracks():
    return [
        {1: {"bbox": [0, 0, 10, 10]}, 2: {"bbox": [20, 0, 30, 10]}},
        {1: {"bbox": [1, 0, 11, 10]}, 2: {"bbox": [21, 0, 31, 10]}},
        {1: {"bbox": [2, 0, 12, 10]}, 2: {"bbox": [22, 0, 32, 10]}},
    ]


@pytest.fixture
def stub_ball_tracks():
    return [
        {1: {"bbox": [4, 4, 6, 6]}},
        {1: {"bbox": [5, 4, 7, 6]}},
        {1: {"bbox": [6, 4, 8, 6]}},
    ]


@pytest.fixture
def stub_team_assignment():
    return [
        {1: 1, 2: 2},
        {1: 1, 2: 2},
        {1: 1, 2: 2},
    ]
