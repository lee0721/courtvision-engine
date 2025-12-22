import pytest

import trajectory_kinetics_analyzer.trajectory_kinetics_analyzer as tka


def test_calculate_distance():
    analyzer = tka.TrajectoryKineticsAnalyzer(100, 50, 10, 5)
    positions = [
        {1: (0, 0)},
        {1: (10, 0)},
        {1: (20, 0)},
    ]

    distances = analyzer.calculate_distance(positions)

    assert distances[0] == {}
    assert distances[1][1] == pytest.approx(0.4)
    assert distances[2][1] == pytest.approx(0.4)


def test_calculate_speed_window(monkeypatch):
    monkeypatch.setattr(tka, "SPEED_WINDOW_SIZE", 1)
    analyzer = tka.TrajectoryKineticsAnalyzer(100, 50, 10, 5)
    distances = [{}, {1: 0.4}, {1: 0.4}]

    speeds = analyzer.calculate_speed(distances, fps=10)

    assert speeds[1][1] == 0
    assert speeds[2][1] == pytest.approx(14.4)
