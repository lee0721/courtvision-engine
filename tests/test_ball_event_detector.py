from ball_event_detector.ball_event_detector import BallEventDetector


def test_detect_passes_and_interceptions():
    ball_acquisition = [-1, 1, 1, 2, 2, 3]
    player_assignment = [
        {1: 1, 2: 1, 3: 2},
        {1: 1, 2: 1, 3: 2},
        {1: 1, 2: 1, 3: 2},
        {1: 1, 2: 1, 3: 2},
        {1: 1, 2: 1, 3: 2},
        {1: 1, 2: 1, 3: 2},
    ]

    detector = BallEventDetector()
    passes = detector.detect_passes(ball_acquisition, player_assignment)
    interceptions = detector.detect_interceptions(ball_acquisition, player_assignment)

    assert len(passes) == len(ball_acquisition)
    assert len(interceptions) == len(ball_acquisition)
    assert passes[3] == 1
    assert interceptions[5] == 2
