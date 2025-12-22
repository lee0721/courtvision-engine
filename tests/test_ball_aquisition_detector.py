from ball_aquisition.ball_aquisition_detector import BallAquisitionDetector


def test_detect_ball_possession_requires_min_frames(
    stub_player_tracks,
    stub_ball_tracks,
):
    detector = BallAquisitionDetector()
    detector.min_frames = 2
    detector.containment_threshold = 0.5
    detector.possession_threshold = 1000

    possession = detector.detect_ball_possession(stub_player_tracks, stub_ball_tracks)

    assert possession == [-1, 1, 1]
