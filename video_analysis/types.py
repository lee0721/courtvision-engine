from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

PlayerId = int
TeamId = int
Position = Sequence[float]
FramePositions = dict[PlayerId, Position]
PositionsByFrame = list[FramePositions]
PerFrameFloat = list[dict[PlayerId, float]]
PlayerAssignments = list[dict[PlayerId, TeamId]]
BallPossession = list[int]


def _int_list(values: Sequence[int]) -> list[int]:
    return [int(value) for value in values]


@dataclass(frozen=True)
class EventTimeline:
    passes: list[int]
    interceptions: list[int]

    def to_dict(self) -> dict[str, list[int]]:
        return {
            "passes": _int_list(self.passes),
            "interceptions": _int_list(self.interceptions),
        }


@dataclass(frozen=True)
class TeamControlRatio:
    team_1: float
    team_2: float
    none: float

    def to_dict(self) -> dict[str, float]:
        return {
            "team_1": float(self.team_1),
            "team_2": float(self.team_2),
            "none": float(self.none),
        }


@dataclass(frozen=True)
class PlayerMetric:
    total_distance_m: float
    avg_speed_kmh: float
    max_speed_kmh: float

    def to_dict(self) -> dict[str, float]:
        return {
            "total_distance_m": float(self.total_distance_m),
            "avg_speed_kmh": float(self.avg_speed_kmh),
            "max_speed_kmh": float(self.max_speed_kmh),
        }


@dataclass(frozen=True)
class AnalysisResult:
    input_video: str
    output_video: str
    frame_count: int
    events: EventTimeline
    ball_possession: BallPossession
    team_ball_control_ratio: TeamControlRatio
    player_metrics: dict[str, PlayerMetric]

    def to_dict(self) -> dict[str, object]:
        return {
            "input_video": self.input_video,
            "output_video": self.output_video,
            "frame_count": int(self.frame_count),
            "events": self.events.to_dict(),
            "ball_possession": _int_list(self.ball_possession),
            "team_ball_control_ratio": self.team_ball_control_ratio.to_dict(),
            "player_metrics": {
                player_id: metric.to_dict()
                for player_id, metric in self.player_metrics.items()
            },
        }
