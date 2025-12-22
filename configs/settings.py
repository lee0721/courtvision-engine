from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="COURTVISION_",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    stubs_dir: str = "stubs"
    output_dir: str = "output_videos"
    data_dir: str = "data"
    jobs_db_path: str = "data/jobs.db"

    player_detector_path: str = "models/player_detector.pt"
    ball_detector_path: str = "models/ball_detector_model.pt"
    arena_mark_detector_path: str = "models/arena_mark_detector.pt"
    action_recognition_model_path: str = "models/action_r2plus1d_best.pt"
    court_image_path: str = "images/basketball_court.png"

    team_1_class_name: str = "dark blue shirt"
    team_2_class_name: str = "white shirt"

    detection_batch_size: int = 20
    detection_confidence: float = 0.5

    action_device: str | None = None
    log_level: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
