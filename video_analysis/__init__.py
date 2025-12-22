from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .video_analysis import VideoAnalysis

__all__ = ["VideoAnalysis"]


def __getattr__(name: str):
    if name == "VideoAnalysis":
        from .video_analysis import VideoAnalysis as _VideoAnalysis

        return _VideoAnalysis
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
