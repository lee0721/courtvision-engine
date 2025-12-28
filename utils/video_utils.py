"""
A module for reading and writing video files.

This module provides utility functions to load video frames into memory and save
processed frames back to video files, with support for common video formats.
"""

from __future__ import annotations

import logging
import cv2
import os
import shutil
import subprocess
from typing import TYPE_CHECKING, Sequence

from configs import OUTPUT_VIDEO_FPS
from utils.logging_utils import log_kv

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger("courtvision.video_utils")

def read_video(video_path: str) -> list["np.ndarray"]:
    """
    Read all frames from a video file into memory.

    Args:
        video_path (str): Path to the input video file.

    Returns:
        list: List of video frames as numpy arrays.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log_kv(logger, logging.ERROR, "video_open_failed", video_path=video_path)
        raise ValueError(f"Failed to open video: {video_path}")

    frames: list["np.ndarray"] = []
    frame_id = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame is None or frame.size == 0:
                log_kv(
                    logger,
                    logging.WARNING,
                    "empty_frame",
                    video_path=video_path,
                    frame_id=frame_id,
                )
                frame_id += 1
                continue
            frames.append(frame)
            frame_id += 1
    except Exception as exc:  # pragma: no cover - defensive
        log_kv(
            logger,
            logging.ERROR,
            "video_read_failed",
            video_path=video_path,
            frame_id=frame_id,
            error=str(exc),
        )
        raise
    finally:
        cap.release()
    return frames

def save_video(ouput_video_frames: Sequence["np.ndarray"], output_video_path: str) -> None:
    """
    Save a sequence of frames as a video file.

    Creates necessary directories if they don't exist and writes frames using XVID codec.

    Args:
        ouput_video_frames (list): List of frames to save.
        output_video_path (str): Path where the video should be saved.
    """
    # If folder doesn't exist, create it
    if not ouput_video_frames:
        log_kv(logger, logging.ERROR, "video_save_failed", error="no_frames")
        raise ValueError("No frames provided for output video.")

    output_dir = os.path.dirname(output_video_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        output_video_path,
        fourcc,
        OUTPUT_VIDEO_FPS,
        (ouput_video_frames[0].shape[1], ouput_video_frames[0].shape[0]),
    )
    if not out.isOpened():
        log_kv(logger, logging.ERROR, "video_writer_open_failed", output_video_path=output_video_path)
        raise ValueError(f"Failed to open output video: {output_video_path}")

    try:
        for frame in ouput_video_frames:
            out.write(frame)
    except Exception as exc:  # pragma: no cover - filesystem issues
        log_kv(
            logger,
            logging.ERROR,
            "video_write_failed",
            output_video_path=output_video_path,
            error=str(exc),
        )
        raise
    finally:
        out.release()


def transcode_to_h264(input_path: str, output_path: str) -> None:
    """
    Transcode a video file to H.264 (mp4) using ffmpeg for web compatibility.

    Args:
        input_path (str): Path to the input video file.
        output_path (str): Path where the transcoded video should be saved.
    """
    if not shutil.which("ffmpeg"):
        logger.error("ffmpeg_not_found")
        raise RuntimeError("ffmpeg is not installed or not in PATH")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output file
        "-v", "error",  # Less verbose
        "-i", input_path,
        "-c:v", "libx264",
        "-preset", "veryfast",  # Fast encoding
        "-pix_fmt", "yuv420p",  # Compatibility
        "-movflags", "+faststart",  # Web optimized
        output_path
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        log_kv(
            logger,
            logging.ERROR,
            "transcode_failed",
            input_path=input_path,
            output_path=output_path,
            error=str(exc.stderr),
        )
        raise RuntimeError(f"ffmpeg transcoding failed: {exc.stderr}") from exc

