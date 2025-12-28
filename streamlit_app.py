import os
import time
from pathlib import Path
from typing import Optional

import requests
from requests import RequestException
import streamlit as st


API_BASE_URL = os.getenv("COURTVISION_API_URL", "http://localhost:8000")
INPUT_DIR = Path("input_videos")
OUTPUT_DIR = Path("output_videos")
STUBS_DIR = Path("stubs")
# Increasing default timeout to 45 minutes as ML models can be slow on CPU
POLL_TIMEOUT_SECONDS = int(os.getenv("COURTVISION_POLL_TIMEOUT_SECONDS", 2700))
POLL_INTERVAL_SECONDS = int(os.getenv("COURTVISION_POLL_INTERVAL_SECONDS", 5))
STATUS_TIMEOUT_SECONDS = int(os.getenv("COURTVISION_STATUS_TIMEOUT_SECONDS", 60))

PIPELINE_STAGES = [
    "Submit",
    "Read video",
    "Init models",
    "Tracking",
    "Arena marks",
    "Action recognition",
    "Drawing",
    "Save result",
]
STAGE_TO_PIPELINE_IDX = {
    "read_video": 1,
    "init_models": 2,
    "tracking": 3,
    "ball_processing": 3,
    "team_assignment": 3,
    "arena_marks": 4,
    "action_recognition": 5,
    "tactical_view": 6,
    "trajectory_metrics": 6,
    "drawing": 6,
    "save_video": 7,
    "build_result": 7,
    "completed": 7,
    "failed": 3,
}


def _list_videos() -> list[Path]:
    return sorted(p for p in INPUT_DIR.glob("*.mp4") if p.is_file())


def _choose_video(videos: list[Path]) -> Path:
    """
    Playlist-style selector: list on the left, player on the right.
    Keeps the selected index in session_state so the choice persists across reruns.
    """
    if "video_idx" not in st.session_state:
        st.session_state["video_idx"] = 0

    names = [p.name for p in videos]
    current_idx = min(st.session_state["video_idx"], max(0, len(videos) - 1))
    st.session_state["video_idx"] = current_idx

    col_list, col_player = st.columns([1, 2])
    with col_list:
        idx = st.radio(
            "Playlist",
            list(range(len(videos))),
            format_func=lambda i: names[i],
            index=current_idx,
        )
        st.session_state["video_idx"] = idx

    selected = videos[st.session_state["video_idx"]]

    with col_player:
        st.caption(f"Preview: {selected.name}")
        st.video(str(selected))

    st.session_state["selected_video"] = str(selected)
    return selected


def _submit_job(video_path: Path, use_stubs: bool, api_base: str) -> str:
    # Output directly to the output directory; container mapping handles the rest.
    output_filename = f"{video_path.stem}_analyzed.mp4"

    payload = {
        "input_video_path": str(video_path),
        "output_video": str(OUTPUT_DIR / output_filename),
        "stub_path": str(STUBS_DIR),
        "use_stubs": use_stubs,
    }
    try:
        resp = requests.post(f"{api_base}/analysis", json=payload, timeout=15)
        resp.raise_for_status()
        return resp.json()["job_id"]
    except RequestException as exc:
        st.error(f"Failed to submit job to {api_base}. Is the backend running?")
        raise exc


def _render_pipeline_steps(container, stage_idx: int, is_done: bool = False, is_failed: bool = False) -> None:
    items = []
    for i, name in enumerate(PIPELINE_STAGES):
        if is_failed and i == stage_idx:
            marker = "❌"
        elif is_done or i < stage_idx:
            marker = "✅"
        elif i == stage_idx:
            marker = "⚙️"
        else:
            marker = "⏳"
        items.append(f"{marker} {name}")
    container.markdown("\n".join(items))


def _poll_status(job_id: str, api_base: str, progress, steps_placeholder) -> dict:
    attempts = max(1, int(POLL_TIMEOUT_SECONDS / POLL_INTERVAL_SECONDS))
    for i in range(attempts):
        fallback_pct = min(100, int(((i + 1) / attempts) * 100))

        try:
            resp = requests.get(f"{api_base}/status/{job_id}", timeout=STATUS_TIMEOUT_SECONDS)
        except RequestException:
            progress.progress(fallback_pct, text=f"Job {job_id} status: retrying connection…")
            time.sleep(POLL_INTERVAL_SECONDS)
            continue

        if resp.status_code == 404:
            raise RuntimeError(f"Job {job_id} not found")

        resp.raise_for_status()
        data = resp.json()
        status = data["status"]
        stage = data.get("stage")
        progress_pct = data.get("progress")
        runtime_ms = data.get("runtime_ms") or 0

        stage_idx = STAGE_TO_PIPELINE_IDX.get(stage, 1 if status == "running" else 0)
        pct_display = (
            max(0, min(100, int(progress_pct)))
            if progress_pct is not None
            else fallback_pct
        )

        if status == "queued":
            stage_idx = 0
            _render_pipeline_steps(steps_placeholder, stage_idx, is_done=False)
        elif status == "running":
            _render_pipeline_steps(steps_placeholder, stage_idx, is_done=False)
            progress.progress(
                pct_display,
                text=(
                    f"Job {job_id} is running"
                    + (f" — {stage}" if stage else "")
                    + f" (Elapsed: {runtime_ms/1000:.1f}s)"
                ),
            )
        elif status == "completed":
            _render_pipeline_steps(steps_placeholder, len(PIPELINE_STAGES) - 1, is_done=True)
            progress.progress(100, text=f"Job {job_id} completed!")
            return data
        elif status == "failed":
            _render_pipeline_steps(steps_placeholder, 3, is_failed=True)  # Assume failed during run
            raise RuntimeError(f"Job {job_id} failed: {data.get('error_message')}")

        time.sleep(POLL_INTERVAL_SECONDS)

    raise TimeoutError(
        f"Job {job_id} did not complete in time ({POLL_TIMEOUT_SECONDS}s). "
        "Check backend logs."
    )


def _fetch_results(job_id: str, api_base: str) -> Optional[dict]:
    resp = requests.get(f"{api_base}/results/{job_id}", timeout=30)
    if resp.status_code == 202:
        return None
    resp.raise_for_status()
    return resp.json()


def _to_host_path(path_str: str) -> Path:
    """
    Map container-style paths (/app/...) back to host paths when using Docker volumes.
    Falls back to the given path if no mapping is found.
    """
    path = Path(path_str)
    if path.exists():
        return path
    try:
        rel = path.relative_to("/app")
        for base in (Path.cwd(), Path.cwd().parent):
            candidate = base / rel
            if candidate.exists():
                return candidate
    except ValueError:
        pass

    alt = OUTPUT_DIR / path.name
    if alt.exists():
        return alt

    return path


def _display_video(path: Path) -> None:
    if not path.exists():
        st.info(f"Video file not found: {path}")
        return
    try:
        st.video(str(path))
    except Exception as exc:
        st.error(f"Failed to load video: {exc}")


def main() -> None:
    st.set_page_config(page_title="CourtVision AI", page_icon="🏀", layout="wide")
    st.title("🏀 CourtVision AI Analytics")
    st.markdown(
        """
        **CourtVision Engine** uses computer vision to analyze basketball game footage.
        Select a video from the `input_videos` directory to begin analysis.
        """
    )

    api_base = st.text_input("API URL", value=API_BASE_URL, help="Docker internal URL: http://api:8000")

    # Check backend health
    try:
        requests.get(f"{api_base}/docs", timeout=2)
        st.success(f"Connected to Backend: {api_base}", icon="✅")
    except Exception:
        st.error(f"Cannot connect to Backend at {api_base}. Ensure Docker containers are running.", icon="⚠️")

    use_stubs = st.toggle("Use Stubs (Fast Dev Mode)", value=True, help="Use pre-calculated tracks to skip slow inference.")

    videos = _list_videos()
    if not videos:
        st.warning("No videos found in `input_videos/` directory.")
        return

    selected_video = _choose_video(videos)

    if st.button("Start Analysis", type="primary", use_container_width=True):
        progress = st.progress(0, text="Submitting job…")
        steps_placeholder = st.empty()

        with st.status("🚀 Processing Video...", expanded=True) as status_box:
            try:
                job_id = _submit_job(selected_video, use_stubs, api_base)
                status_box.write(f"Job ID: `{job_id}` created.")

                job_status = _poll_status(job_id, api_base, progress, steps_placeholder)
                status_box.update(label="✅ Analysis Complete", state="complete", expanded=False)

                result = _fetch_results(job_id, api_base)
            except Exception as exc:
                status_box.update(label="❌ Analysis Failed", state="error")
                st.error(f"Error: {exc}")
                return

        # Display Results
        st.divider()
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Analysis Metrics")
            if result and result.get("result"):
                st.json(result["result"])
            else:
                st.info("No detailed metrics available.")

        with col2:
            st.subheader("Output Video")
            output_video_path = job_status.get("output_video_path")
            if output_video_path:
                local_path = _to_host_path(output_video_path)
                if local_path.exists():
                    st.success(f"Playing backend output: {local_path.name}")
                    _display_video(local_path)
                    with open(local_path, "rb") as vfile:
                        st.download_button("Download Video", vfile, file_name=local_path.name)
                else:
                    st.error(f"Output video file not found locally: {local_path}")
            else:
                st.warning("Job completed but no output video path returned.")


if __name__ == "__main__":
    main()
