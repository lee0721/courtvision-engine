import argparse
import json
import logging
from pathlib import Path

from configs import(
    STUBS_DEFAULT_PATH,
    LOG_FILE,
    LOG_LEVEL,
    OUTPUT_VIDEO_PATH,
)
from utils.logging_utils import setup_logging
from video_analysis.video_analysis import VideoAnalysis

def parse_args():
    parser = argparse.ArgumentParser(description='Basketball Video Analysis')
    parser.add_argument('input_video', type=str, help='Path to input video file')
    parser.add_argument('--output_video', type=str, default=OUTPUT_VIDEO_PATH, 
                        help='Path to output video file')
    parser.add_argument('--stub_path', type=str, default=STUBS_DEFAULT_PATH,
                        help='Path to stub directory')
    return parser.parse_args()

def main():
    setup_logging(LOG_LEVEL, LOG_FILE)
    logger = logging.getLogger("courtvision.cli")
    args = parse_args()
    analyzer = VideoAnalysis(
        input_path=args.input_video,
        output_path=args.output_video,
        stub_path=args.stub_path,
        job_id="cli",
    )
    results = analyzer.run()
    output_json_path = Path(args.output_video).with_suffix(".json")
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with output_json_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=True)
    logger.info("analysis_result_written path=%s", output_json_path)

if __name__ == '__main__':
    main()
    
