import argparse
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
    args = parse_args()
    analyzer = VideoAnalysis(
        input_path=args.input_video,
        output_path=args.output_video,
        stub_path=args.stub_path,
        job_id="cli",
    )
    analyzer.run()

if __name__ == '__main__':
    main()
    
