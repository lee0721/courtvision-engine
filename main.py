import argparse
from configs import(
    STUBS_DEFAULT_PATH,
    OUTPUT_VIDEO_PATH,
)
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
    args = parse_args()
    analyzer = VideoAnalysis(
        input_path=args.input_video,
        output_path=args.output_video,
        stub_path=args.stub_path
    )
    analyzer.run()

if __name__ == '__main__':
    main()
    