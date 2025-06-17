import cv2

class FrameNumberDrawer:
    def __init__(self, fps=30):
        self.fps = fps  # 每秒影格數

    def draw(self, frames):
        output_frames = []
        for i in range(len(frames)):
            frame = frames[i].copy()

            # 計算時間
            total_seconds = i / self.fps
            minutes = int(total_seconds // 60)
            seconds = int(total_seconds % 60)
            milliseconds = int((total_seconds - int(total_seconds)) * 10)  # 顯示1位小數，例如0.1秒

            time_text = f"{minutes:02}:{seconds:02}.{milliseconds}"

            # 畫白底黑字的框框
            text_size = cv2.getTextSize(time_text, cv2.FONT_HERSHEY_DUPLEX, 1, 2)[0]
            text_x = 10
            text_y = frame.shape[0] - 10
            cv2.rectangle(frame, (text_x - 5, text_y - text_size[1] - 5),
                          (text_x + text_size[0] + 5, text_y + 5), (255, 255, 255), -1)
            cv2.putText(frame, time_text, (text_x, text_y),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2)

            output_frames.append(frame)
        return output_frames