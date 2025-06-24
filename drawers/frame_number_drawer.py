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
            milliseconds = int((total_seconds - int(total_seconds)) * 100)  # 兩位小數（百分之一秒）

            timer_text = f"{minutes:02d}:{seconds:02d}.{milliseconds:02d}"

            # 畫白底黑字的框框
            text_size = cv2.getTextSize(timer_text, cv2.FONT_HERSHEY_DUPLEX, 1, 2)[0]
            text_x = 50
            text_y = frame.shape[0] - 150
            cv2.rectangle(frame, (text_x - 5, text_y - text_size[1] - 5),
                          (text_x + text_size[0] + 5, text_y + 5), (255, 255, 255), -1)
            cv2.putText(frame, timer_text, (text_x, text_y),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2)

            output_frames.append(frame)
        return output_frames