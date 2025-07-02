import cv2

class FrameNumberDrawer:
    def __init__(self):
        pass

    def draw(self, frames):
        output_frames = []
        for i in range(len(frames)):
            frame = frames[i].copy()

            frame_text = f"{i}"

            # 計算文字大小以畫框
            text_size = cv2.getTextSize(frame_text, cv2.FONT_HERSHEY_DUPLEX, 1, 2)[0]
            text_x = 50
            text_y = frame.shape[0] - 150  # 可以再往上移就調整這裡

            # 畫白底框框
            cv2.rectangle(frame, 
                          (text_x - 5, text_y - text_size[1] - 5),
                          (text_x + text_size[0] + 5, text_y + 5), 
                          (255, 255, 255), 
                          -1)

            # 畫黑字
            cv2.putText(frame, frame_text, (text_x, text_y),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2)

            output_frames.append(frame)
        return output_frames