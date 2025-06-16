import cv2

class CourtKeypointDrawer:
    def __init__(self):
        self.keypoint_color = (44, 44, 255)  # 紅藍色 BGR
        self.text_color = (255, 255, 255)    # 白色
        self.radius = 4                      # 小圓點半徑

    def draw(self, frames, court_keypoints):
        output_frames = []
        for index, frame in enumerate(frames):
            annotated_frame = frame.copy()
            keypoints = court_keypoints[index].cpu().numpy()

            for i, (x, y) in enumerate(keypoints):
                x, y = int(x), int(y)
                # 畫數字在 keypoint 上方
                cv2.putText(annotated_frame, str(i), (x - 6, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1)
                # 畫圓點在 keypoint 原位
                cv2.circle(annotated_frame, (x, y), self.radius, self.keypoint_color, -1)

            output_frames.append(annotated_frame)

        return output_frames