import cv2
import numpy as np

class CourtKeypointDrawer:
    def __init__(self):
        self.keypoint_color = (44, 44, 255)  # 紅藍色 BGR
        self.text_color = (255, 255, 255)    # 白色
        self.radius = 4                      # 小圓點半徑

    def draw(self, frames, court_keypoints):
        output_frames = []
        for index, frame in enumerate(frames):
            annotated_frame = frame.copy()
            keypoints_tensor = court_keypoints[index]

            # ➤ 強制轉換為 (N, 2) 的 numpy array（可避免 unpack 錯誤）
            try:
                keypoints = keypoints_tensor.cpu().numpy().reshape(-1, 2)
            except Exception:
                continue  # 若轉換失敗，跳過該 frame

            for i, point in enumerate(keypoints):
                if len(point) != 2:
                    continue
                x, y = int(point[0]), int(point[1])

                # 畫 index 文字
                cv2.putText(annotated_frame, str(i), (x - 6, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1)
                # 畫圓點
                cv2.circle(annotated_frame, (x, y), self.radius, self.keypoint_color, -1)

            output_frames.append(annotated_frame)

        return output_frames