import cv2

class ActionRecognitionDrawer():
    """
    負責將動作辨識結果畫到每一幀上
    """
    def __init__(self, action_predictions):
        """
        初始化 ActionRecognitionDrawer
        
        Args:
            action_predictions (dict): 預測的動作標籤字典，格式為 {frame_index: action_label}
        """
        self.action_predictions = action_predictions  # 動作預測結果字典

    def draw(self, video_frames, player_tracks):
        """
        在每一幀上畫出動作標籤

        Args:
            video_frames (list): 每一幀的影像
            player_tracks (dict): 每一幀的球員追蹤資料，格式為 {frame_index: {player_id: {'bbox': (x1, y1, x2, y2)}}}
        
        Returns:
            list: 輸出帶有動作標籤的影片幀
        """
        output_video_frames = []

        for frame_idx, frame in enumerate(video_frames):
            output_frame = frame.copy()

            # 獲取當前幀的動作標籤
            action = self.action_predictions.get(frame_idx, None)

            # 如果有動作標籤，將其繪製到框框旁邊
            if action is not None:
                # 設定顯示位置（例如：框的旁邊）
                for player_id, player_data in player_tracks.get(frame_idx, {}).items():
                    bbox = player_data['bbox']
                    x1, y1, x2, y2 = bbox
                    position = [int((x1 + x2) / 2), int(y2)]
                    position[1] += 40  # 將位置稍微移動以顯示動作標籤
                    
                    # 顯示動作標籤
                    cv2.putText(output_frame, f"Action: {action}", (position[0], position[1]), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            output_video_frames.append(output_frame)

        return output_video_frames