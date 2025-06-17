import torch
import cv2
import numpy as np
from models import action_r2plus1d_best  # 假設這是你的模型檔案
from torchvision.transforms import Compose, ToTensor, Normalize, Resize

# 載入訓練好的動作辨識模型
def load_model(model_path='models/action_r2plus1d_best.pt'):
    """
    載入訓練好的動作辨識模型
    """
    model = torch.load(model_path)
    model.eval()
    return model

# 對一段影片進行動作辨識
def predict_action(video_frames, model, frame_range=None):
    """
    預測影片中的動作
    
    Args:
        video_frames (list): 影片的所有幀，格式為 [frame1, frame2, ...]
        model (torch.nn.Module): 載入的動作辨識模型
        frame_range (tuple): 預設值為 None，若指定則僅處理範圍內的幀數 (start_frame, end_frame)
    
    Returns:
        actions (dict): 預測的動作結果，格式為 {frame_id: action_label}
    """
    actions = {}

    # 設定處理影片幀的範圍
    if frame_range:
        video_frames = video_frames[frame_range[0]:frame_range[1]]

    # 用來處理每一幀的預處理方法
    transform = Compose([
        Resize((112, 112)),  # 調整圖片尺寸
        ToTensor(),  # 轉換為 Tensor
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 標準化
    ])

    # 預測每一幀的動作
    for frame_idx, frame in enumerate(video_frames):
        # 將幀從 BGR 轉換為 RGB 並預處理
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = transform(frame_rgb).unsqueeze(0)  # 增加一維，使其符合模型輸入尺寸

        # 預測動作
        with torch.no_grad():
            output = model(frame_tensor)
            prediction = torch.argmax(output, dim=1).item()  # 預測結果為最大值的索引

        # 儲存每幀的動作預測
        actions[frame_idx] = prediction

    return actions