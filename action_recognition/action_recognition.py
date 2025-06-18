from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
import torch
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from PIL import Image
import cv2
import numpy as np

class ActionRecognitionModel:
    def __init__(self, model_path):
        model = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)
        model.fc = torch.nn.Linear(model.fc.in_features, 10)  # 10 類動作分類

        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)

        self.model = model.eval()

    def predict(self, video_frames):
        actions = {}

        transform = Compose([
            Resize((112, 112)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        clip_len = 16
        for start in range(0, len(video_frames) - clip_len + 1):
            clip = video_frames[start:start+clip_len]
            clip_tensor = []

            for frame in clip:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if frame_rgb.shape[2] == 1:
                    frame_rgb = np.repeat(frame_rgb, 3, axis=2)

                frame_pil = Image.fromarray(frame_rgb)
                frame_tensor = transform(frame_pil)
                clip_tensor.append(frame_tensor)

            clip_tensor = torch.stack(clip_tensor).permute(1, 0, 2, 3).unsqueeze(0)

            with torch.no_grad():
                output = self.model(clip_tensor)
                prediction = torch.argmax(output, dim=1).item()

            for i in range(clip_len):
                actions[start + i] = prediction

        return actions