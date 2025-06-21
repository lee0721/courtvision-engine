import torch
import cv2
import numpy as np
from utils.stubs_utils import read_stub, save_stub
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
from PIL import Image

class ActionRecognitionModel:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)
        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(model.fc.in_features, 10)
        )
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        self.model = model.to(self.device).eval()
        self.transform = Compose([
            Resize((112, 112)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.clip_len = 16

    def preprocess_clips(self, clips):
        tensor_batch = []
        for clip in clips:
            clip_tensor = []
            for frame in clip:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if frame_rgb.shape[2] == 1:
                    frame_rgb = np.repeat(frame_rgb, 3, axis=2)
                frame_pil = Image.fromarray(frame_rgb)
                frame_tensor = self.transform(frame_pil)
                clip_tensor.append(frame_tensor)
            clip_tensor = torch.stack(clip_tensor).permute(1, 0, 2, 3)  # C, T, H, W
            tensor_batch.append(clip_tensor)
        return torch.stack(tensor_batch).to(self.device)  # B, C, T, H, W

    def predict_batch(self, clips):
        if len(clips) == 0:
            return []
        input_tensor = self.preprocess_clips(clips)
        with torch.no_grad():
            output = self.model(input_tensor)
            predictions = torch.argmax(output, dim=1).cpu().numpy().tolist()
        return predictions
