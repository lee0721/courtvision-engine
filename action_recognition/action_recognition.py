import torch
import cv2
import numpy as np
from utils.stubs_utils import read_stub, save_stub  # 确保您导入了 save_stub
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
from PIL import Image  # 引入 PIL 库，用于图像转换

class ActionRecognitionModel:
    """
    A class for loading and using a pre-trained action recognition model.
    This class is responsible for loading the model and performing action predictions on video frames.
    """
    def __init__(self, model_path):
        """
        Initialize the action recognition model by loading the pre-trained weights.
        
        Args:
            model_path (str): Path to the pre-trained model.
        """
        model = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)
        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(model.fc.in_features, 10)
        )
        # Load the model state dict (weights)
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        
        self.model = model.eval()  # Set the model to evaluation mode

    def predict(self, video_frames, read_from_stub=False, stub_path=None):
        """
        Perform action recognition on a list of video frames.
        
        Args:
            video_frames (list): A list of video frames (numpy.ndarray format).
            read_from_stub (bool, optional): If True, reads predictions from the stub file.
            stub_path (str, optional): Path to the stub file. If None, a default path may be used.
        
        Returns:
            dict: A dictionary of action predictions for each frame in the video.
        """
        actions = read_stub(read_from_stub, stub_path)
        
        if actions is not None:
            # If predictions are already cached in the stub, return them
            if len(actions) == len(video_frames):
                return actions
        
        actions = {}
        
        transform = Compose([
            Resize((112, 112)),  # Resize the frame to match the input size of the model
            ToTensor(),  # Convert the frame to a tensor
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the frame
        ])
        
        # 每 16 張為一段影片，為了符合模型要求的 5D 輸入
        clip_len = 16
        for start in range(0, len(video_frames) - clip_len + 1):
            clip = video_frames[start:start+clip_len]
            clip_tensor = []

            for frame in clip:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

                # If the frame has only 1 channel (grayscale), repeat it to make 3 channels
                if frame_rgb.shape[2] == 1:
                    frame_rgb = np.repeat(frame_rgb, 3, axis=2)  # Repeat grayscale to make 3 channels
                
                # Convert numpy array (BGR) to PIL Image
                frame_pil = Image.fromarray(frame_rgb)
                
                # Apply transformations to ensure we have 3 channels (RGB)
                frame_tensor = transform(frame_pil)
                clip_tensor.append(frame_tensor)

            # stack the frames and permute to match the expected input shape (3, 16, 112, 112)
            clip_tensor = torch.stack(clip_tensor).permute(1, 0, 2, 3).unsqueeze(0)  # Add batch dimension

            # Perform action prediction
            with torch.no_grad():
                output = self.model(clip_tensor)
                prediction = torch.argmax(output, dim=1).item()  # Get the predicted action class
            
            # Store the same prediction for all frames in the clip
            for i in range(clip_len):
                actions[start + i] = prediction
        
        # Save the predictions as a stub
        save_stub(stub_path, actions)
        
        return actions