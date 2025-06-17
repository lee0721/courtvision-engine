import torch
import cv2
import numpy as np
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights

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
        # Define the model architecture
        model = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)
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
        
        # Predict action for each frame
        for frame_idx, frame in enumerate(video_frames):
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            frame_tensor = transform(frame_rgb).unsqueeze(0)  # Add batch dimension
            
            # Perform action prediction
            with torch.no_grad():
                output = self.model(frame_tensor)
                prediction = torch.argmax(output, dim=1).item()  # Get the predicted action class
            
            actions[frame_idx] = prediction
        
        # Save the predictions as a stub
        save_stub(stub_path, actions)
        
        return actions