from __future__ import print_function
from imutils.object_detection import non_max_suppression
import cv2
import numpy as np
from easydict import EasyDict
from random import randint
import sys
from imutils.video import FPS
import os

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
from ultralytics import YOLO

from utils.checkpoints import load_weights
from deep_sort_realtime.deepsort_tracker import DeepSort

(major, minor) = cv2.__version__.split(".")[:2]
print(cv2.__version__)

args = EasyDict({
    'detector': "yolov5",
    'videoPath': "input_videos/video_1.mp4",
    'classes': ["person"],
    'tracker': "CSRT",
    'trackerTypes': ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT'],
    'singleTracker': False,
    'draw_line': False,
    'base_model_name': 'r2plus1d_multiclass',
    'pretrained': True,
    'lr': 0.0001,
    'start_epoch': 19,
    'num_classes': 10,
    'labels': {"0": "block", "1": "pass", "2": "run", "3": "dribble", "4": "shoot", "5": "ball in hand", "6": "defense", "7": "pick", "8": "no_action", "9": "walk", "10": "discard"},
    'model_path': "model_checkpoints/r2plus1d_augmented-2/r2plus1d_multiclass_20_0.0001_20250623_031019.pt",
    'history_path': "histories/history_r2plus1d_augmented-2.txt",
    'seq_length': 16,
    'vid_stride': 8,
    'output_path': "output_videos/"
})

def extractFrame(videoPath):
    cap = cv2.VideoCapture(videoPath)
    model = YOLO("model/player_detector.pt")
    deepsort = DeepSort(max_age=30)

    videoFrames = []
    playerTracks = {}
    colors = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        Width = frame.shape[1]
        Height = frame.shape[0]
        results = model(frame)[0]

        detections = []
        for box in results.boxes:
            if int(box.cls[0]) == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append(([x1, y1, x2 - x1, y2 - y1], 0.9, 'person'))

        tracks = deepsort.update_tracks(detections, frame=frame)

        current_boxes = {}
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            l, t, r, b = track.to_ltrb()
            box = [int(l), int(t), int(r - l), int(b - t)]
            current_boxes[track_id] = box
            if track_id not in playerTracks:
                playerTracks[track_id] = []
                colors[track_id] = tuple(map(int, np.random.randint(0, 255, size=3)))
            playerTracks[track_id].append(box)

        videoFrames.append(frame)

    cap.release()
    return videoFrames, playerTracks, Width, Height, colors

def cropVideo(clip, crop_window, player=0):

    video = []
    #print(len(clip))
    for i, frame in enumerate(clip):
        x = int(crop_window[i][player][0])
        y = int(crop_window[i][player][1])
        w = int(crop_window[i][player][2])
        h = int(crop_window[i][player][3])

        cropped_frame = frame[y:y+h, x:x+w]
        # resize to 128x176
        try:
            resized_frame = cv2.resize(
                cropped_frame,
                dsize=(int(128),
                       int(176)),
                interpolation=cv2.INTER_NEAREST
            )
        except:
            # Use previous frame
            if len(video) == 0:
                resized_frame = np.zeros((int(176), int(128), 3), dtype=np.uint8)
            else:
                resized_frame = video[i-1]
        assert resized_frame.shape == (176, 128, 3)
        video.append(resized_frame)

    return video

def inference_batch(batch):
    # (batch, t, h, w, c) --> (batch, c, t, h, w)
    batch = batch.permute(0, 4, 1, 2, 3)
    return batch

def cropWindows(vidFrames, playerTracks, seq_length=16, vid_stride=8):
    player_frames = {}
    for player_id, boxes in playerTracks.items():
        player_frames[player_id] = []

        total_frames = len(boxes)
        n_clips = total_frames // vid_stride
        for clip_n in range(n_clips):
            start_idx = clip_n * vid_stride
            end_idx = start_idx + seq_length
            if end_idx <= total_frames:
                clip = vidFrames[start_idx:end_idx]
                crop_window = {i: [boxes[start_idx + i]] for i in range(seq_length)}
                player_frames[player_id].append(np.asarray(cropVideo(clip, crop_window, 0)))

        # Optional: handle tail
        if total_frames % vid_stride != 0 and total_frames > seq_length:
            start_idx = total_frames - seq_length
            clip = vidFrames[start_idx:start_idx+seq_length]
            crop_window = {i: [boxes[start_idx + i]] for i in range(seq_length)}
            player_frames[player_id].append(np.asarray(cropVideo(clip, crop_window, 0)))

    return player_frames

def writeVideo(videoPath, videoFrames, playerTracks, predictions, colors, frame_width=1280, frame_height=720, vid_stride=8):
    out = cv2.VideoWriter(videoPath, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 10, (frame_width, frame_height))

    for frame_idx, frame in enumerate(videoFrames):
        for track_id, boxes in playerTracks.items():
            if frame_idx < len(boxes):
                box = boxes[frame_idx]
                x, y, w, h = box
                p1 = (int(x), int(y))
                p2 = (int(x + w), int(y + h))
                color = colors[track_id]
                cv2.rectangle(frame, p1, p2, color, 2, 1)

                # 預測標籤貼在框上（按 clip index 算）
                clip_idx = frame_idx // vid_stride
                try:
                    label_id = predictions[track_id][clip_idx]
                    label = args.labels[str(label_id)]
                    cv2.putText(frame, label, (p1[0] - 10, p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                except (KeyError, IndexError):
                    # 預測不存在或不足 clip 數，不畫文字
                    continue

        # 輸出影格
        out.write(frame)

    out.release()
    cv2.destroyAllWindows()

def main():
    os.makedirs(args.output_path, exist_ok=True)

    # 取得影片影格、追蹤的每位球員的框、解析度與顏色
    videoFrames, playerTracks, Width, Height, colors = extractFrame(args.videoPath)

    print("Video Dimensions: ({}, {})".format(Width, Height))
    print("Total Frames: {}".format(len(videoFrames)))
    print("Tracked Players: {}".format(len(playerTracks)))

    if len(videoFrames) == 0 or len(playerTracks) == 0:
        print("⚠️ 沒有偵測到畫面或人，請檢查影片或模型")
        return

    # 擷取每位球員的時序視窗
    frames = cropWindows(videoFrames, playerTracks, seq_length=args.seq_length, vid_stride=args.vid_stride)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 載入動作辨識模型
    weights = R2Plus1D_18_Weights.KINETICS400_V1 if args.pretrained else None
    model = r2plus1d_18(weights=weights)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, args.num_classes, bias=True)
    model = load_weights(model, args)

    if torch.cuda.is_available():
        model = model.to(device)
    model.eval()

    predictions = {}
    for track_id, clips in frames.items():
        if len(clips) == 0:
            continue

        print(f"🔍 推論 Track ID {track_id}，Clip 數：{len(clips)}")
        input_frames = inference_batch(torch.FloatTensor(np.array(clips)))
        input_frames = input_frames.to(device=device)

        with torch.no_grad():
            outputs = model(input_frames)
            _, preds = torch.max(outputs, 1)

        predictions[track_id] = preds.cpu().numpy().tolist()

    print("✅ 所有玩家推論完成")
    output_path = os.path.join(args.output_path, "video_result_1.mp4")
    writeVideo(output_path, videoFrames, playerTracks, predictions, colors,
               frame_width=Width, frame_height=Height, vid_stride=args.vid_stride)

if __name__ == "__main__":
    main()
