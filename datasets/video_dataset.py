"""
VideoDataset — PyTorch Dataset for video anomaly detection.

Reads video files from disk, samples T frames per clip, and returns
tensors of shape (T, C, H, W) along with a binary label
(0 = normal, 1 = anomaly).

Also provides VideoGraphDataset for the graph-based pipeline where
each clip is processed through object detection → graph construction.
"""

import os
import glob
import random
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class VideoDataset(Dataset):
    """
    Standard video dataset for baseline (CNN + LSTM) pipeline.

    Structure expected:
        root/
            normal/   → label 0
            anomaly/  → label 1

    Each __getitem__ returns:
        frames : Tensor (T, C, H, W)   — sequence of T frames
        label  : int (0 or 1)
    """

    EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")

    def __init__(
        self,
        root_dir: str,
        sequence_length: int = config.SEQUENCE_LENGTH,
        frame_size: Tuple[int, int] = (config.FRAME_HEIGHT, config.FRAME_WIDTH),
        frame_stride: int = config.FRAME_STRIDE,
        transform: Optional[transforms.Compose] = None,
        is_train: bool = True,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.frame_size = frame_size
        self.frame_stride = frame_stride
        self.is_train = is_train

        # Default transforms (ImageNet normalisation)
        self.transform = transform or transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.frame_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # Collect video paths + labels
        self.samples: List[Tuple[str, int]] = []
        self._load_samples()

    # ----------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------
    def _load_samples(self):
        """Walk root_dir/normal and root_dir/anomaly."""
        for label_name, label_id in [("normal", 0), ("anomaly", 1)]:
            folder = os.path.join(self.root_dir, label_name)
            if not os.path.isdir(folder):
                continue
            for ext in self.EXTENSIONS:
                for path in sorted(glob.glob(os.path.join(folder, f"*{ext}"))):
                    self.samples.append((path, label_id))

        if len(self.samples) == 0:
            print(f"[WARNING] No video files found in {self.root_dir}")

    def _read_video_frames(self, video_path: str) -> List[np.ndarray]:
        """Read all frames from a video file."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        return frames

    def _sample_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Sample `sequence_length` frames with `frame_stride`.
        If video is too short, loop frames.
        If training, sample with random start offset.
        """
        total = len(frames)
        needed = self.sequence_length * self.frame_stride

        # Loop if video is too short
        if total < needed:
            factor = (needed // total) + 1
            frames = frames * factor

        total = len(frames)
        max_start = total - needed

        if self.is_train and max_start > 0:
            start = random.randint(0, max_start)
        else:
            start = max(0, max_start // 2)  # center crop for eval

        indices = list(range(start, start + needed, self.frame_stride))
        return [frames[i] for i in indices[: self.sequence_length]]

    # ----------------------------------------------------------
    # Dataset API
    # ----------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        video_path, label = self.samples[idx]

        # Read & sample frames
        all_frames = self._read_video_frames(video_path)
        if len(all_frames) == 0:
            # Return zeros if video unreadable
            dummy = torch.zeros(
                self.sequence_length, 3, self.frame_size[0], self.frame_size[1]
            )
            return dummy, label

        sampled = self._sample_frames(all_frames)

        # Apply transforms → stack into (T, C, H, W)
        tensors = [self.transform(f) for f in sampled]
        clip = torch.stack(tensors, dim=0)  # (T, C, H, W)

        return clip, label


class VideoGraphDataset(Dataset):
    """
    Graph-based dataset: each sample is a clip whose frames are
    first processed by YOLO → Graph Builder → PyG Data.

    This dataset lazily loads videos and builds graphs on-the-fly
    so we don't need to pre-compute features for the entire dataset.

    Returns:
        graphs : List[torch_geometric.data.Data]  — one graph per frame
        label  : int (0 or 1)
    """

    EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")

    def __init__(
        self,
        root_dir: str,
        sequence_length: int = config.SEQUENCE_LENGTH,
        frame_size: Tuple[int, int] = (config.FRAME_HEIGHT, config.FRAME_WIDTH),
        frame_stride: int = config.FRAME_STRIDE,
        detector=None,
        graph_builder=None,
        is_train: bool = True,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.frame_size = frame_size
        self.frame_stride = frame_stride
        self.is_train = is_train
        self.detector = detector
        self.graph_builder = graph_builder

        self.samples: List[Tuple[str, int]] = []
        self._load_samples()

    def _load_samples(self):
        for label_name, label_id in [("normal", 0), ("anomaly", 1)]:
            folder = os.path.join(self.root_dir, label_name)
            if not os.path.isdir(folder):
                continue
            for ext in self.EXTENSIONS:
                for path in sorted(glob.glob(os.path.join(folder, f"*{ext}"))):
                    self.samples.append((path, label_id))

    def _read_and_sample(self, video_path: str) -> List[np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        if len(frames) == 0:
            return []

        total = len(frames)
        needed = self.sequence_length * self.frame_stride
        if total < needed:
            frames = frames * ((needed // total) + 1)

        total = len(frames)
        max_start = total - needed
        start = random.randint(0, max(0, max_start)) if self.is_train else max(0, max_start // 2)
        indices = list(range(start, start + needed, self.frame_stride))
        return [frames[i] for i in indices[: self.sequence_length]]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[list, int]:
        video_path, label = self.samples[idx]
        frames = self._read_and_sample(video_path)

        if len(frames) == 0 or self.detector is None or self.graph_builder is None:
            return [], label

        graphs = []
        prev_detections = None
        for frame in frames:
            # Detect objects
            detections = self.detector(frame)
            # Build graph (with velocity if previous detections available)
            graph = self.graph_builder.build(
                detections=detections,
                frame=frame,
                prev_detections=prev_detections,
            )
            graphs.append(graph)
            prev_detections = detections

        return graphs, label


def get_dataloaders(
    train_dir: str = config.TRAIN_DIR,
    test_dir: str = config.TEST_DIR,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
) -> Dict[str, torch.utils.data.DataLoader]:
    """Convenience function to create train/test DataLoaders."""

    train_dataset = VideoDataset(train_dir, is_train=True)
    test_dataset = VideoDataset(test_dir, is_train=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return {"train": train_loader, "test": test_loader}
