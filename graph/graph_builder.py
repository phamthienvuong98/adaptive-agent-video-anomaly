"""
Graph Builder — CORE CONTRIBUTION MODULE

Converts detected objects (from YOLO) into a graph structure
suitable for GCN/GAT processing.

Node features:
    - position (cx, cy)           — centre of bounding box
    - bounding box size (w, h)    — normalised
    - velocity (vx, vy)           — computed from tracking
    - appearance embedding        — from CNN encoder

Edge construction:
    - distance-based (within threshold)
    - k-nearest neighbours fallback

Edge features:
    - Euclidean distance
    - relative position (dx, dy)
    - relative velocity (dvx, dvy)

Output:
    torch_geometric.data.Data with node_features, edge_index, edge_attr

References:
    - Hierarchical ST-GCN for Video Anomaly Detection
    - WAGCN (Adaptive GCN for Video Anomaly Detection)
    - MissionGNN (GNN + Knowledge Graph for anomaly detection)
"""

import numpy as np
import torch
from typing import List, Optional, Tuple, Dict, Any

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class Detection:
    """Represents a single detected object in a frame."""

    def __init__(
        self,
        bbox: Tuple[float, float, float, float],  # (x1, y1, x2, y2)
        class_id: int = 0,
        confidence: float = 1.0,
        track_id: Optional[int] = None,
    ):
        self.bbox = bbox
        self.class_id = class_id
        self.confidence = confidence
        self.track_id = track_id

        # Derived attributes
        x1, y1, x2, y2 = bbox
        self.cx = (x1 + x2) / 2.0
        self.cy = (y1 + y2) / 2.0
        self.w = x2 - x1
        self.h = y2 - y1

        # Velocity (computed later if previous frame available)
        self.vx = 0.0
        self.vy = 0.0

    def set_velocity(self, prev_cx: float, prev_cy: float, dt: float = 1.0):
        """Compute velocity from previous position."""
        self.vx = (self.cx - prev_cx) / dt
        self.vy = (self.cy - prev_cy) / dt


class GraphBuilder:
    """
    Builds a graph from object detections in a video frame.

    Supports:
        - Distance-based edge construction
        - K-nearest neighbour edges
        - Velocity computation from tracking
        - Appearance embedding injection

    This is the CORE CONTRIBUTION of the project — the graph structure
    captures spatial interactions between objects that CNN alone cannot model.
    """

    def __init__(
        self,
        distance_threshold: float = config.GRAPH_DISTANCE_THRESHOLD,
        use_velocity: bool = config.GRAPH_USE_VELOCITY,
        use_appearance: bool = config.GRAPH_USE_APPEARANCE,
        node_feature_dim: int = config.GRAPH_NODE_FEATURE_DIM,
        frame_width: int = config.FRAME_WIDTH,
        frame_height: int = config.FRAME_HEIGHT,
        k_neighbors: int = 5,
    ):
        self.distance_threshold = distance_threshold
        self.use_velocity = use_velocity
        self.use_appearance = use_appearance
        self.node_feature_dim = node_feature_dim
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.k_neighbors = k_neighbors

        # Appearance encoder (lazy loaded)
        self._appearance_encoder = None

    @property
    def appearance_encoder(self):
        """Lazy load appearance encoder to avoid import at init."""
        if self._appearance_encoder is None and self.use_appearance:
            from models.cnn_encoder import AppearanceEncoder
            self._appearance_encoder = AppearanceEncoder(embed_dim=self.node_feature_dim)
            self._appearance_encoder.eval()
        return self._appearance_encoder

    def _compute_velocities(
        self,
        detections: List[Detection],
        prev_detections: Optional[List[Detection]] = None,
    ):
        """
        Compute velocity for each detection by matching with
        previous frame detections (via track_id or nearest position).
        """
        if prev_detections is None or len(prev_detections) == 0:
            return

        # Build lookup: track_id → detection
        prev_by_id = {}
        prev_positions = []
        for d in prev_detections:
            if d.track_id is not None:
                prev_by_id[d.track_id] = d
            prev_positions.append((d.cx, d.cy))
        prev_positions = np.array(prev_positions)

        for det in detections:
            # Try matching by track_id first
            if det.track_id is not None and det.track_id in prev_by_id:
                prev = prev_by_id[det.track_id]
                det.set_velocity(prev.cx, prev.cy)
            elif len(prev_positions) > 0:
                # Fallback: nearest position matching
                dists = np.sqrt(
                    (prev_positions[:, 0] - det.cx) ** 2
                    + (prev_positions[:, 1] - det.cy) ** 2
                )
                nearest_idx = np.argmin(dists)
                if dists[nearest_idx] < self.distance_threshold:
                    prev = prev_detections[nearest_idx]
                    det.set_velocity(prev.cx, prev.cy)

    def _build_node_features(
        self,
        detections: List[Detection],
        frame: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """
        Build node feature matrix.

        Base features (per node):
            - normalised position (cx/W, cy/H)        : 2
            - normalised bbox size (w/W, h/H)         : 2
            - velocity (vx, vy)                        : 2
            - class one-hot (simplified)               : 1
            - confidence                               : 1
            Total base: 8

        + Optional appearance embedding: node_feature_dim
        """
        features = []

        for det in detections:
            # Base features (normalised)
            base = [
                det.cx / self.frame_width,
                det.cy / self.frame_height,
                det.w / self.frame_width,
                det.h / self.frame_height,
                det.vx / self.frame_width,   # normalised velocity
                det.vy / self.frame_height,
                float(det.class_id),
                det.confidence,
            ]
            features.append(base)

        node_feat = torch.tensor(features, dtype=torch.float32)  # (N, 8)

        # Add appearance embeddings if enabled
        if self.use_appearance and frame is not None and self.appearance_encoder is not None:
            crops = self._extract_crops(detections, frame)
            if crops is not None:
                with torch.no_grad():
                    appearance = self.appearance_encoder(crops)  # (N, embed_dim)
                node_feat = torch.cat([node_feat, appearance], dim=1)

        return node_feat

    def _extract_crops(
        self,
        detections: List[Detection],
        frame: np.ndarray,
        crop_size: Tuple[int, int] = (64, 64),
    ) -> Optional[torch.Tensor]:
        """Extract and resize object crops from frame."""
        from torchvision import transforms

        h, w = frame.shape[:2]
        crops = []
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        for det in detections:
            x1 = max(0, int(det.bbox[0]))
            y1 = max(0, int(det.bbox[1]))
            x2 = min(w, int(det.bbox[2]))
            y2 = min(h, int(det.bbox[3]))

            if x2 <= x1 or y2 <= y1:
                # Invalid crop → use zeros
                crops.append(torch.zeros(3, crop_size[0], crop_size[1]))
                continue

            crop = frame[y1:y2, x1:x2]
            if crop.shape[0] == 0 or crop.shape[1] == 0:
                crops.append(torch.zeros(3, crop_size[0], crop_size[1]))
                continue

            crops.append(transform(crop))

        return torch.stack(crops) if crops else None

    def _build_edges(
        self,
        detections: List[Detection],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build edge_index and edge_attr.

        Edge construction strategy:
            1. Connect all pairs within distance_threshold
            2. Ensure each node has at least k_neighbors edges (KNN fallback)
            3. Edges are bidirectional

        Edge features:
            - Euclidean distance (normalised)
            - relative position (dx, dy)
            - relative velocity (dvx, dvy)
            Total: 5
        """
        n = len(detections)
        if n <= 1:
            # Return empty graph
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 5), dtype=torch.float32)
            return edge_index, edge_attr

        # Compute pairwise distances
        positions = np.array([[d.cx, d.cy] for d in detections])
        velocities = np.array([[d.vx, d.vy] for d in detections])

        src_list, dst_list, attr_list = [], [], []

        for i in range(n):
            dists = np.sqrt(np.sum((positions - positions[i]) ** 2, axis=1))

            # Find neighbours: within threshold OR k-nearest
            within_threshold = np.where(
                (dists < self.distance_threshold) & (np.arange(n) != i)
            )[0]

            # KNN fallback
            if len(within_threshold) < self.k_neighbors:
                sorted_idx = np.argsort(dists)
                # Skip self (index 0 in sorted)
                neighbours = sorted_idx[1: self.k_neighbors + 1]
            else:
                neighbours = within_threshold

            for j in neighbours:
                # Edge features
                dist_norm = dists[j] / max(self.frame_width, self.frame_height)
                dx = (positions[j][0] - positions[i][0]) / self.frame_width
                dy = (positions[j][1] - positions[i][1]) / self.frame_height
                dvx = (velocities[j][0] - velocities[i][0]) / self.frame_width
                dvy = (velocities[j][1] - velocities[i][1]) / self.frame_height

                src_list.append(i)
                dst_list.append(j)
                attr_list.append([dist_norm, dx, dy, dvx, dvy])

        if len(src_list) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 5), dtype=torch.float32)
        else:
            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
            edge_attr = torch.tensor(attr_list, dtype=torch.float32)

        return edge_index, edge_attr

    def build(
        self,
        detections: List[Detection],
        frame: Optional[np.ndarray] = None,
        prev_detections: Optional[List[Detection]] = None,
    ) -> Dict[str, Any]:
        """
        Main entry point: build a graph from detections.

        Args:
            detections: list of Detection objects (from YOLO)
            frame: original frame (for appearance features)
            prev_detections: previous frame detections (for velocity)

        Returns:
            dict with keys:
                - node_features: (N, F) tensor
                - edge_index:    (2, E) tensor
                - edge_attr:     (E, 5) tensor
                - num_nodes:     int
        """
        if len(detections) == 0:
            # Empty graph (no detections)
            return {
                "node_features": torch.zeros((0, 8), dtype=torch.float32),
                "edge_index": torch.zeros((2, 0), dtype=torch.long),
                "edge_attr": torch.zeros((0, 5), dtype=torch.float32),
                "num_nodes": 0,
            }

        # Step 1: Compute velocities from tracking
        if self.use_velocity:
            self._compute_velocities(detections, prev_detections)

        # Step 2: Build node features
        node_features = self._build_node_features(detections, frame)

        # Step 3: Build edges
        edge_index, edge_attr = self._build_edges(detections)

        return {
            "node_features": node_features,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "num_nodes": len(detections),
        }

    def build_from_yolo_results(
        self,
        results,
        frame: Optional[np.ndarray] = None,
        prev_detections: Optional[List[Detection]] = None,
    ) -> Tuple[Dict[str, Any], List[Detection]]:
        """
        Build graph directly from YOLO inference results.

        Args:
            results: ultralytics YOLO result object
            frame: original frame
            prev_detections: previous detections for velocity

        Returns:
            (graph_dict, detections_list)
        """
        detections = []

        if results is not None and len(results) > 0:
            result = results[0]  # first image result
            boxes = result.boxes

            if boxes is not None and len(boxes) > 0:
                for i in range(len(boxes)):
                    bbox = boxes.xyxy[i].cpu().numpy()
                    cls = int(boxes.cls[i].cpu().item())
                    conf = float(boxes.conf[i].cpu().item())
                    track_id = None
                    if boxes.id is not None:
                        track_id = int(boxes.id[i].cpu().item())

                    det = Detection(
                        bbox=tuple(bbox),
                        class_id=cls,
                        confidence=conf,
                        track_id=track_id,
                    )
                    detections.append(det)

        graph = self.build(detections, frame, prev_detections)
        return graph, detections


def graph_dict_to_pyg(graph_dict: Dict[str, Any]):
    """
    Convert graph dict to PyTorch Geometric Data object.

    Requires torch_geometric to be installed.
    """
    from torch_geometric.data import Data

    return Data(
        x=graph_dict["node_features"],
        edge_index=graph_dict["edge_index"],
        edge_attr=graph_dict["edge_attr"],
        num_nodes=graph_dict["num_nodes"],
    )


if __name__ == "__main__":
    # Quick test with synthetic detections
    builder = GraphBuilder(use_appearance=False)

    dets = [
        Detection(bbox=(100, 100, 150, 250), class_id=0, confidence=0.9, track_id=1),
        Detection(bbox=(200, 120, 260, 280), class_id=0, confidence=0.85, track_id=2),
        Detection(bbox=(400, 300, 470, 450), class_id=2, confidence=0.7, track_id=3),
        Detection(bbox=(110, 110, 160, 260), class_id=0, confidence=0.6, track_id=4),
    ]

    graph = builder.build(dets)
    print(f"Nodes: {graph['num_nodes']}")
    print(f"Node features shape: {graph['node_features'].shape}")
    print(f"Edge index shape:    {graph['edge_index'].shape}")
    print(f"Edge attr shape:     {graph['edge_attr'].shape}")
