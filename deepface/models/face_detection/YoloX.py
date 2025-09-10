"""
YoloX face detection model for DeepFace
Simplified implementation that can work with or without MMDetection
"""

import os
import warnings
import numpy as np
from typing import List, Optional, Tuple

# DeepFace interfaces
from deepface.models.Detector import Detector, FacialAreaRegion

# Try MMDetection imports first
try:
    from mmdet.apis import init_detector, inference_detector
    MMDET_AVAILABLE = True
    print("✅ MMDetection available")
except ImportError:
    MMDET_AVAILABLE = False
    print("⚠️  MMDetection not available. Using fallback implementation.")


class YOLOXFaceClient(Detector):
    """
    DeepFace Detector adapter for a YOLOX model trained for face detection.
    
    This implementation tries MMDetection first, then falls back to a simple
    face detection method if MMDetection is not available.
    """

    def __init__(
        self,
        score_thr: float = 0.25,
        device: str = "cpu",
        max_dets: Optional[int] = None,
        assume_rgb: bool = True,
    ):
        """
        Args:
            score_thr: confidence threshold for a face to be returned
            device: device to run inference on
            max_dets: optional cap on number of faces per image (after score filter)
            assume_rgb: whether input images are in RGB format (True) or BGR (False)
        """
        self.score_thr = float(score_thr)
        self.max_dets = max_dets
        self.assume_rgb = bool(assume_rgb)
        self.device = device
        
        # Get paths from environment variables or use defaults
        self.cfg_path = os.environ.get("YOLOX_CONFIG_PATH")
        self.ckpt_path = os.environ.get("YOLOX_CHECKPOINT_PATH")
        
        if not self.cfg_path:
            self.cfg_path = "/Users/brianwu/Documents/Projects/FieldAI/deepface/configs/yolox_s_face_minimal.py"
        
        if not self.ckpt_path:
            self.ckpt_path = "/Users/brianwu/Documents/Projects/FieldAI/deepface/models/detection/yolox.pth"
        
        print(f"🔍 Looking for YOLOX checkpoint: {self.ckpt_path}")
        
        if MMDET_AVAILABLE and os.path.exists(self.ckpt_path):
            print("🚀 Attempting to use MMDetection...")
            try:
                self.model = self._load_with_mmdetection()
                self.use_mmdet = True
                print("✅ Successfully loaded with MMDetection")
            except Exception as e:
                print(f"⚠️  MMDetection loading failed: {e}")
                print("🔄 Falling back to simple implementation")
                self.use_mmdet = False
                self.model = self._load_fallback()
        else:
            print("🔄 Using fallback implementation")
            self.use_mmdet = False
            self.model = self._load_fallback()

    def _load_with_mmdetection(self):
        """Try to load model with MMDetection"""
        import os  # Add missing import
        
        if self.cfg_path and os.path.exists(self.cfg_path):
            # Monkey patch torch.load to temporarily disable weights_only
            import torch
            original_load = torch.load
            
            def patched_load(*args, **kwargs):
                # Force weights_only=False for MMDetection checkpoints
                kwargs['weights_only'] = False
                return original_load(*args, **kwargs)
            
            try:
                print("🔧 Temporarily patching torch.load for MMDetection checkpoint")
                torch.load = patched_load
                model = init_detector(self.cfg_path, self.ckpt_path, device=self.device)
                print("✅ Successfully loaded YOLOX model with MMDetection")
                return model
            finally:
                # Restore original torch.load
                torch.load = original_load
        else:
            print("⚠️  No config file provided")
            raise ValueError("MMDetection requires a config file for proper loading")

    def _load_fallback(self):
        """Fallback to OpenCV Haar cascades or simple detection"""
        print("� Loading fallback face detection (OpenCV Haar Cascades)")
        try:
            import cv2
            # Load OpenCV face detector as fallback
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            return cv2.CascadeClassifier(cascade_path)
        except Exception as e:
            print(f"⚠️  Could not load OpenCV cascade: {e}")
            return None

    def _xyxy_to_xywh(self, box_xyxy: np.ndarray) -> Tuple[int, int, int, int]:
        """Convert bounding box from xyxy to xywh format"""
        x1, y1, x2, y2 = box_xyxy.tolist()
        x, y = int(round(x1)), int(round(y1))
        w, h = int(round(x2 - x1)), int(round(y2 - y1))
        return x, y, w, h

    def _detect_with_mmdetection(self, img_in: np.ndarray) -> List[FacialAreaRegion]:
        """Detect faces using MMDetection"""
        result = inference_detector(self.model, img_in)
        
        # Handle different MMDetection result formats
        if hasattr(result, 'pred_instances'):
            pred = result.pred_instances
        elif hasattr(result, 'pred_instance'):
            pred = result.pred_instance
        elif isinstance(result, dict) and 'pred_instances' in result:
            pred = result['pred_instances']
        else:
            # Fallback: assume result is a list or array of detections
            print(f"🔍 MMDetection result type: {type(result)}")
            return []

        # Extract bboxes and scores
        try:
            if hasattr(pred, 'bboxes'):
                bboxes = pred.bboxes.detach().cpu().numpy()
            elif hasattr(pred, 'boxes'):
                bboxes = pred.boxes.detach().cpu().numpy()
            else:
                bboxes = np.array([])
                
            if hasattr(pred, 'scores'):
                scores = pred.scores.detach().cpu().numpy()
            else:
                scores = np.array([])
        except Exception as e:
            print(f"⚠️  Error extracting predictions: {e}")
            return []

        # Filter by confidence threshold
        resp = []
        if len(scores) > 0:
            keep = scores >= self.score_thr
            bboxes, scores = bboxes[keep], scores[keep]

            # Optional: limit number of detections
            if self.max_dets is not None and len(scores) > self.max_dets:
                order = np.argsort(-scores)[: self.max_dets]
                bboxes, scores = bboxes[order], scores[order]

            # Convert to DeepFace format
            for box, sc in zip(bboxes, scores):
                x, y, w, h = self._xyxy_to_xywh(box)
                
                if w > 0 and h > 0:
                    resp.append(
                        FacialAreaRegion(
                            x=x, y=y, w=w, h=h,
                            left_eye=None, right_eye=None,
                            confidence=float(sc),
                            nose=None, mouth_left=None, mouth_right=None,
                        )
                    )
        return resp

    def _detect_with_fallback(self, img_in: np.ndarray) -> List[FacialAreaRegion]:
        """Detect faces using OpenCV fallback"""
        if self.model is None:
            return []
            
        # Convert to grayscale for OpenCV
        gray = img_in if len(img_in.shape) == 2 else img_in[:,:,0] if img_in.shape[2] == 1 else np.dot(img_in[...,:3], [0.2989, 0.5870, 0.1140])
        gray = gray.astype(np.uint8)
        
        # Detect faces
        faces = self.model.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        resp = []
        for (x, y, w, h) in faces:
            # OpenCV gives reasonable confidence, but we'll use a fixed value
            confidence = 0.8  # Placeholder confidence
            if confidence >= self.score_thr:
                resp.append(
                    FacialAreaRegion(
                        x=int(x), y=int(y), w=int(w), h=int(h),
                        left_eye=None, right_eye=None,
                        confidence=float(confidence),
                        nose=None, mouth_left=None, mouth_right=None,
                    )
                )
        
        # Apply max_dets limit
        if self.max_dets is not None and len(resp) > self.max_dets:
            # Sort by confidence and take top detections
            resp.sort(key=lambda x: x.confidence, reverse=True)
            resp = resp[:self.max_dets]
            
        return resp

    def detect_faces(self, img: np.ndarray) -> List[FacialAreaRegion]:
        """
        Detect faces in an image using YOLOX or fallback method.

        Args:
            img (np.ndarray): HxWx3 image (RGB by default)

        Returns:
            List[FacialAreaRegion]: all detected faces
        """
        resp: List[FacialAreaRegion] = []
        
        if img is None or not isinstance(img, np.ndarray) or img.ndim != 3 or img.shape[2] != 3:
            return resp

        # Convert RGB → BGR if needed
        if self.assume_rgb:
            img_in = img[:, :, ::-1].copy()  # RGB to BGR
        else:
            img_in = img

        try:
            if self.use_mmdet:
                resp = self._detect_with_mmdetection(img_in)
            else:
                resp = self._detect_with_fallback(img_in)
        except Exception as e:
            print(f"❌ Face detection failed: {e}")
            return []

        return resp
