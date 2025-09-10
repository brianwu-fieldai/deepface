# YOLOX + ArcFace in DeepFace

## Table of Contents
- [YOLOX + ArcFace in DeepFace](#yolox--arcface-in-deepface)
  - [Table of Contents](#table-of-contents)
  - [Model File Locations](#model-file-locations)
    - [Required Model Files](#required-model-files)
    - [Model Format Requirements](#model-format-requirements)
      - [YOLOX (.pth) Format](#yolox-pth-format)
      - [ArcFace (.onnx) Format](#arcface-onnx-format)
  - [CLI Commands](#cli-commands)
    - [Face Verification](#face-verification)
    - [Face Recognition](#face-recognition)
    - [Face Analysis](#face-analysis)
    - [Extract Faces](#extract-faces)
    - [Represent (Embeddings)](#represent-embeddings)
    - [Encryption](#encryption)
    - [Video Processing](#video-processing)
  - [Python API Examples](#python-api-examples)
    - [Face Detection](#face-detection)
    - [Face Verification](#face-verification-1)
    - [Face Recognition](#face-recognition-1)
    - [Face Analysis](#face-analysis-1)
  - [Troubleshooting](#troubleshooting)
    - [Common Issues and Solutions](#common-issues-and-solutions)

## Model File Locations

### Required Model Files

1. **YOLOX Detection Model**:
   - Path: `/deepface/models/detection/yolox.pth`
   - Type: Face detection model (MMDetection-style checkpoint)
   - Can be configured via environment variable:
     ```bash
     export YOLOX_CHECKPOINT_PATH="/path/to/your/yolox.pth"
     ```

2. **ArcFace Recognition Model**:
   - Path: `/deepface/test_conversion/cvtd_arcface.onnx`
   - Type: Face recognition model (ONNX format, PyTorch is also acceptable)

### Model Format Requirements

#### YOLOX (.pth) Format
The YOLOX model file should contain one of:
- Full model (with architecture and weights)
- State dictionary (`state_dict`)
- MMDetection-style checkpoint with `state_dict` under a key

#### ArcFace (.onnx) Format
Standard ONNX model with:
- Input shape: `[1, 3, 112, 112]` (RGB face image)
- Output shape: `[1, 512]` (face embedding vector)

## CLI Commands

### Face Verification

Compares two face images to determine if they show the same person.

```bash
# Basic verification
python -m cli.cli_module verify --img1 tests/dataset/img1.jpg --img2 tests/dataset/img2.jpg --detector yolox --model ArcFaceTorchONNX

# With custom threshold and metric
python -m cli.cli_module verify --img1 tests/dataset/img1.jpg --img2 tests/dataset/img2.jpg --detector yolox --model ArcFaceTorchONNX --metric cosine

# Save results to JSON
python -m cli.cli_module verify --img1 tests/dataset/img1.jpg --img2 tests/dataset/img2.jpg --detector yolox --model ArcFaceTorchONNX --json results.json

# Without face alignment (faster but less accurate)
python -m cli.cli_module verify --img1 tests/dataset/img1.jpg --img2 tests/dataset/img2.jpg --detector yolox --model ArcFaceTorchONNX --no-align

# Process even if face detection fails
python -m cli.cli_module verify --img1 tests/dataset/img1.jpg --img2 tests/dataset/img2.jpg --detector yolox --model ArcFaceTorchONNX --no-enforce-detection
```

### Face Recognition

Searches for matching faces in a directory of images.

```bash
# Basic face recognition
python -m cli.cli_module find --img tests/dataset/img1.jpg --db /path/to/face/database --detector yolox --model ArcFaceTorchONNX

# With verbose output
python -m cli.cli_module find --img tests/dataset/img1.jpg --db /path/to/face/database --detector yolox --model ArcFaceTorchONNX --verbose

# Save results to JSON
python -m cli.cli_module find --img tests/dataset/img1.jpg --db /path/to/face/database --detector yolox --model ArcFaceTorchONNX --json results.json

# With custom distance metric
python -m cli.cli_module find --img tests/dataset/img1.jpg --db /path/to/face/database --detector yolox --model ArcFaceTorchONNX --metric euclidean_l2
```

### Face Analysis

Analyzes facial attributes like age, gender, emotion, and race.

```bash
# Analyze all attributes
python -m cli.cli_module analyze --img tests/dataset/img1.jpg --detector yolox --actions age gender emotion race

# Analyze specific attributes
python -m cli.cli_module analyze --img tests/dataset/img1.jpg --detector yolox --actions age gender

# Save results to JSON
python -m cli.cli_module analyze --img tests/dataset/img1.jpg --detector yolox --actions age gender emotion --json analysis.json
```

### Extract Faces

Detects and extracts faces from images.

```bash
# Basic face extraction
python -m cli.cli_module extract-faces --img tests/dataset/img1.jpg --detector yolox

# Save extracted faces to specific directory
python -m cli.cli_module extract-faces --img tests/dataset/img1.jpg --detector yolox --save-dir extracted_faces

# With anti-spoofing check
python -m cli.cli_module extract-faces --img tests/dataset/img1.jpg --detector yolox --anti-spoofing

# Save metadata to JSON
python -m cli.cli_module extract-faces --img tests/dataset/img1.jpg --detector yolox --json faces.json

# Without alignment
python -m cli.cli_module extract-faces --img tests/dataset/img1.jpg --detector yolox --no-align
```

### Represent (Embeddings)

Extracts raw facial embedding vectors from images.

```bash
# Get embeddings with default settings
python -m cli.cli_module represent --img tests/dataset/img1.jpg --detector yolox --model ArcFaceTorchONNX

# Save embeddings to JSON
python -m cli.cli_module represent --img tests/dataset/img1.jpg --detector yolox --model ArcFaceTorchONNX --json embeddings.json

# Without face alignment
python -m cli.cli_module represent --img tests/dataset/img1.jpg --detector yolox --model ArcFaceTorchONNX --no-align
```

### Encryption

Encrypts embeddings and computes homomorphic cosine similarity (PHE).

```bash
# Basic encryption and comparison
python -m cli.cli_module encrypt --img1 tests/dataset/img1.jpg --img2 tests/dataset/img2.jpg --detector yolox --model ArcFaceTorchONNX

# Save encrypted results to JSON
python -m cli.cli_module encrypt --img1 tests/dataset/img1.jpg --img2 tests/dataset/img2.jpg --detector yolox --model ArcFaceTorchONNX --json encrypted.json
```

### Video Processing

Processes video files or webcam streams with face detection and tracking.

```bash
# Process a video file
python -m cli.cli_module video --source video.mp4 --output processed.mp4 --detector yolox --recognition-model ArcFaceTorchONNX

# Process webcam (device 0)
python -m cli.cli_module video --source 0 --output webcam.mp4 --detector yolox 

# With face recognition against a database
python -m cli.cli_module video --source video.mp4 --output processed.mp4 --detector yolox --recognition-model ArcFaceTorchONNX --db-path /path/to/face/database

# With custom detection interval (frames)
python -m cli.cli_module video --source video.mp4 --output processed.mp4 --detector yolox --detection-interval 5
```

## Python API Examples

### Face Detection

```python
from deepface.models.face_detection.YoloX import YOLOXFaceClient
import cv2

# Load image
img = cv2.imread("tests/dataset/img1.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Initialize detector
detector = YOLOXFaceClient(score_thr=0.25)

# Detect faces
faces = detector.detect_faces(img_rgb)

# Print results
print(f"Found {len(faces)} faces")
for i, face in enumerate(faces):
    print(f"Face {i+1}: x={face.x}, y={face.y}, w={face.w}, h={face.h}, confidence={face.confidence:.2f}")
```

### Face Verification

```python
from deepface import DeepFace

# Verify two faces
result = DeepFace.verify(
    img1_path="tests/dataset/img1.jpg",
    img2_path="tests/dataset/img2.jpg",
    detector_backend="yolox",
    model_name="ArcFaceTorchONNX",
    distance_metric="cosine"
)

# Print results
print(f"Verified: {result['verified']}")
print(f"Distance: {result['distance']:.4f}")
print(f"Threshold: {result['threshold']:.4f}")
print(f"Model: {result['model']}")
print(f"Detector: {result['detector_backend']}")
print(f"Similarity metric: {result['distance_metric']}")
```

### Face Recognition

```python
from deepface import DeepFace

# Find matches in a database
matches = DeepFace.find(
    img_path="tests/dataset/img1.jpg",
    db_path="/path/to/face/database",
    detector_backend="yolox",
    model_name="ArcFaceTorchONNX",
    distance_metric="cosine"
)

# Print results
print(f"Found {len(matches)} matches")
for i, match in enumerate(matches):
    print(f"Match {i+1}: {match['identity']}, distance: {match['distance']:.4f}")
```

### Face Analysis

```python
from deepface import DeepFace

# Analyze face attributes
results = DeepFace.analyze(
    img_path="tests/dataset/img1.jpg",
    detector_backend="yolox",
    actions=["age", "gender", "emotion", "race"]
)

# Print results
for i, result in enumerate(results):
    print(f"Face {i+1}:")
    print(f"  Age: {result.get('age', 'N/A')}")
    print(f"  Gender: {result.get('dominant_gender', 'N/A')}")
    print(f"  Emotion: {result.get('dominant_emotion', 'N/A')}")
    print(f"  Race: {result.get('dominant_race', 'N/A')}")
```

## Troubleshooting

### Common Issues and Solutions

1. **Model loading errors**:
   ```
   Solution: Ensure the model files are in the correct locations:
   - YOLOX: /deepface/models/detection/yolox.pth
   - ArcFace: /deepface/test_conversion/cvtd_arcface.onnx
   ```

2. **No face detected**:
   ```
   Solutions:
   - Try a different detector (--detector retinaface)
   - Use --no-enforce-detection to skip detection requirement
   - Ensure the image has a clear, frontal face
   ```

3. **TensorFlow conflicts**:
   ```
   Solution: Suppress TensorFlow with environment variable:
   export CUDA_VISIBLE_DEVICES=""
   ```

4. **Model path configuration**:
   ```
   Solution: Set model paths explicitly:
   export YOLOX_CHECKPOINT_PATH="/path/to/your/yolox.pth"
   ```

5. **Slow detection**:
   ```
   Solutions:
   - Use --no-align for faster processing (less accurate)
   - Set YOLOX_DEVICE="cpu" for CPU processing
   ```
