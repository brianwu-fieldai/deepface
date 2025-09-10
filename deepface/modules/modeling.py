# built-in dependencies
from typing import Any

# Import face detection models (these work without TensorFlow)
from deepface.models.face_detection import (
    FastMtCnn,
    MediaPipe,
    MtCnn,
    OpenCv,
    Dlib as DlibDetector,
    RetinaFace,
    Ssd,
    Yolo as YoloFaceDetector,
    YuNet,
    CenterFace,
    YoloX,  # Add YoloX import
)
from deepface.models.demography import Age, Gender, Race, Emotion
from deepface.models.spoofing import FasNet

# Dynamically import facial recognition models
facial_recognition_models = {}

# Try to import TensorFlow-free models first
try:
    from deepface.models.facial_recognition.VGGFaceTorchONNX import VggFaceTorchONNXClient
    facial_recognition_models['VGG-Face'] = VggFaceTorchONNXClient
except ImportError:
    pass

try:
    from deepface.models.facial_recognition.ArcFaceTorchONNX import ArcFaceTorchONNXClient
    facial_recognition_models['ArcFace'] = ArcFaceTorchONNXClient
    facial_recognition_models['ArcFaceTorchONNX'] = ArcFaceTorchONNXClient
except ImportError:
    pass

try:
    from deepface.models.facial_recognition.FacenetTorchONNX import FaceNetTorchONNXClient
    facial_recognition_models['Facenet'] = FaceNetTorchONNXClient
    facial_recognition_models['Facenet512'] = FaceNetTorchONNXClient
except ImportError:
    pass

try:
    from deepface.models.facial_recognition.DeepIDTorchONNX import DeepIdTorchONNXClient
    facial_recognition_models['DeepID'] = DeepIdTorchONNXClient
except ImportError:
    pass

try:
    from deepface.models.facial_recognition.GhostFaceNetTorchONNX import GhostFaceNetTorchONNXClient
    facial_recognition_models['GhostFaceNet'] = GhostFaceNetTorchONNXClient
except ImportError:
    pass

try:
    from deepface.models.facial_recognition.TinyTorchFace import TinyTorchFaceClient
    facial_recognition_models['TinyTorchFace'] = TinyTorchFaceClient
except ImportError:
    pass

try:
    from deepface.models.facial_recognition.Buffalo_L import Buffalo_L
    facial_recognition_models['Buffalo_L'] = Buffalo_L
except ImportError:
    pass

# Fallback to TensorFlow models if available
try:
    from deepface.models.facial_recognition import (
        VGGFace,
        OpenFace,
        FbDeepFace,
        DeepID,
        ArcFace,
        SFace,
        Dlib,
        Facenet,
        GhostFaceNet,
    )
    # Only add TensorFlow models if TorchONNX versions aren't available
    if 'VGG-Face' not in facial_recognition_models:
        facial_recognition_models['VGG-Face'] = VGGFace.VggFaceClient
    if 'ArcFace' not in facial_recognition_models:
        facial_recognition_models['ArcFace'] = ArcFace.ArcFaceClient
    if 'Facenet' not in facial_recognition_models:
        facial_recognition_models['Facenet'] = Facenet.FaceNet128dClient
        facial_recognition_models['Facenet512'] = Facenet.FaceNet512dClient
    if 'DeepID' not in facial_recognition_models:
        facial_recognition_models['DeepID'] = DeepID.DeepIdClient
    if 'GhostFaceNet' not in facial_recognition_models:
        facial_recognition_models['GhostFaceNet'] = GhostFaceNet.GhostFaceNetClient
    
    # TensorFlow-only models
    facial_recognition_models['OpenFace'] = OpenFace.OpenFaceClient
    facial_recognition_models['DeepFace'] = FbDeepFace.DeepFaceClient
    facial_recognition_models['SFace'] = SFace.SFaceClient
    facial_recognition_models['Dlib'] = Dlib.DlibClient
except ImportError:
    # TensorFlow not available, that's fine - we have TorchONNX models
    pass


def build_model(task: str, model_name: str) -> Any:
    """
    This function loads a pre-trained models as singletonish way
    Parameters:
        task (str): facial_recognition, facial_attribute, face_detector, spoofing
        model_name (str): model identifier
            - VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID, Dlib,
                ArcFace, SFace and GhostFaceNet for face recognition
            - Age, Gender, Emotion, Race for facial attributes
            - opencv, mtcnn, ssd, dlib, retinaface, mediapipe, yolov8, 'yolov11n',
                'yolov11s', 'yolov11m', yunet, fastmtcnn or centerface for face detectors
            - Fasnet for spoofing
    Returns:
            built model class
    """

    # singleton design pattern
    global cached_models

    models = {
        "facial_recognition": facial_recognition_models,
        "spoofing": {
            "Fasnet": FasNet.Fasnet,
        },
        "facial_attribute": {
            "Emotion": Emotion.EmotionClient,
            "Age": Age.ApparentAgeClient,
            "Gender": Gender.GenderClient,
            "Race": Race.RaceClient,
        },
        "face_detector": {
            "opencv": OpenCv.OpenCvClient,
            "mtcnn": MtCnn.MtCnnClient,
            "ssd": Ssd.SsdClient,
            "dlib": DlibDetector.DlibClient,
            "retinaface": RetinaFace.RetinaFaceClient,
            "mediapipe": MediaPipe.MediaPipeClient,
            "yolov8": YoloFaceDetector.YoloDetectorClientV8n,
            "yolov11n": YoloFaceDetector.YoloDetectorClientV11n,
            "yolov11s": YoloFaceDetector.YoloDetectorClientV11s,
            "yolov11m": YoloFaceDetector.YoloDetectorClientV11m,
            "yunet": YuNet.YuNetClient,
            "fastmtcnn": FastMtCnn.FastMtCnnClient,
            "centerface": CenterFace.CenterFaceClient,
            "yolox": YoloX.YOLOXFaceClient,  # Add YoloX detector
        },
    }

    if models.get(task) is None:
        raise ValueError(f"unimplemented task - {task}")

    if not "cached_models" in globals():
        cached_models = {current_task: {} for current_task in models.keys()}

    if cached_models[task].get(model_name) is None:
        model = models[task].get(model_name)
        if model:
            cached_models[task][model_name] = model()
        else:
            raise ValueError(f"Invalid model_name passed - {task}/{model_name}")

    return cached_models[task][model_name]
