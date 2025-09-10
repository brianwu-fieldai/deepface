
"""
One-file CLI showcasing how to call DeepFace:
  • verify         — are two faces the same person?
  • find           — who does this face match in a folder DB?
  • analyze        — age, gender, emotion, race
  • represent      — get raw embedding vectors
  • extract-faces  — crop faces (optionally anti-spoofing)
  • stream         — webcam demo (optional DB for recognition)
  • encrypt        — homomorphic cosine similarity (PHE)

Examples:
  python deepface_demo.py verify --img1 img1.jpg --img2 img2.jpg --model ArcFace
  python deepface_demo.py find --img query.jpg --db ./my_db --detector retinaface
  python deepface_demo.py analyze --img img.jpg --actions age gender emotion race
  python deepface_demo.py represent --img img.jpg --model Facenet512 --json out.json
  python deepface_demo.py extract-faces --img img.jpg --anti-spoofing --save-dir ./faces
  python deepface_demo.py stream --db ./my_db --anti-spoofing
  python deepface_demo.py encrypt --img1 img1.jpg --img2 img2.jpg --model ArcFace --json out.json

Notes:
  - Install dependencies:  pip install deepface
  - Directory layout for DB:
        my_db/
          Alice/ Alice1.jpg Alice2.jpg (support multiple image files a person)
          Bob/   Bob.jpg
"""

import argparse
import sys
import json
from pathlib import Path
from typing import List, Optional, Dict, Any

def _lazy_import():
    try:
        from deepface import DeepFace
        return DeepFace
    except Exception as e:
        sys.stderr.write(
            "DeepFace import failed. Try: pip install deepface\n"
        )
        raise

# -------------------------
# Utilities
# -------------------------

def _write_json(payload: Dict[str, Any], path: Optional[str]) -> None:
    if not path or not path.endswith(".json"):
        raise ValueError("Output path for writing a JSON file must end with .json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

def _print_h1(title: str) -> None:
    print(f"\n=== {title} ===")

def _exit(code: int, msg: Optional[str] = None) -> None:
    if msg:
        (sys.stderr if code else sys.stdout).write(msg + "\n")
    sys.exit(code)

# -------------------------
# Commands
# -------------------------

def cmd_verify(args: argparse.Namespace) -> None:
    DeepFace = _lazy_import()
    res = DeepFace.verify(
        img1_path=args.img1,
        img2_path=args.img2,
        model_name=args.model,
        detector_backend=args.detector,
        align=not args.no_align,
        distance_metric=args.metric,
        enforce_detection=not args.no_enforce_detection,
    )
    _print_h1("Verification Result")
    print(f"verified={res.get('verified')} distance={res.get('distance')} "
          f"threshold={res.get('threshold')} model={res.get('model')} "
          f"detector={res.get('detector_backend')} metric={args.metric}")
    _write_json(res, args.json)

def cmd_find(args: argparse.Namespace) -> None:
    DeepFace = _lazy_import()
    dfs = DeepFace.find(
        img_path=args.img,
        db_path=args.db,
        model_name=args.model,
        detector_backend=args.detector,
        align=not args.no_align,
        distance_metric=args.metric,
        enforce_detection=not args.no_enforce_detection,
        silent=not args.verbose,
    )
    # dfs is a list of DataFrames (one per detected face). Summarize to JSON.
    out: Dict[str, Any] = {"matches": []}
    try:
        import pandas as pd  # type: ignore
    except Exception:
        pd = None

    for i, df in enumerate(dfs):
        if df is None:
            continue
        if pd is not None:
            # Show top-5
            top = df.head(5)[["identity", "distance"]]
            _print_h1(f"Top matches for face #{i+1}")
            print(top.to_string(index=False))
            out["matches"].append(top.to_dict(orient="records"))
        else:
            # Fallback print
            rows = df.head(5)[["identity", "distance"]].to_dict(orient="records")
            _print_h1(f"Top matches for face #{i+1}")
            for r in rows:
                print(f"{r['identity']}\t{r['distance']:.4f}")
            out["matches"].append(rows)

    _write_json(out, args.json)

def cmd_analyze(args: argparse.Namespace) -> None:
    DeepFace = _lazy_import()
    actions = args.actions or ["age", "gender", "emotion", "race"]
    res = DeepFace.analyze(
        img_path=args.img,
        actions=actions,
        detector_backend=args.detector,
        align=not args.no_align,
        enforce_detection=not args.no_enforce_detection,
    )
    # res can be a list (multiple faces) or dict (single face)
    payload = {"faces": res if isinstance(res, list) else [res]}
    _print_h1("Analysis Result")
    for idx, face in enumerate(payload["faces"]):
        dom_emotion = None
        emo = face.get("emotion")
        if isinstance(emo, dict):
            dom_emotion = max(emo, key=emo.get)
        age = face.get("age")
        gender = face.get("dominant_gender") or face.get("gender")
        race = face.get("dominant_race") or face.get("race")
        print(f"Face #{idx+1}: age≈{age} gender={gender} emotion={dom_emotion} race={race}")
    _write_json(payload, args.json)

def cmd_represent(args: argparse.Namespace) -> None:
    DeepFace = _lazy_import()
    res = DeepFace.represent(
        img_path=args.img,
        model_name=args.model,
        detector_backend=args.detector,
        align=not args.no_align,
        enforce_detection=not args.no_enforce_detection,
    )
    payload = {"embeddings": res}
    _print_h1("Embeddings")
    for i, obj in enumerate(res):
        emb = obj.get("embedding", [])
        print(f"Face #{i+1}: dim={len(emb)} model={obj.get('model')} "
              f"detector={obj.get('detector_backend')}")
    _write_json(payload, args.json)

def cmd_extract_faces(args: argparse.Namespace) -> None:
    DeepFace = _lazy_import()
    faces = DeepFace.extract_faces(
        img_path=args.img,
        detector_backend=args.detector,
        align=not args.no_align,
        enforce_detection=not args.no_enforce_detection,
        anti_spoofing=args.anti_spoofing,
    )
    _print_h1("Extracted Faces")
    payload = {"faces": []}
    save_dir: Optional[Path] = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
    for i, fobj in enumerate(faces):
        box = fobj.get("facial_area") or {}
        is_real = fobj.get("is_real")
        print(f"Face #{i+1}: box={box} is_real={is_real}")
        if save_dir is not None:
            # Save cropped image array via OpenCV
            try:
                import cv2  # type: ignore
                crop = fobj.get("face")
                if crop is not None:
                    out_path = save_dir / f"face_{i+1}.png"
                    cv2.imwrite(str(out_path), (crop * 255).astype("uint8"))
            except Exception:
                pass
        payload["faces"].append({"facial_area": box, "is_real": is_real})
    _write_json(payload, args.json)

def cmd_stream(args: argparse.Namespace) -> None:
    DeepFace = _lazy_import()
    DeepFace.stream(
        db_path=args.db,
        detector_backend=args.detector,
        align=not args.no_align,
        anti_spoofing=args.anti_spoofing,
    )
    
def _l2_normalize(vec):
    import math
    n = math.sqrt(sum((x*x) for x in vec)) or 1.0
    return [x / n for x in vec]

def cmd_encrypt(args: argparse.Namespace) -> None:
    """
    Homomorphic cosine similarity: encrypt the source embedding and compute
    the dot product with a *plain* target embedding in untrusted infra.
    Finally decrypt locally and compare against DeepFace's cosine threshold.
    """
    DeepFace = _lazy_import()
    try:
        from lightphe import LightPHE
    except Exception as e:
        _exit(1, "Missing dependency 'lightphe'. Install it with: pip install lightphe")

    src_objs = DeepFace.represent(
        img_path=args.img1,
        model_name=args.model,
        detector_backend=args.detector,
        align=not args.no_align,
        enforce_detection=not args.no_enforce_detection,
    )
    tgt_objs = DeepFace.represent(
        img_path=args.img2,
        model_name=args.model,
        detector_backend=args.detector,
        align=not args.no_align,
        enforce_detection=not args.no_enforce_detection,
    )

    if not src_objs or not tgt_objs:
        _exit(1, "Could not extract embeddings for one or both images.")

    alpha = src_objs[0].get("embedding", [])
    beta  = tgt_objs[0].get("embedding", [])
    if not alpha or not beta or len(alpha) != len(beta):
        _exit(1, "Embeddings missing or have mismatched dimensions.")

    alpha_n = _l2_normalize(alpha)
    beta_n  = _l2_normalize(beta)

    cs = LightPHE(algorithm_name="Paillier", precision=19)
    enc_alpha = cs.encrypt(alpha_n)
    enc_cos_sim = enc_alpha @ beta_n
    cos_sim = cs.decrypt(enc_cos_sim)[0]
    
    verify_res = DeepFace.verify(
        img1_path=args.img1,
        img2_path=args.img2,
        model_name=args.model,
        detector_backend=args.detector,
        align=not args.no_align,
        distance_metric="cosine",
        enforce_detection=not args.no_enforce_detection,
    )

    cos_dist_threshold = verify_res.get("threshold", 0.3)
    decision = cos_sim >= (1.0 - cos_dist_threshold)

    _print_h1("Encrypted Embedding Similarity (Paillier PHE)")
    print(f"model={args.model or verify_res.get('model')}, "
          f"detector={args.detector or verify_res.get('detector_backend')}")
    print(f"cosine_similarity≈{cos_sim:.6f}  (threshold for match: ≥ {1.0 - cos_dist_threshold:.6f})")
    print(f"decision={'SAME PERSON' if decision else 'DIFFERENT PERSON'}")

    payload = {
        "model": args.model or verify_res.get("model"),
        "detector": args.detector or verify_res.get("detector_backend"),
        "cosine_similarity": float(cos_sim),
        "cosine_similarity_threshold": float(1.0 - cos_dist_threshold),
        "decision": bool(decision),
        "encryption": {
            "scheme": "Paillier (PHE)",
            "precision": 19,
            "encrypted_source_dimensions": len(alpha_n),
            "plain_target_dimensions": len(beta_n),
        },
    }
    _write_json(payload, args.json)
    
def _nn(d: dict) -> dict:
    """Return a copy without None-valued keys."""
    return {k: v for k, v in d.items() if v is not None}

def cmd_video(args: argparse.Namespace) -> None:
    """
    Process a video: detect faces per frame, compute embeddings, assign stable IDs,
    draw bounding boxes + IDs, and write an annotated video.
    Prints:
      • total frames processed
      • final active tracks
      • UNIQUE faces detected (distinct IDs)
      • UNIQUE faces out of TOTAL face detection hits across all frames
    """
    # Imports
    DeepFace = _lazy_import()
    try:
        import cv2
    except Exception:
        _exit(1, "OpenCV not found. Install it with: pip install opencv-python")

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        _exit(1, f"Could not open input video: {args.input}")

    in_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    in_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    fps   = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if not fps or fps < 1 or fps > 240:
        fps = args.fps or 30.0

    fourcc = cv2.VideoWriter_fourcc(*("mp4v"))
    writer = cv2.VideoWriter(args.output, fourcc, fps, (in_w, in_h))
    if not writer.isOpened():
        _exit(1, f"Could not create output video: {args.output}")

    # Tracking + counters
    tracks = []     # each: {"id": int, "emb": List[float], "bbox": (x,y,w,h), "last_seen": frame_idx}
    next_id = 1
    unique_ids = set()      # all track IDs ever created (unique people)
    total_face_hits = 0     # raw detector hits across all processed frames
    frame_idx = 0

    def cosine_distance(a, b):
        # both a and b must be L2-normalized
        return 1.0 - sum(x*y for x, y in zip(a, b))

    def represent_face(face_img):
        """
        face_img: numpy array HxWx3 from DeepFace.extract_faces()['face'] (0..1 float).
        Convert to uint8 and skip detection to speed embedding extraction.
        """
        import numpy as np
        if face_img is None:
            return None
        if face_img.dtype != np.uint8:
            face_img = (face_img * 255.0).clip(0, 255).astype("uint8")
        rep = DeepFace.represent(
            img_path=face_img,
            **_nn({"model_name": args.model, "detector_backend": "skip"}),
            align=not args.no_align,
            enforce_detection=False,
        )
        if not rep:
            return None
        emb = rep[0].get("embedding")
        if not emb:
            return None
        return _l2_normalize(emb)

    def draw_track(f, t, color=(0, 255, 0)):
        x, y, w, h = t["bbox"]
        x = max(0, x); y = max(0, y)
        w = max(0, w); h = max(0, h)
        cv2.rectangle(f, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            f, f"ID {t['id']}", (x, max(0, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, lineType=cv2.LINE_AA
        )

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Optionally skip frames to speed up
        do_detect = True
        if args.skip > 0 and (frame_idx % (args.skip + 1) != 0):
            do_detect = False

        if do_detect:
            # Detect faces (and optionally anti-spoofing) on this frame
            faces = DeepFace.extract_faces(
                img_path=frame,
                **_nn({"detector_backend": args.detector}),
                align=True,
                enforce_detection=False,
                anti_spoofing=args.anti_spoofing,
            )

            # Count ALL raw hits returned by the detector this frame
            total_face_hits += len(faces)

            detections = []
            for fobj in faces:
                # Optionally drop spoofs before tracking
                if args.anti_spoofing and (fobj.get("is_real") is False):
                    continue

                area = fobj.get("facial_area") or {}
                x = int(area.get("x", 0)); y = int(area.get("y", 0))
                w = int(area.get("w", 0)); h = int(area.get("h", 0))
                if w <= 0 or h <= 0:
                    continue

                emb = represent_face(fobj.get("face"))
                if emb is None:
                    continue

                detections.append({"bbox": (x, y, w, h), "emb": emb})

            # Match detections to existing tracks
            new_tracks = []
            for det in detections:
                best_t = None
                best_dist = float("inf")
                for t in tracks:
                    dist = cosine_distance(det["emb"], t["emb"])
                    if dist < best_dist:
                        best_dist = dist
                        best_t = t

                if best_t is not None and best_dist <= args.match_threshold:
                    # Update existing track with EMA on embedding
                    old = best_t["emb"]
                    ema = args.ema
                    merged = _l2_normalize([(1.0 - ema) * old[i] + ema * det["emb"][i] for i in range(len(old))])
                    best_t["emb"] = merged
                    best_t["bbox"] = det["bbox"]
                    best_t["last_seen"] = frame_idx
                else:
                    # Create a new track (counts toward UNIQUE faces)
                    t = {"id": next_id, "emb": det["emb"], "bbox": det["bbox"], "last_seen": frame_idx}
                    unique_ids.add(next_id)
                    next_id += 1
                    new_tracks.append(t)

            # Age out stale tracks and keep updated/new ones
            tracks = [t for t in tracks if (frame_idx - t["last_seen"]) <= args.max_age]
            tracks.extend(new_tracks)

        # Draw all current tracks
        for t in tracks:
            draw_track(frame, t)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    _print_h1("Video processing complete")
    print(f"Output saved to: {args.output}")
    print(f"Frames processed: {frame_idx}, final active tracks: {len(tracks)}")
    print(f"Unique faces detected: {len(unique_ids)} out of {total_face_hits} total face detection hits")

# -------------------------
# CLI
# -------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DeepFace demo CLI")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", default=None,
                        help="Model name (e.g., ArcFace, VGG-Face, Facenet512, ...)")
    common.add_argument("--detector", default=None,
                        help="Detector backend (opencv, ssd, dlib, mtcnn, fastmtcnn, retinaface, mediapipe, yolov8, yolov11s, yolov11n, yolov11m, yunet, centerface, yolox)")
    common.add_argument("--metric", default="cosine",
                        choices=["cosine","euclidean","euclidean_l2","angular"],
                        help="Distance metric for similarity (verify/find)")
    common.add_argument("--no-align", action="store_true",
                        help="Disable alignment")
    common.add_argument("--no-enforce-detection", action="store_true",
                        help="Do not enforce face detection (process anyway)")
    common.add_argument("--json", default="/Users/brianwu/Documents/Projects/FieldAI/machine_learning/projects/facial_recognition/deepface/test.json",
                        help="Write raw JSON result to this file")

    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("verify", parents=[common], help="Face verification")
    sp.add_argument("--img1", required=True)
    sp.add_argument("--img2", required=True)
    sp.set_defaults(func=cmd_verify)

    sp = sub.add_parser("find", parents=[common], help="Face recognition / search in DB")
    sp.add_argument("--img", required=True)
    sp.add_argument("--db", required=True)
    sp.add_argument("--verbose", action="store_true")
    sp.set_defaults(func=cmd_find)

    sp = sub.add_parser("analyze", parents=[common], help="Facial attribute analysis")
    sp.add_argument("--img", required=True)
    sp.add_argument("--actions", nargs="*", default=None,
                    help="Subset of: age gender emotion race")
    sp.set_defaults(func=cmd_analyze)

    sp = sub.add_parser("represent", parents=[common], help="Get embeddings")
    sp.add_argument("--img", required=True)
    sp.set_defaults(func=cmd_represent)

    sp = sub.add_parser("extract-faces", parents=[common], help="Crop faces (optionally anti-spoofing)")
    sp.add_argument("--img", required=True)
    sp.add_argument("--save-dir", default=None, help="Directory to save cropped faces as PNGs")
    sp.add_argument("--anti-spoofing", action="store_true")
    sp.set_defaults(func=cmd_extract_faces)

    sp = sub.add_parser("stream", parents=[common], help="Webcam stream demo")
    sp.add_argument("--db", default=None, help="Optional DB for recognition")
    sp.add_argument("--anti-spoofing", action="store_true")
    sp.set_defaults(func=cmd_stream)
    
    sp = sub.add_parser("encrypt", parents=[common],
                        help="Encrypt embeddings and compute homomorphic cosine similarity (PHE)")
    sp.add_argument("--img1", required=True, help="Source image to encrypt")
    sp.add_argument("--img2", required=True, help="Target image (plain)")
    sp.set_defaults(func=cmd_encrypt)
    
    sp = sub.add_parser(
        "video",
        parents=[common],
        help="Process a video, overlay face boxes + track IDs, and write an annotated video"
    )
    sp.add_argument("--input", required=True, help="Input video path")
    sp.add_argument("--output", required=True, help="Output video path (e.g., out.mp4)")
    sp.add_argument("--fps", type=float, default=None, help="Override output FPS (default: from input or 30)")
    sp.add_argument("--skip", type=int, default=0, help="Process every Nth frame (0 = process all)")
    sp.add_argument("--match-threshold", type=float, default=0.30,
                    help="Cosine distance threshold for matching (lower is stricter, typical 0.25–0.40)")
    sp.add_argument("--ema", type=float, default=0.15, help="EMA factor for track embedding updates (0–1)")
    sp.add_argument("--max-age", type=int, default=20, help="Keep unmatched tracks up to N frames")
    sp.add_argument("--anti-spoofing", action="store_true", help="Filter out spoofed faces")
    sp.set_defaults(func=cmd_video)

    return p

def main(argv: Optional[List[str]] = None) -> Optional[int]:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
        return 0
    except KeyboardInterrupt:
        _exit(130, "Interrupted")
    except SystemExit as se:
        # argparse may raise; propagate
        raise se
    except Exception as e:
        _exit(1, f"Error: {e}")

if __name__ == "__main__":
    raise SystemExit(main())
