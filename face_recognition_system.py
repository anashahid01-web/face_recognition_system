"""
Face Recognition System
=======================
Encode known faces from a folder, then recognize them in images or in a
live webcam stream. Built on `face_recognition` (dlib) and OpenCV.

Usage (from a terminal):
    python face_recognition_system.py check
    python face_recognition_system.py encode
    python face_recognition_system.py image path/to/photo.jpg
    python face_recognition_system.py webcam
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

try:
    import face_recognition
except ImportError:
    sys.exit("face_recognition is not installed. See the Installation section of the README.")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_KNOWN_DIR = "known_faces"
DEFAULT_ENCODINGS = "encodings.npz"
DEFAULT_TOLERANCE = 0.5   # lower = stricter (library default is 0.6)

ENV_HELP = (
    "\nThis error usually means dlib and NumPy do not match (for example NumPy 2.x with an "
    "older dlib build).\nFix, option A:  pip uninstall -y dlib  &&  pip install dlib-bin"
    "\nFix, option B:  pip install \"numpy<2\" \"opencv-python<4.11\"  and reinstall dlib"
)


# ----------------------------------------------------------------------------
# Image loading
# ----------------------------------------------------------------------------
def load_image_rgb(path):
    """Load any image as a contiguous 8-bit RGB array (the format dlib needs).
    EXIF rotation is applied so phone photos are not sideways."""
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return np.ascontiguousarray(np.array(img), dtype=np.uint8)


def iter_known_images(folder):
    """Yield (person_name, image_path).
    Supports  known_faces/alice.jpg  and  known_faces/alice/photo1.jpg."""
    folder = Path(folder)
    for p in sorted(folder.rglob("*")):
        if (p.is_file() and p.suffix.lower() in IMAGE_EXTS
                and ".ipynb_checkpoints" not in p.parts):
            name = p.parent.name if p.parent != folder else p.stem
            yield name, p


# ----------------------------------------------------------------------------
# Encoding known faces
# ----------------------------------------------------------------------------
def encode_known_faces(folder=DEFAULT_KNOWN_DIR, out_file=DEFAULT_ENCODINGS):
    """Encode every face image in `folder` and save to `out_file` (.npz).
    Returns (encodings, names)."""
    folder = Path(folder)
    if not folder.exists():
        folder.mkdir(parents=True)
        print(f"[INFO] Created '{folder}'. Add face photos there and run again.")
        return [], []

    encodings, names = [], []
    paths = list(iter_known_images(folder))
    if not paths:
        print(f"[WARNING] No images found in '{folder}'. Supported: {sorted(IMAGE_EXTS)}")
        return [], []

    for name, path in paths:
        try:
            rgb = load_image_rgb(path)
            locations = face_recognition.face_locations(rgb)
            if not locations:
                print(f"[SKIP] No face found in {path.name}")
                continue
            if len(locations) > 1:
                print(f"[WARN] {len(locations)} faces in {path.name}; using the largest one")
                locations = [max(locations, key=lambda b: (b[2] - b[0]) * (b[1] - b[3]))]
            enc = face_recognition.face_encodings(rgb, locations)[0]
            encodings.append(enc)
            names.append(name)
            print(f"[OK]   {path.name} -> '{name}'")
        except Exception as e:
            print(f"[ERROR] {path.name}: {e}")
            if "Unsupported image type" in str(e):
                print(ENV_HELP)
                break

    if encodings:
        # np.savez with plain arrays: no pickle, so loading is safe
        np.savez(out_file, encodings=np.array(encodings), names=np.array(names))
        print(f"\nSaved {len(encodings)} encoding(s) for {len(set(names))} person(s) to '{out_file}'")
    else:
        print("\n[WARNING] No faces were encoded.")
    return encodings, names


def load_encodings(path=DEFAULT_ENCODINGS):
    """Load saved encodings. Returns (list_of_encodings, list_of_names)."""
    try:
        data = np.load(path, allow_pickle=False)
        return list(data["encodings"]), [str(n) for n in data["names"]]
    except FileNotFoundError:
        print(f"[ERROR] '{path}' not found. Run the encode step first.")
        return [], []


# ----------------------------------------------------------------------------
# Matching, drawing
# ----------------------------------------------------------------------------
def match_face(encoding, known_encodings, known_names, tolerance=DEFAULT_TOLERANCE):
    """Return (name, distance). Distance is the Euclidean distance between the
    two 128-d encodings: lower means more similar."""
    if not known_encodings:
        return "Unknown", None
    distances = face_recognition.face_distance(known_encodings, encoding)
    best = int(np.argmin(distances))
    name = known_names[best] if distances[best] <= tolerance else "Unknown"
    return name, float(distances[best])


def detect_and_match(frame_bgr, known_encodings, known_names,
                     tolerance=DEFAULT_TOLERANCE, scale=1.0):
    """Find faces in a BGR frame and match them.
    Returns a list of dicts: {name, distance, box=(top, right, bottom, left)}
    with the box in the coordinates of the original frame."""
    small = cv2.resize(frame_bgr, (0, 0), fx=scale, fy=scale) if scale != 1.0 else frame_bgr
    rgb = np.ascontiguousarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
    locations = face_recognition.face_locations(rgb, model="hog")
    encodings = face_recognition.face_encodings(rgb, locations)
    results = []
    for box, enc in zip(locations, encodings):
        name, dist = match_face(enc, known_encodings, known_names, tolerance)
        top, right, bottom, left = [int(v / scale) for v in box]
        results.append({"name": name, "distance": dist, "box": (top, right, bottom, left)})
    return results


def draw_results(frame_bgr, results):
    """Draw boxes and labels on the frame (in place) and return it."""
    for r in results:
        top, right, bottom, left = r["box"]
        known = r["name"] != "Unknown"
        color = (0, 180, 0) if known else (0, 0, 255)          # BGR: green / red
        label = r["name"] if r["distance"] is None else f"{r['name']} ({r['distance']:.2f})"
        cv2.rectangle(frame_bgr, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame_bgr, (left, bottom), (right, bottom + 26), color, cv2.FILLED)
        cv2.putText(frame_bgr, label, (left + 6, bottom + 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return frame_bgr


# ----------------------------------------------------------------------------
# Recognition: image and webcam
# ----------------------------------------------------------------------------
def recognize_image(image_path, encodings_file=DEFAULT_ENCODINGS,
                    tolerance=DEFAULT_TOLERANCE, output_path="output/recognized.jpg"):
    """Recognize faces in one image and save an annotated copy."""
    known_encodings, known_names = load_encodings(encodings_file)
    if not known_encodings:
        return []
    rgb = load_image_rgb(image_path)
    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    scale = min(1.0, 1000 / max(frame.shape[:2]))      # speed up very large photos
    results = detect_and_match(frame, known_encodings, known_names, tolerance, scale)

    print(f"Found {len(results)} face(s) in {Path(image_path).name}")
    for i, r in enumerate(results, 1):
        d = "n/a" if r["distance"] is None else f"{r['distance']:.2f}"
        print(f"  Face {i}: {r['name']} (distance {d})")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), draw_results(frame, results))
    print(f"Annotated image saved to '{output_path}'")
    return results


def recognize_webcam(encodings_file=DEFAULT_ENCODINGS, tolerance=DEFAULT_TOLERANCE,
                     camera_index=0, scale=0.5, process_every=2):
    """Live recognition from the webcam. Press 'q' to quit."""
    known_encodings, known_names = load_encodings(encodings_file)
    if not known_encodings:
        return
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {camera_index}.")
        return
    print("Webcam started. Press 'q' in the video window to quit.")
    results, count = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if count % process_every == 0:                  # detect on every Nth frame only
            results = detect_and_match(frame, known_encodings, known_names, tolerance, scale)
        count += 1
        cv2.imshow("Face Recognition (press q to quit)", draw_results(frame, results))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


# ----------------------------------------------------------------------------
# Environment check
# ----------------------------------------------------------------------------
def check_environment():
    """Print library versions and run a tiny test to catch dlib/NumPy problems early."""
    import dlib
    print(f"Python  : {sys.version.split()[0]}")
    print(f"NumPy   : {np.__version__}")
    print(f"OpenCV  : {cv2.__version__}")
    print(f"dlib    : {dlib.__version__}")
    try:
        face_recognition.face_locations(np.zeros((100, 100, 3), dtype=np.uint8))
        print("Result  : OK, face_recognition works with this setup.")
        return True
    except Exception as e:
        print(f"Result  : FAILED ({e})")
        print(ENV_HELP)
        return False


# ----------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Face recognition system")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="check that dlib, NumPy and OpenCV work together")
    e = sub.add_parser("encode", help="encode faces from the known_faces folder")
    e.add_argument("--folder", default=DEFAULT_KNOWN_DIR)
    e.add_argument("--out", default=DEFAULT_ENCODINGS)
    i = sub.add_parser("image", help="recognize faces in an image")
    i.add_argument("path")
    i.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    i.add_argument("--out", default="output/recognized.jpg")
    w = sub.add_parser("webcam", help="live recognition from the webcam")
    w.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    w.add_argument("--camera", type=int, default=0)
    a = p.parse_args()

    if a.cmd == "check":
        check_environment()
    elif a.cmd == "encode":
        encode_known_faces(a.folder, a.out)
    elif a.cmd == "image":
        recognize_image(a.path, tolerance=a.tolerance, output_path=a.out)
    elif a.cmd == "webcam":
        recognize_webcam(tolerance=a.tolerance, camera_index=a.camera)


if __name__ == "__main__":
    main()
