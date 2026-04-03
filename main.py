import cv2
import numpy as np
import onnxruntime as ort
import os
import urllib.request

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PICTURES_DIR = os.path.join(BASE_DIR, "Pictures")
MODEL_PATH   = os.path.join(BASE_DIR, "emotion-ferplus-8.onnx")
MODEL_URL    = (
    "https://github.com/onnx/models/raw/main/validated/"
    "vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"
)

# emotion-ferplus outputs scores in this order
EMOTIONS = ["neutral", "happiness", "surprise", "sadness",
            "anger",   "disgust",   "fear",     "contempt"]

# ── Emotion → hamster image ────────────────────────────────────────────────────
EMOTION_IMAGES = {
    "neutral":      "Neutral.jpg",
    "happiness":    "Happy.jpg",
    "surprise":     "interested(seeing a hidden gem).jpg",
    "sadness":      "Neutral.jpg",     # closest remaining
    "anger":        "angry.jpg",
    "disgust":      "angry.jpg",       # closest to disgust
    "fear":         "Neutral.jpg",     # closest remaining
    "contempt":     "Neutral.jpg",     # closest to contempt
    "eyes_closed":  "dead (closing eyes).jpg",
    "none":         "Serious.jpg",
}

HAMSTER_SIZE = 480


# ── Helpers ────────────────────────────────────────────────────────────────────

def download_model():
    print("Downloading emotion model (first run only)…")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded!")


def load_images():
    images = {}
    for emotion, filename in EMOTION_IMAGES.items():
        path = os.path.join(PICTURES_DIR, filename)
        img = cv2.imread(path)
        if img is not None:
            images[emotion] = img
        else:
            print(f"[WARNING] Could not load image for '{emotion}': {path}")
    return images


def make_hamster_panel(img, size=HAMSTER_SIZE):
    panel = np.ones((size, size, 3), dtype=np.uint8) * 255
    h, w = img.shape[:2]
    scale = min(size / w, size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h))
    x_off = (size - new_w) // 2
    y_off = (size - new_h) // 2
    panel[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return panel


def draw_label(frame, text, color=(0, 220, 0)):
    font = cv2.FONT_HERSHEY_DUPLEX
    scale, thickness = 1.2, 2
    (tw, _), _ = cv2.getTextSize(text, font, scale, thickness)
    x = (frame.shape[1] - tw) // 2
    cv2.putText(frame, text, (x + 2, 47), font, scale, (0, 0, 0), thickness + 2)
    cv2.putText(frame, text, (x,     45), font, scale, color,      thickness)


def preprocess_face(face_gray):
    """Resize face ROI to 64x64 float32 tensor expected by emotion-ferplus."""
    face = cv2.resize(face_gray, (64, 64)).astype(np.float32)
    return face.reshape(1, 1, 64, 64)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(MODEL_PATH):
        download_model()

    print("Loading models…")
    session   = ort.InferenceSession(MODEL_PATH)
    input_name = session.get_inputs()[0].name

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye.xml"
    )

    images = load_images()

    cap = None
    for index in range(3):
        for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
            c = cv2.VideoCapture(index, backend)
            if c.isOpened():
                cap = c
                print(f"Webcam found at index {index}")
                break
            c.release()
        if cap is not None:
            break
    if cap is None:
        print("ERROR: No webcam found. Make sure it's connected and not in use.")
        return

    current_emotion = "none"
    frame_skip = 0
    eyes_closed_counter = 0
    no_face_counter = 0
    EYES_CLOSED_THRESHOLD = 15
    NO_FACE_THRESHOLD = 4  # frames before switching to "none"

    print("Face Detecter running!  Press  Q  to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── Detect every 3 frames for performance ─────────────────────────────
        frame_skip += 1
        if frame_skip >= 3:
            frame_skip = 0
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05,
                                                  minNeighbors=3, minSize=(60, 60))
            if len(faces) > 0:
                no_face_counter = 0
                x, y, w, h = faces[0]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 220, 0), 2)

                face_roi = gray[y:y + h, x:x + w]

                # Check if eyes are closed (eye cascade finds open eyes only)
                eye_region = face_roi[: h // 2, :]
                eyes = eye_cascade.detectMultiScale(eye_region, scaleFactor=1.1,
                                                    minNeighbors=10, minSize=(25, 25))
                if len(eyes) == 0:
                    eyes_closed_counter += 1
                    if eyes_closed_counter >= EYES_CLOSED_THRESHOLD:
                        current_emotion = "eyes_closed"
                else:
                    tensor = preprocess_face(face_roi)
                    scores = session.run(None, {input_name: tensor})[0][0]
                    detected = EMOTIONS[int(np.argmax(scores))]
                    if detected != "neutral":
                        eyes_closed_counter = 0
                    current_emotion = detected
            else:
                no_face_counter += 1
                if no_face_counter >= NO_FACE_THRESHOLD:
                    current_emotion = "none"

        # ── Build side-by-side display ─────────────────────────────────────────
        hamster_panel = make_hamster_panel(
            images.get(current_emotion, images["neutral"])
        )

        cam_h, cam_w = frame.shape[:2]
        cam_resized  = cv2.resize(frame, (int(cam_w * HAMSTER_SIZE / cam_h), HAMSTER_SIZE))

        label_map = {"none": "WHERE ARE YOU??", "eyes_closed": "EYES CLOSED"}
        label = label_map.get(current_emotion, current_emotion.upper())
        draw_label(cam_resized, label)

        cv2.imshow("Face Detecter", np.hstack([cam_resized, hamster_panel]))

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
