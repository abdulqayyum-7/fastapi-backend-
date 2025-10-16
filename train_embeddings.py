# backend/train_embeddings.py
import os
import pickle
from pathlib import Path
import face_recognition
import cv2
import numpy as np
from PIL import Image

# HEIC support
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    print("⚠ pillow-heif not installed. Run: pip install pillow-heif")

dataset_dir = Path("./dataset")  # run from backend/
output_file = "model.pkl"

known_encodings = []
known_names = []


def load_image_as_rgb(path):
    """Load any image safely and ensure contiguous 8-bit RGB (Windows fix)."""
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")  # Force RGB
            np_img = np.array(img, dtype=np.uint8)

            # ✅ Ensure contiguous memory layout (important for Windows + dlib)
            np_img = np.ascontiguousarray(np_img)

        return np_img
    except Exception as e:
        print(f"⚠️ Failed to load {path}: {e}")
        return None


# Loop through each person
for person_dir in sorted(dataset_dir.iterdir()):
    if not person_dir.is_dir():
        continue

    name = person_dir.name

    for img_path in person_dir.glob("*"):
        try:
            # Convert HEIC to JPG if needed
            if img_path.suffix.lower() == ".heic":
                jpg_path = img_path.with_suffix(".jpg")
                if not jpg_path.exists():
                    image = Image.open(img_path)
                    image = image.convert("RGB")
                    image.save(jpg_path, "JPEG", quality=95)
                    print(f"🖼 Converted {img_path.name} → {jpg_path.name}")
                img_path = jpg_path

            # Load image safely
            image = load_image_as_rgb(img_path)
            if image is None:
                continue

            # Ensure correct dtype & color
            if image.dtype != np.uint8:
                image = image.astype(np.uint8)
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

            # ✅ Detect faces
            face_locations = face_recognition.face_locations(image, model="hog")
            if len(face_locations) == 0:
                print(f"No faces found in {img_path}, skipping.")
                continue

            # ✅ Encode using face_recognition
            encs = face_recognition.face_encodings(image, known_face_locations=face_locations)
            if len(encs) == 0:
                print(f"No encodings from {img_path}, skipping.")
                continue

            known_encodings.append(encs[0])
            known_names.append(name)
            print(f"✅ Encoded {img_path}")

        except Exception as e:
            print(f"❌ Error processing {img_path}: {e}")

# Save results
data = {"encodings": known_encodings, "names": known_names}
with open(output_file, "wb") as f:
    pickle.dump(data, f)

print(f"\n✅ Saved {len(known_encodings)} encodings to {output_file}")
