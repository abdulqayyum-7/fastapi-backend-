import pickle
import face_recognition
import numpy as np

# Load model.pkl
with open("model.pkl", "rb") as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
known_names = data["names"]

def recognize_face(image):
    encs = face_recognition.face_encodings(image)
    if len(encs) == 0:
        return "No face detected"
    
    face_encoding = encs[0]
    matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
    face_distances = face_recognition.face_distance(known_encodings, face_encoding)

    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
        return known_names[best_match_index]
    else:
        return "Unknown"
