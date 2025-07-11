import cv2
import dlib
import numpy as np
import pickle
import time

detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

with open("encodings.pkl", "rb") as f:
    known_encodings, known_names = pickle.load(f)

print(f"✅ Loaded {len(known_encodings)} known faces.")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prev_time = 0
fps_target = 60
frame_interval = 1.0 / fps_target

while True:
    current_time = time.time()
    elapsed = current_time - prev_time

    if elapsed < frame_interval:
        time.sleep(frame_interval - elapsed)

    prev_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector(rgb_frame)

    for face in faces:
        shape = shape_predictor(rgb_frame, face)
        face_encoding = np.array(face_rec_model.compute_face_descriptor(rgb_frame, shape))

        distances = np.linalg.norm(known_encodings - face_encoding, axis=1)
        min_distance = np.min(distances)
        best_match_index = np.argmin(distances)

        tolerance = 0.6
        is_known = min_distance < tolerance
        name = known_names[best_match_index] if is_known else "Unknown"

      
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        color = (0, 255, 0) if is_known else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, name, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

   
    cv2.imshow("Face Recognition", frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
