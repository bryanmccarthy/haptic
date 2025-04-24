import sys
import numpy as np
import cv2
import mediapipe as mp
from keypoint_classifier import KeypointClassifier

def extract_hand_keypoints(frame, mp_hands):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    processed_frame = frame.copy()
    
    results = mp_hands.process(image_rgb)
    
    hand_keypoints = np.zeros(63)
    
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0] # Use first detected hand
        
        for i, landmark in enumerate(hand_landmarks.landmark):
            hand_keypoints[i] = landmark.x
            hand_keypoints[i + 21] = landmark.y
            hand_keypoints[i + 42] = landmark.z
            
        mp.solutions.drawing_utils.draw_landmarks(
            processed_frame,
            hand_landmarks, 
            mp.solutions.hands.HAND_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
            mp.solutions.drawing_styles.get_default_hand_connections_style()
        )
    
    return hand_keypoints, processed_frame

def main():
    model_path = "saved_landmarks/keypoint_classifier.h5"
    labels_path = "saved_landmarks/keypoint_classifier_labels.txt"
    
    classifier = KeypointClassifier(model_path=model_path, label_path=labels_path)
    
    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        sys.exit(1)
    
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read frame")
            break
        
        keypoints, processed_frame = extract_hand_keypoints(frame, mp_hands)
        
        if np.any(keypoints):
            try:
                prediction, confidence = classifier.predict(keypoints)
                
                cv2.putText(
                    processed_frame,
                    f"{prediction} ({confidence:.2f})",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
            except Exception as e:
                print(f"Error during prediction: {e}")
                cv2.putText(
                    processed_frame,
                    f"Error: {str(e)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )
        else:
            cv2.putText(
                processed_frame,
                "No hand detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )
        
        cv2.imshow("Keypoint Classifier", processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    mp_hands.close()

if __name__ == "__main__":
    main()
