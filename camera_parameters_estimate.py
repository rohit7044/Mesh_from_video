# camera_parameters_estimate.py

import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"  # For fast loading

import cv2
from mediapipe_landmark import FaceLandmarkerDetector

def main():
    cap = cv2.VideoCapture(0)
    detector = FaceLandmarkerDetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Flip to mirror view
        detector.detect_async(frame)

        # Get latest landmarks
        result = detector.get_latest_result()
        if result and result.face_landmarks:
            for i, landmark in enumerate(result.face_landmarks[0]):
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        cv2.imshow("Face Landmarks", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    detector.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
