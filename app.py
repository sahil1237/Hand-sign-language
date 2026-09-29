import cv2
import mediapipe as mp
import joblib
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


HAND_MODEL = "model/hand_landmarker.task"
ASL_MODEL = "model/asl_model.pkl"


def main():

    print("Loading ASL model...")

    # Load trained ASL model
    model = joblib.load(ASL_MODEL)

    print("ASL model loaded successfully!")
    print("Classes:", model.classes_)

    # Load MediaPipe hand model
    base_options = python.BaseOptions(
        model_asset_path=HAND_MODEL
    )

    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    detector = vision.HandLandmarker.create_from_options(options)

    # Start webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open webcam.")
        return

    print("Webcam started!")
    print("Show an ASL letter.")
    print("Press Q to quit.")

    while True:

        success, frame = cap.read()

        if not success:
            print("Could not read frame.")
            break

        # Mirror webcam
        frame = cv2.flip(frame, 1)

        # Convert image to RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Create MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Detect hand
        result = detector.detect(mp_image)

        predicted_letter = None
        confidence = 0.0

        if result.hand_landmarks:

            # Get first detected hand
            hand = result.hand_landmarks[0]

            # Extract 21 landmarks
            features = []

            for landmark in hand:

                features.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z
                ])

            # Convert to numpy array
            features = np.array(features).reshape(1, -1)
            features = np.array(features).reshape(1, -1)

feature_names = []
for i in range(21):
    feature_names.extend([f"x{i}", f"y{i}", f"z{i}"])

features_df = pd.DataFrame(features, columns=feature_names)

predicted_letter = model.predict(features_df)[0]
probabilities = model.predict_proba(features_df)[0]
confidence = np.max(probabilities) * 100

            # Draw hand connections
connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),
                (0, 5), (5, 6), (6, 7), (7, 8),
                (0, 9), (9, 10), (10, 11), (11, 12),
                (0, 13), (13, 14), (14, 15), (15, 16),
                (0, 17), (17, 18), (18, 19), (19, 20),
                (5, 9), (9, 13), (13, 17)
            ]

for start, end in connections:

                x1 = int(hand[start].x * frame.shape[1])
                y1 = int(hand[start].y * frame.shape[0])

                x2 = int(hand[end].x * frame.shape[1])
                y2 = int(hand[end].y * frame.shape[0])

                cv2.line(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

            # Draw landmarks
for landmark in hand:

                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 0, 255),
                    -1
                )

        # Title
cv2.putText(
            frame,
            "ASL Alphabet Translator",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        # Display prediction
if predicted_letter is not None:

            cv2.putText(
                frame,
                f"Sign: {predicted_letter}",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.3,
                (0, 255, 255),
                3
            )

            cv2.putText(
                frame,
                f"Confidence: {confidence:.1f}%",
                (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        else:

            cv2.putText(
                frame,
                "Show your hand",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2
            )

        # Instructions
cv2.putText(
            frame,
            "Press Q to quit",
            (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # Show webcam
cv2.imshow(
            "ASL Alphabet Translator",
            frame
        )

        # Quit
if cv2.waitKey(1) & 0xFF == ord("q"):
         break

cap.release()
cv2.destroyAllWindows()
detector.close()


if __name__ == "__main__":
    main()

