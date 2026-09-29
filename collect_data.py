import csv
import os

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = "model/hand_landmarker.task"
DATASET_PATH = "dataset/hand_landmarks.csv"

LABELS = {
    "h": "hello",
    "y": "yes",
    "n": "no",
}


def get_landmarks(result):
    """Get landmarks for up to two hands."""

    values = []

    for hand_index in range(2):
        if hand_index < len(result.hand_landmarks):
            hand = result.hand_landmarks[hand_index]

            for landmark in hand:
                values.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z
                ])
        else:
            # If only one hand is detected,
            # fill the second hand with zeros.
            values.extend([0.0] * 63)

    return values


def draw_hand(frame, hand):
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


def main():

    os.makedirs("dataset", exist_ok=True)

    file_exists = os.path.exists(DATASET_PATH)

    csv_file = open(
        DATASET_PATH,
        "w",
        newline=""
    )

    writer = csv.writer(csv_file)

    # Header
    header = ["label"]

    for hand_number in range(2):
        for i in range(21):
            header.extend([
                f"hand{hand_number + 1}_x{i}",
                f"hand{hand_number + 1}_y{i}",
                f"hand{hand_number + 1}_z{i}"
            ])

    writer.writerow(header)

    base_options = python.BaseOptions(
        model_asset_path=MODEL_PATH
    )

    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    detector = vision.HandLandmarker.create_from_options(
        options
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open webcam.")
        csv_file.close()
        return

    print()
    print("======================================")
    print("     TWO-HAND SIGN LANGUAGE DATA")
    print("======================================")
    print()
    print("H = Hello")
    print("Y = Yes")
    print("N = No")
    print("Q = Quit")
    print()
    print("Show BOTH hands when the sign requires them.")
    print("Press the corresponding key to save a sample.")
    print()

    while True:

        success, frame = cap.read()

        if not success:
            print("Could not read webcam frame.")
            break

        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        result = detector.detect(mp_image)

        # Draw all detected hands
        for hand in result.hand_landmarks:
            draw_hand(frame, hand)

        # Show number of hands detected
        hand_count = len(result.hand_landmarks)

        cv2.putText(
            frame,
            f"Hands detected: {hand_count}/2",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "H=Hello  Y=Yes  N=No  Q=Quit",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "Two-Hand Sign Language Data Collector",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key in [ord("h"), ord("y"), ord("n")]:

            if len(result.hand_landmarks) == 2:

                landmarks = get_landmarks(result)

                writer.writerow([
                    LABELS[chr(key)],
                    *landmarks
                ])

                csv_file.flush()

                print(
                    f"Saved TWO-HAND sample: "
                    f"{LABELS[chr(key)]}"
                )

            else:

                print(
                    "Please show BOTH hands. "
                    "Sample was not saved."
                )

    cap.release()
    cv2.destroyAllWindows()

    detector.close()
    csv_file.close()

    print()
    print("Data collection finished.")
    print(f"Dataset saved to: {DATASET_PATH}")


if __name__ == "__main__":
    main()