import os
import csv
import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


ASL_DATASET = "asl_data/asl_alphabet_train/asl_alphabet_train"
OUTPUT_FILE = "dataset/asl_landmarks_normalized.csv"
MODEL_PATH = "model/hand_landmarker.task"


def normalize_landmarks(hand):
    """
    Make the hand landmarks independent of:
    - hand position
    - hand size
    """

    # Use wrist as the starting point
    wrist = hand[0]

    points = []

    for landmark in hand:
        x = landmark.x - wrist.x
        y = landmark.y - wrist.y
        z = landmark.z - wrist.z

        points.append([x, y, z])

    # Find the largest distance from wrist
    max_distance = 0

    for point in points:
        distance = (
            point[0] ** 2 +
            point[1] ** 2 +
            point[2] ** 2
        ) ** 0.5

        if distance > max_distance:
            max_distance = distance

    # Avoid division by zero
    if max_distance == 0:
        return None

    # Scale the hand to a similar size
    normalized = []

    for point in points:
        normalized.extend([
            point[0] / max_distance,
            point[1] / max_distance,
            point[2] / max_distance
        ])

    return normalized


def main():

    print("Loading MediaPipe hand model...")

    base_options = python.BaseOptions(
        model_asset_path=MODEL_PATH
    )

    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    detector = vision.HandLandmarker.create_from_options(options)

    os.makedirs("dataset", exist_ok=True)

    header = ["label"]

    for i in range(21):
        header.append(f"x{i}")
        header.append(f"y{i}")
        header.append(f"z{i}")

    total_images = 0
    total_landmarks = 0

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.writer(csv_file)
        writer.writerow(header)

        for label in sorted(os.listdir(ASL_DATASET)):

            label_path = os.path.join(
                ASL_DATASET,
                label
            )

            if not os.path.isdir(label_path):
                continue

            # Only A-Z for now
            if label not in list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
                continue

            print(f"Processing {label}...")

            for filename in os.listdir(label_path):

                image_path = os.path.join(
                    label_path,
                    filename
                )

                image = cv2.imread(image_path)

                if image is None:
                    continue

                total_images += 1

                rgb = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2RGB
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )

                result = detector.detect(mp_image)

                if not result.hand_landmarks:
                    continue

                hand = result.hand_landmarks[0]

                normalized = normalize_landmarks(hand)

                if normalized is None:
                    continue

                if len(normalized) != 63:
                    continue

                writer.writerow(
                    [label] + normalized
                )

                total_landmarks += 1

    detector.close()

    print()
    print("===================================")
    print("ASL EXTRACTION COMPLETE")
    print("===================================")
    print(f"Images processed: {total_images}")
    print(f"Landmarks extracted: {total_landmarks}")
    print(f"Output file: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

