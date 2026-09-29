import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

DATASET_FILE = "dataset/asl_landmarks.csv"
MODEL_FILE = "model/asl_model.pkl"


def main():
    print()
    print("======================================")
    print("        ASL MODEL TRAINING")
    print("======================================")
    print()

    print("Loading ASL landmark data...")

    data = pd.read_csv(DATASET_FILE)

    print("Samples:", len(data))
    print("Features:", len(data.columns) - 1)

    # Remove classes that are not needed for the A-Z alphabet model
    data = data[data["label"].isin(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))]

    print()
    print("Using A-Z alphabet classes only.")
    print("Samples after filtering:", len(data))

    X = data.drop("label", axis=1)
    y = data["label"]

    print()
    print("ASL classes:")
    print(sorted(y.unique()))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print()
    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    print()
    print("Training Random Forest model...")
    print("Please wait...")
    print()

    model = RandomForestClassifier(
    n_estimators=40,
    random_state=42,
    n_jobs=1,
    max_depth=15
)

    model.fit(X_train, y_train)

    print("Training completed!")

    print()
    print("Testing model...")

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print()
    print("======================================")
    print("MODEL RESULT")
    print("======================================")
    print()

    print(f"Accuracy: {accuracy * 100:.2f}%")

    print()
    print("Classification Report:")
    print(classification_report(y_test, predictions))

    joblib.dump(model, MODEL_FILE)

    print()
    print("======================================")
    print("Model saved successfully!")
    print("======================================")
    print()
    print("Saved to:")
    print(MODEL_FILE)
    print()


if __name__ == "__main__":
    main()

