import requests
import time

BASE_URL = "http://127.0.0.1:8000"
SERIES_ID = "sensor-001"

def test_api():
    """
    Makes requests to the train and predict routes of the API.
    """
    print("--- Testing API Endpoints ---")

    # 1. Train a model
    print(f"\n[1] Training model for series_id: {SERIES_ID}")
    train_data = {
        "timestamps": [int(time.time()) + i for i in range(10)],
        "values": [10.1, 10.2, 9.9, 10.0, 10.3, 9.8, 10.1, 10.2, 10.0, 9.9]
    }
    try:
        train_response = requests.post(f"{BASE_URL}/fit/{SERIES_ID}", json=train_data)
        train_response.raise_for_status()
        print("Train request successful:")
        print(train_response.json())
        model_version = train_response.json().get("version")
    except requests.exceptions.RequestException as e:
        print(f"Error during training request: {e}")
        return

    # 2. Predict a non-anomalous data point
    print("\n[2] Predicting a non-anomalous point")
    predict_data_normal = {
        "timestamp": str(int(time.time()) + 10),
        "value": 10.1
    }
    try:
        predict_response_normal = requests.post(
            f"{BASE_URL}/predict/{SERIES_ID}",
            json=predict_data_normal
        )
        predict_response_normal.raise_for_status()
        print("Predict request (normal) successful:")
        print(predict_response_normal.json())
    except requests.exceptions.RequestException as e:
        print(f"Error during prediction request: {e}")


    # 3. Predict an anomalous data point
    print("\n[3] Predicting an anomalous point")
    predict_data_anomaly = {
        "timestamp": str(int(time.time()) + 11),
        "value": 50.0
    }
    try:
        predict_response_anomaly = requests.post(
            f"{BASE_URL}/predict/{SERIES_ID}",
            json=predict_data_anomaly
        )
        predict_response_anomaly.raise_for_status()
        print("Predict request (anomaly) successful:")
        print(predict_response_anomaly.json())
    except requests.exceptions.RequestException as e:
        print(f"Error during prediction request: {e}")


    if model_version:
        print(f"\n[4] Predicting using specific version: {model_version}")
        try:
            predict_response_versioned = requests.post(
                f"{BASE_URL}/predict/{SERIES_ID}?version={model_version}",
                json=predict_data_normal
            )
            predict_response_versioned.raise_for_status()
            print("Predict request (versioned) successful:")
            print(predict_response_versioned.json())
        except requests.exceptions.RequestException as e:
            print(f"Error during versioned prediction request: {e}")


if __name__ == "__main__":
    # Make sure the API is running before executing this script.
    # uvicorn src.main:app --host 0.0.0.0 --port 8000
    test_api()