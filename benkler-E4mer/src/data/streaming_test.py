import requests
import pandas as pd
# from io import StringIO
from src.STATIC import TARGET_IP, PORT, API_KEY

def test():
    print('Testing Connection')
    response = requests.get(
                f"http://{TARGET_IP}:{PORT}/test_connection",
                headers={"x-api-key": API_KEY}
            )
    response.raise_for_status()
    print(response.json())

    print('Testing Toy Data Streaming')
    response = requests.get(
                f"http://{TARGET_IP}:{PORT}/test_data_streaming",
                headers={"x-api-key": API_KEY}
            )
    response.raise_for_status()
    print(response.json())

    print('Testing Real Data Streaming')
    response = requests.get(
                f"http://{TARGET_IP}:{PORT}/get_datasets",
                params={
                    "dataset_code": 'WESAD',
                    "batch_size": 100,
                    "offset": 0
                },
                headers={"x-api-key": API_KEY}
            )
    response.raise_for_status()
    response_json = response.json()
    print(response_json.keys())
    print(pd.read_json(response_json.json()['train_json'], orient='records').head())


if __name__ == '__main__':
    test()