import argparse
import requests
import pandas as pd
from io import StringIO
from src.STATIC import TARGET_IP, PORT, API_KEY

def test_connection():
    print('Testing Connection')
    response = requests.get(
                f"http://{TARGET_IP}:{PORT}/test_connection",
                headers={"x-api-key": API_KEY}
            )
    response.raise_for_status()
    print(response.json())

def test_toy_data_streaming():
    print('Testing Toy Data Streaming')
    response = requests.get(
                f"http://{TARGET_IP}:{PORT}/test_data_streaming",
                headers={"x-api-key": API_KEY}
            )
    response.raise_for_status()
    print(response.json())

def test_real_data_streaming():
    print('Testing Real Data Streaming')
    response = requests.get(
                f"http://{TARGET_IP}:{PORT}/get_datasets",
                params={
                    "dataset_code": 'WESAD',
                    "batch_size": 100,
                    'columns': ['datetime','subject_id','acc_l2_mean','hrv_cvsd','eda_tonic_mean','eda_phasic_mean','binary_stress'],
                    "offset": 0
                },
                headers={"x-api-key": API_KEY}
            )
    response.raise_for_status()
    response_json = response.json()
    print(response_json.keys())
    print(pd.read_json(StringIO(response_json['train_json']), orient='records').head())

def test_full_data_streaming():
    print('Testing Full Data Streaming')
    response = requests.get(
                f"http://{TARGET_IP}:{PORT}/get_full_datasets",
                params={
                    "dataset_code": 'WESAD',
                    'columns': ['datetime','subject_id','acc_l2_mean','hrv_cvsd','eda_tonic_mean','eda_phasic_mean','binary_stress'],
                },
                headers={"x-api-key": API_KEY}
            )
    response.raise_for_status()
    response_json = response.json()
    print(response_json.keys())
    print(pd.read_json(StringIO(response_json['train_json']), orient='records').head())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run specific tests.")
    parser.add_argument('--test', type=str, default='ctr', help="Specify tests to run: c (connection), t (toy data), r (real data), f (full data). Default is 'ctr'.")

    args = parser.parse_args()
    test_flags = args.test

    if 'c' in test_flags:
        test_connection()
    if 't' in test_flags:
        test_toy_data_streaming()
    if 'r' in test_flags:
        test_real_data_streaming()
    if 'f' in test_flags:
        test_full_data_streaming()

    if not test_flags:
        print("No tests specified. Use --help to see available options.")