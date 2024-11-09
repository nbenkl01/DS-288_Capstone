import argparse
import requests
import pandas as pd
from io import StringIO
from src.model.data import get_data, fetch_data_from_url
from src.utils.config import Config
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

def test_real_data_streaming(config):
    print('Testing Real Data Streaming')
    response_json = fetch_data_from_url(endpoint=f"http://{TARGET_IP}:{PORT}/get_datasets", config=config, offset=0)
    print(response_json.keys())
    print(pd.read_json(StringIO(response_json['train_json']), orient='records').head())

def test_full_data_streaming(config):
    insert = 'Batched' if config.batch_train else 'Full'
    print(f'Testing {insert} Data Streaming')
    train_data, val_data= get_data(config, subset=['train','val'], batch_index=15)
    print(f"Train data size: {len(train_data)}")
    print(f"Val data size: {len(val_data)}")
    print("Train data head:")
    print(train_data.head())
    print("Val data head:")
    print(val_data.head())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run specific tests.")
    parser.add_argument('--test', type=str, default='ctr', help="Specify tests to run: c (connection), t (toy data), r (real data), f (full data), b (batched data). Default is 'ctr'.")

    args = parser.parse_args()
    test_flags = args.test
    config = Config(dataset_code='WESAD', 
                    task='classification',
                    input_columns=['acc_l2_mean','hrv_cvsd','eda_tonic_mean','eda_phasic_mean'],
                    target_columns='binary_stress',
                    id_columns=['subject_id','condition'],
                    data_batch_size=1000,
                    batch_train=False)

    if 'c' in test_flags:
        test_connection()
    if 't' in test_flags:
        test_toy_data_streaming()
    if 'r' in test_flags:
        test_real_data_streaming(config)
    if 'f' in test_flags:
        test_full_data_streaming(config)
    if 'b' in test_flags:
        config.batch_train = True
        test_full_data_streaming(config)
    # if 'n' in test_flags:
    #     config = Config(dataset_code='Nurses', 
    #                 task='classification',
    #                 input_columns=['acc_l2_mean','hrv_cvsd','eda_tonic_mean','eda_phasic_mean'],
    #                 target_columns='binary_stress',
    #                 id_columns=['subject_id','condition'],
    #                 data_batch_size=1000,
    #                 batch_train=False)

    if not test_flags:
        print("No tests specified. Use --help to see available options.")