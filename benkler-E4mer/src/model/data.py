import os
from tsfm_public.toolkit.dataset import PretrainDFDataset
from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor
import pandas as pd
# from src.data import load, preprocess
import requests  # For streaming data from the local server
from src.STATIC import API_KEY, ROOT_DIR, TARGET_IP, PORT


def preprocess_batch(batch_data, timestamp_column, input_columns, id_columns, context_length, tsp=None):
    """
    Preprocess a batch of data and return a PretrainDFDataset object.
    """
    relevant_columns = [timestamp_column] + id_columns + input_columns
    batch_data = batch_data.loc[:, relevant_columns].copy()
    
    if tsp is None:  # Initialize the preprocessor if not done already
        tsp = TimeSeriesPreprocessor(
            timestamp_column=timestamp_column,
            id_columns=id_columns,
            input_columns=input_columns,
            target_columns=input_columns,
            context_length=context_length,
            scaling=True,
        )
        tsp.train(batch_data)  # Only train the scaler once with the first batch

    return tsp, PretrainDFDataset(
        tsp.preprocess(batch_data),
        id_columns=id_columns,
        timestamp_column=timestamp_column,
        context_length=context_length,
    )


# def fetch_next_batch(batch_index):
#     """
#     Fetch the next batch of data from a local server. You should run a local server that serves the batches sequentially.
#     """
#     response = requests.get(f"http://localhost:8000/get_batch?batch_index={batch_index}",
#                             headers={"x-api-key": API_KEY})
#     response.raise_for_status()  # Ensure the request was successful
#     return response.json()  # Assuming JSON format; adjust if necessary


def fetch_data(dataset_code, location='local', batch_size=1000):
    if location == 'remote':
        train_data = []
        val_data = []
        test_data = []
        offset = 0

        while True:
            # Fetch data in batches
            response = requests.get(
                f"http://{TARGET_IP}:{PORT}/get_datasets",
                params={
                    "dataset_code": dataset_code,
                    "batch_size": batch_size,
                    "offset": offset
                },
                headers={"x-api-key": API_KEY}
            )
            response.raise_for_status()
            response_json = response.json()

            # Append fetched data to corresponding lists
            train_data.extend(pd.read_json(response_json['train_json'], orient='records').to_dict(orient='records'))
            val_data.extend(pd.read_json(response_json['val_json'], orient='records').to_dict(orient='records'))
            test_data.extend(pd.read_json(response_json['test_json'], orient='records').to_dict(orient='records'))

            # Stop fetching if less than batch_size items were returned (indicating end of data)
            if len(response_json['train_json']) < batch_size and len(response_json['val_json']) < batch_size and len(response_json['test_json']) < batch_size:
                break

            # Increment offset for next batch
            offset += batch_size

        # Convert lists of dictionaries to DataFrames
        train_data = pd.DataFrame(train_data)
        val_data = pd.DataFrame(val_data)
        test_data = pd.DataFrame(test_data)

    else:
        # Local data loading logic
        data_dir = os.path.join(ROOT_DIR, f'./e4data/train_test_split/{dataset_code}')
        train_file = os.path.join(data_dir, 'train_data.csv')
        val_file = os.path.join(data_dir, 'val_data.csv')
        test_file = os.path.join(data_dir, 'test_data.csv')

        # Read the CSV files
        train_data = pd.read_csv(train_file)
        val_data = pd.read_csv(val_file)
        test_data = pd.read_csv(test_file)

    # Convert datetime column to datetime type
    for dataset in [train_data, val_data, test_data]:
        dataset['datetime'] = pd.to_datetime(dataset['datetime'])

    return train_data, val_data, test_data

def clean_data(data, label_column, input_columns):
    data['label']=data[label_column].astype(float)
    for col in input_columns:
        data[col] = data[col].astype(float)
    return data