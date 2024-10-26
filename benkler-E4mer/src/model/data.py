from tsfm_public.toolkit.dataset import PretrainDFDataset
from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor
import pandas as pd
import requests
from src.STATIC import API_KEY


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


def fetch_data(dataset_code):
    response = requests.get(f"http://localhost:3116/get_datasets?dataset_code={dataset_code}",
                            headers={"x-api-key": API_KEY})
    response.raise_for_status()  # Ensure the request was successful
    response_json = response.json()  # Assuming JSON format; adjust if necessary
    train_data = pd.read_json(response_json['train_json'], orient='records')
    val_data = pd.read_json(response_json['val_json'], orient='records')
    test_data = pd.read_json(response_json['test_json'], orient='records')

    for dataset in [train_data, val_data, test_data]:
        dataset['datetime'] = pd.to_datetime(dataset['datetime'])

    return train_data, val_data, test_data
