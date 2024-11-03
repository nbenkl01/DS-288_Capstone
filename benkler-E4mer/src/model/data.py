import os
from tsfm_public.toolkit.dataset import PretrainDFDataset, ClassificationDFDataset
from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor
import pandas as pd
from io import StringIO
# from src.data import load, preprocess
import requests  # For streaming data from the local server
from src.STATIC import API_KEY, ROOT_DIR, TARGET_IP, PORT
from tqdm import tqdm


def preprocess_pretrain_batch(train_data, val_data, input_columns, id_columns, context_length, tsp=None, timestamp_column='datetime'):
    """
    Preprocess a batch of data and return a PretrainDFDataset object.
    """
    relevant_columns = [timestamp_column] + id_columns + input_columns
    train_data = train_data.loc[:, relevant_columns].copy()
    val_data = val_data.loc[:,relevant_columns].copy()
    
    if tsp is None:  # Initialize the preprocessor if not done already
        tsp = TimeSeriesPreprocessor(
            timestamp_column=timestamp_column,
            id_columns=id_columns,
            input_columns=input_columns,
            target_columns=input_columns,
            context_length=context_length,
            scaling=True,
        )
    tsp.train(train_data)

    train_dataset = PretrainDFDataset(
        tsp.preprocess(train_data),
        id_columns=id_columns,
        timestamp_column="datetime",
#         observable_columns=forecast_columns,
#         target_columns=forecast_columns,
        context_length=context_length,
#         prediction_length=forecast_horizon,
    )
    valid_dataset = PretrainDFDataset(
        tsp.preprocess(val_data),
        id_columns=id_columns,
        timestamp_column="datetime",
#         observable_columns=forecast_columns,
#         target_columns=forecast_columns,
        context_length=context_length,
#         prediction_length=forecast_horizon,
    )

    return tsp, train_dataset, valid_dataset

def preprocess_classifier_batch(train_data, val_data, input_columns, 
                                id_columns = ['subject_id','condition'],
                                context_length = 512, 
                                stride = 16,
                                tsp=None, timestamp_column='datetime', target_columns = 'label'):
    """
    Preprocess a batch of data and return a PretrainDFDataset object.
    """
    relevant_columns = [timestamp_column] + id_columns + input_columns + [target_columns]
    train_data = train_data.loc[:, relevant_columns].copy()
    val_data = val_data.loc[:,relevant_columns].copy()
    
    if tsp is None:  # Initialize the preprocessor if not done already
        tsp = TimeSeriesPreprocessor(
            timestamp_column=timestamp_column,
            # id_columns=[id_columns[1]],
            id_columns=id_columns,
            # id_columns=[],
            input_columns=input_columns,
            target_columns=[target_columns],
            context_length=context_length,
            scaling=False,
        )
    tsp.train(train_data)  # Only train the scaler once with the first batch
    
    train_dataset = ClassificationDFDataset(
        tsp.preprocess(train_data),
        id_columns=id_columns,
        timestamp_column = timestamp_column,
        input_columns=input_columns,
        label_column=target_columns,
        context_length=context_length,
        stride = stride,
    #     prediction_length=forecast_horizon,
    )
    valid_dataset = ClassificationDFDataset(
        tsp.preprocess(val_data),
        id_columns=id_columns,
        timestamp_column = timestamp_column,
        input_columns=input_columns,
        label_column=target_columns,
        context_length=context_length,
        stride = stride,
    #     prediction_length=forecast_horizon,
    )

    return tsp,  train_dataset, valid_dataset

def preprocess_singular_classifier_dataset(data,
                                           input_columns,
                                            id_columns = ['subject_id','condition'],
                                            context_length = 512, 
                                            stride = 16,
                                            tsp=None,
                                            fit = False,
                                            timestamp_column='datetime',
                                            target_columns = 'label'):
    """
    Preprocess a batch of data and return a PretrainDFDataset object.
    """
    relevant_columns = [timestamp_column] + id_columns + input_columns + [target_columns]
    data = data.loc[:, relevant_columns].copy()
    
    if tsp is None:  # Initialize the preprocessor if not done already
        tsp = TimeSeriesPreprocessor(
            timestamp_column=timestamp_column,
            # id_columns=[id_columns[1]],
            id_columns=id_columns,
            # id_columns=[],
            input_columns=input_columns,
            target_columns=[target_columns],
            context_length=context_length,
            scaling=False,
        )
    if fit:
        tsp.train(data)  # Only train the scaler once with the first batch
    
    dataset = ClassificationDFDataset(
        tsp.preprocess(data),
        id_columns=id_columns,
        timestamp_column = timestamp_column,
        input_columns=input_columns,
        label_column=target_columns,
        context_length=context_length,
        stride = stride,
    #     prediction_length=forecast_horizon,
    )

    return tsp, dataset


# def fetch_next_batch(batch_index):
#     """
#     Fetch the next batch of data from a local server. You should run a local server that serves the batches sequentially.
#     """
#     response = requests.get(f"http://localhost:8000/get_batch?batch_index={batch_index}",
#                             headers={"x-api-key": API_KEY})
#     response.raise_for_status()  # Ensure the request was successful
#     return response.json()  # Assuming JSON format; adjust if necessary

def fetch_next_batch(dataset_code, batch_index, 
                     columns = ['datetime','subject_id','acc_l2_mean','hrv_cvsd','eda_tonic_mean','eda_phasic_mean', 'condition', 'binary_stress'],
                     batch_size=500):
    offset = batch_index*batch_size

    response = requests.get(
        f"http://{TARGET_IP}:{PORT}/get_training_datasets",
        params={
            "dataset_code": dataset_code,
            "batch_size": batch_size,
            "columns": columns,
            "offset": offset
        },
        headers={"x-api-key": API_KEY}
    )
    response.raise_for_status()
    response_json = response.json()

    # Append fetched data to corresponding lists
    train_data = pd.read_json(StringIO(response_json['train_json']), orient='records')
    val_data = pd.read_json(StringIO(response_json['val_json']), orient='records')
    # test_data = pd.read_json(StringIO(response_json['test_json']), orient='records').to_dict(orient='records')
    if len(train_data)<=0 and len(val_data)<=0: 
        return None, None
    
    # Convert datetime column to datetime type
    for dataset in [train_data, val_data]:
        dataset['datetime'] = pd.to_datetime(dataset['datetime'])

    return train_data, val_data

def fetch_test_dataset(dataset_code, #batch_index, 
                     columns = ['datetime','subject_id','acc_l2_mean','hrv_cvsd','eda_tonic_mean','eda_phasic_mean', 'condition', 'binary_stress'],
                     #batch_size=500
                     ):
    # offset = batch_index*batch_size

    response = requests.get(
        f"http://{TARGET_IP}:{PORT}/get_test_data",
        params={
            "dataset_code": dataset_code,
            # "batch_size": batch_size,
            "columns": columns,
            # "offset": offset
        },
        headers={"x-api-key": API_KEY}
    )
    response.raise_for_status()
    response_json = response.json()

    # Append fetched data to corresponding lists
    test_data = pd.read_json(StringIO(response_json['test_json']), orient='records')

    # if len(test_data)<=0: 
    #     return None, None
    
    # Convert datetime column to datetime type
    test_data['datetime'] = pd.to_datetime(test_data['datetime'])

    return test_data


def fetch_data(dataset_code, location='local', columns=['datetime', 'subject_id', 'acc_l2_mean', 'hrv_cvsd', 'eda_tonic_mean', 'eda_phasic_mean', 'condition', 'binary_stress'], batch_size=500):
    if location == 'remote':
        train_data, val_data, test_data = pd.DataFrame([]), pd.DataFrame([]), pd.DataFrame([])
        offset = 0
        
        test_response = requests.get(f"http://{TARGET_IP}:{PORT}/test_connection", headers={"x-api-key": API_KEY})
        test_response.raise_for_status()
        print(test_response.json()['message'])

        pbar = tqdm(desc="Fetching Data", unit="batch")
        while True:
            response = requests.get(
                f"http://{TARGET_IP}:{PORT}/get_datasets",
                params={
                    "dataset_code": dataset_code,
                    "batch_size": batch_size,
                    "columns": columns,
                    "offset": offset
                },
                headers={"x-api-key": API_KEY}
            )
            response.raise_for_status()
            response_json = response.json()

            train_data = pd.concat([train_data,pd.read_json(StringIO(response_json['train_json']), orient='records')])
            val_data = pd.concat([val_data,pd.read_json(StringIO(response_json['val_json']), orient='records')])
            test_data = pd.concat([test_data,pd.read_json(StringIO(response_json['test_json']), orient='records')])

            pbar.set_postfix({"Train Size": len(train_data), "Val Size": len(val_data), "Test Size": len(test_data)})
            pbar.update(1)

            if len(response_json['train_json']) < batch_size and len(response_json['val_json']) < batch_size and len(response_json['test_json']) < batch_size:
                break

            offset += batch_size

        pbar.close()

    else:
        data_dir = os.path.join(ROOT_DIR, f'./e4data/train_test_split/{dataset_code}')
        train_data = pd.read_csv(os.path.join(data_dir, 'train_data.csv'))
        val_data = pd.read_csv(os.path.join(data_dir, 'val_data.csv'))
        test_data = pd.read_csv(os.path.join(data_dir, 'test_data.csv'))

    for dataset in [train_data, val_data, test_data]:
        dataset['datetime'] = pd.to_datetime(dataset['datetime'])

    return train_data, val_data, test_data

def clean_data(data, label_column, input_columns):
    data['label']=data[label_column].astype(float)
    for col in input_columns:
        data[col] = data[col].astype(float)
    return data