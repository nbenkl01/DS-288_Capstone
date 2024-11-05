import os
from tsfm_public.toolkit.dataset import PretrainDFDataset, ClassificationDFDataset
from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor
import pandas as pd
from io import StringIO
import itertools
import requests
from src.STATIC import API_KEY, ROOT_DIR, TARGET_IP, PORT
from tqdm import tqdm

def preprocess(data, config, tsp=None, fit = False):
    """
    Preprocess a batch of data and return a PretrainDFDataset object.
    """
    relevant_columns = config.relevant_columns
    if config.task == 'classification':
        relevant_columns.remove(config.target_columns)
        relevant_columns = relevant_columns + ['label']
    
    data = data.loc[:, relevant_columns].copy()

    shared_params = {'timestamp_column':config.timestamp_column,
                    'id_columns':config.id_columns,
                    'context_length':config.context_length}
    processor_params = {'input_columns':config.input_columns}
    dataset_params = {}
    
    if config.task == 'classification':
        processor_params.update({'scaling':False, 'target_columns':['label']})
        dataset_params.update({'input_columns':config.input_columns,
                               'label_column':'label', 
                               'stride': config.stride})
    else:
        processor_params.update({'scaling':True, 'target_columns':config.input_columns})
    
    if tsp is None:  # Initialize the preprocessor if not done already
        tsp = TimeSeriesPreprocessor(
            **shared_params,
            **processor_params
        )
    if fit:
        tsp.train(data)  # Only train the scaler once with the first batch

    dataset_class = ClassificationDFDataset if config.task == 'classification' else PretrainDFDataset
    dataset = dataset_class(tsp.preprocess(data),
                            **shared_params,
                            **dataset_params
                            )

    return tsp, dataset

# Helper function to parse JSON response to DataFrame
def parse_json_to_df(json_data, key, config):
    df = pd.read_json(StringIO(json_data[key]), orient='records')
    if not df.empty:
        df[config.timestamp_column] = pd.to_datetime(df[config.timestamp_column])
    return df

def parse_local_df(data_dir, data_key, config):
    path = os.path.join(data_dir, f"{data_key}_data.csv")
    df = pd.read_csv(path)
    df[config.timestamp_column] = pd.to_datetime(df[config.timestamp_column])
    return df

# Helper function to fetch data from a URL
def fetch_data_from_url(endpoint, config, subset = None, offset=None):
    params = {
        "dataset_code": config.dataset_code,
        "columns": config.relevant_columns
    }
    # print(type(offset))
    if type(offset) is int:
        # print('posting with offset')
        params.update({"batch_size": config.data_batch_size, "offset": offset})
    if subset:
        params.update({'subset':subset})
    if config.batch_train:
        params.update({'loop':True})
    
    # print(params)
    response = requests.get(endpoint, params=params, headers={"x-api-key": API_KEY})
    response.raise_for_status()
    return response.json()


def fetch_data(config, subset = None, batch_index = None):
    # print(f'batch_index: {batch_index}')
    offset = batch_index * config.data_batch_size if type(batch_index) is int else None
    # print(f'offset: {offset}')
    endpoint = f"http://{TARGET_IP}:{PORT}/get_datasets"

    response_json = fetch_data_from_url(endpoint, config, subset, offset)
    
    data_requested = subset or ['train','val','test']
    parsed_data, pd2 = itertools.tee(map(lambda data_key: parse_json_to_df(response_json, f"{data_key}_json", config), data_requested), 2)
    empty_dfs = list(map(lambda output: output.empty, list(pd2)))
    if all(empty_dfs):
        return map(lambda _: None, empty_dfs)
    
    # print(empty_dfs)
    # print('returning parsed data')

    return parsed_data
    

# General fetch function to handle remote and local fetching
def get_data(config, subset = None, batch_index = None):
    if config.batch_train:
        # Fetch & return data in batches but only return once complete
        return map_return(fetch_data(config, subset = subset, batch_index = batch_index), subset)
    elif config.data_loc == 'remote':
        batch_index = 0

        pbar = tqdm(desc="Fetching Data", unit="batch")
        data_requested = subset or ['train','val','test']
        current_data = map(lambda _: pd.DataFrame([]), data_requested)
        while True:
            fetched_data, fd2 = itertools.tee(fetch_data(config, subset = subset, batch_index = batch_index), 2)
            null_returns = list(map(lambda output: output is None, list(fd2)))
            if all(null_returns):
                pbar.update(1)
                break
            
            current_data = map(lambda stored, fetched: pd.concat([stored, fetched]), list(current_data), list(fetched_data))

            pbar.update(1)

            batch_index += 1

        pbar.close()
        return map_return(current_data, subset)
        
    else:
        data_dir = os.path.join(ROOT_DIR, f'./e4data/train_test_split/{config.dataset_code}')
        data_requested = subset or ['train','val','test']
        return map_return(map(lambda data_key: parse_local_df(data_dir, data_key, config), data_requested), subset)

def clean_data(data, config):
    if config.task == 'classification':
        data['label']=data[config.target_columns].astype(float)
    for col in config.input_columns:
        data[col] = data[col].astype(float)
    return data

def map_return(map_contents, subset):
    return list(map_contents)[0] if len(subset) == 1 else map_contents