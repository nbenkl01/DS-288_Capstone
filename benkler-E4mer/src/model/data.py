import os
from tsfm_public.toolkit.dataset import PretrainDFDataset, ClassificationDFDataset
from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor
import pandas as pd
from io import StringIO
import itertools
# from src.data import load, preprocess
import requests  # For streaming data from the local server
from src.STATIC import API_KEY, ROOT_DIR, TARGET_IP, PORT
from tqdm import tqdm

def preprocess(data, config, tsp=None, fit = False):
    """
    Preprocess a batch of data and return a PretrainDFDataset object.
    """
    data = data.loc[:, config.relevant_columns].copy()

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
    df[config.timestamp_column] = pd.to_datetime(df[config.timestamp_column])
    return df

# Helper function to fetch data from a URL
def fetch_data_from_url(endpoint, config, subset = None, offset=None):
    params = {
        "dataset_code": config.dataset_code,
        "columns": config.relevant_columns
    }
    if type(offset) == int:
        params.update({"batch_size": config.data_batch_size, "offset": offset})
    if subset:
        params.update({'subset':subset})
    
    response = requests.get(endpoint, params=params, headers={"x-api-key": API_KEY})
    response.raise_for_status()
    return response.json()


def fetch_data(config, subset = None, batch_index = None):
    # If were doing batch training batch_index will be int so offset should be index*size otherwise it should be None
    # If we're not doing batch training, just batch loading, it should be 0
    offset = batch_index * config.data_batch_size if batch_index else None if config.batch_train else 0
    endpoint = f"http://{TARGET_IP}:{PORT}/get_datasets"

    response_json = fetch_data_from_url(endpoint, config, subset, offset)
    
    data_requested = subset or ['train','val','test']
    parsed_data, pd2 = itertools.tee(map(lambda data_key: parse_json_to_df(response_json, f"{data_key}_json", config), data_requested), 2)
    empty_dfs = list(map(lambda output: output.empty, list(pd2)))
    if all(empty_dfs):
        return map(lambda _: None, empty_dfs)

    return parsed_data
    

# General fetch function to handle remote and local fetching
def get_data(config, subset = None, batch_index = None):
    '''
    Not close to finished, need to fix up elif and else portions
    '''
    if config.batch_train:
        # Fetch & return data in batches but only return once complete
        return fetch_data(config, subset = subset, batch_index = batch_index)
    elif config.data_loc == 'remote':
        train_data, val_data, test_data = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        offset = 0
        
        # Fetch data in batches
        pbar = tqdm(desc="Fetching Data", unit="batch")
        while True:
            response_json = fetch_data_from_url(f"http://{TARGET_IP}:{PORT}/get_datasets", config, offset=offset)

            train_data = pd.concat([train_data, parse_json_to_df(response_json, 'train_json', config)])
            val_data = pd.concat([val_data, parse_json_to_df(response_json, 'val_json', config)])
            test_data = pd.concat([test_data, parse_json_to_df(response_json, 'test_json', config)])

            pbar.set_postfix({"Train Size": len(train_data), "Val Size": len(val_data), "Test Size": len(test_data)})
            pbar.update(1)

            # Stop if the response batch is smaller than requested
            if len(response_json['train_json']) < config.data_batch_size and len(response_json['val_json']) < config.data_batch_size and len(response_json['test_json']) < config.data_batch_size:
                break
            
            offset += config.data_batch_size

        pbar.close()

        pass

        # Integration Attempt Starts Here
        # if not subset:
        #     subset = ['train', 'val', 'test']
        # if 'train' in subset:
        #     train_data = pd.DataFrame()
        # if 'val' in subset:
        #     val_data = pd.DataFrame()
        # if 'test' in subset:
        #     test_data = pd.DataFrame()

        # batch_index = 0
        # pbar = tqdm(desc="Fetching Data", unit="batch")
        # while True:
        #     data_map = list(fetch_data(config, subset = subset, batch_index = batch_index))
        #     data_dict = {key: data_map[i] for i, key in enumerate(subset)}
            
        #     train_data = pd.concat([train_data, parse_json_to_df(response_json, 'train_json', config)])
        #     val_data = pd.concat([val_data, parse_json_to_df(response_json, 'val_json', config)])
        #     test_data = pd.concat([test_data, parse_json_to_df(response_json, 'test_json', config)])

        #     pbar.set_postfix({"Train Size": len(train_data), "Val Size": len(val_data), "Test Size": len(test_data)})
        #     pbar.update(1)

        #     # Stop if the response batch is smaller than requested
        #     if len(response_json['train_json']) < config.data_batch_size and len(response_json['val_json']) < config.data_batch_size and len(response_json['test_json']) < config.data_batch_size:
        #         break

        #     batch_index += 1

        # pbar.close()
        
    else:
        data_dir = os.path.join(ROOT_DIR, f'./e4data/train_test_split/{config.dataset_code}')
        train_data = pd.read_csv(os.path.join(data_dir, 'train_data.csv'))
        val_data = pd.read_csv(os.path.join(data_dir, 'val_data.csv'))
        test_data = pd.read_csv(os.path.join(data_dir, 'test_data.csv'))
        
        for dataset in [train_data, val_data, test_data]:
            dataset[config.timestamp_column] = pd.to_datetime(dataset[config.timestamp_column])

    return train_data, val_data, test_data

def clean_data(data, config):
    if config.task == 'classification':
        data['label']=data[config.target_columns].astype(float)
    for col in config.input_columns:
        data[col] = data[col].astype(float)
    return data