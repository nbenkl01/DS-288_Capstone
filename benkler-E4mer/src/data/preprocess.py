import pandas as pd
from sklearn.model_selection import GroupKFold, train_test_split

from tsfm_public.toolkit.dataset import PretrainDFDataset
from tsfm_public.toolkit.dataset import RegressionDFDataset
from tsfm_public.toolkit.dataset import ClassificationDFDataset
from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor
# from tsfm_public.toolkit.util import select_by_index, select_by_fixed_fraction

#Splitting
def stratified_split_by_subject(data, subject_col='subject_id', 
                                condition_col='condition',
                                test_size=0.2, val_size=0.2, 
                                random_state=42):
    # Step 1: Get unique subjects and their corresponding condition distributions
    subject_conditions = data.groupby(subject_col)[condition_col].apply(lambda x: x.mode()[0]).reset_index()
    
    # Step 2: Initial split to get test subjects
    train_val_subjects, test_subjects = train_test_split(
        subject_conditions,
        test_size=test_size,
        stratify=subject_conditions[condition_col],
        random_state=random_state
    )
    
    # Step 3: Further split train_val_subjects into train and validation sets
    train_subjects, val_subjects = train_test_split(
        train_val_subjects,
        test_size=val_size / (1 - test_size),  # Adjust validation size based on initial train_val split
        stratify=train_val_subjects[condition_col],
        random_state=random_state
    )
    
    # Step 4: Filter the main dataset by subjects in each set
    train_data = data[data[subject_col].isin(train_subjects[subject_col])]
    val_data = data[data[subject_col].isin(val_subjects[subject_col])]
    test_data = data[data[subject_col].isin(test_subjects[subject_col])]

    return train_data, val_data, test_data

def stratified_group_split(data, subject_col='subject_id', condition_col='condition', test_size=0.2, val_size=0.2, random_state=42):
    # Step 1: Initialize GroupKFold for the initial split (train+val, test)
    group_kfold = GroupKFold(n_splits=int(1 / test_size))
    
    # Step 2: Create arrays for groups (subjects) and conditions
    groups = data[subject_col]
    conditions = data[condition_col]
    
    # Step 3: Split into initial train+val and test sets
    for train_val_idx, test_idx in group_kfold.split(data, y=conditions, groups=groups):
        train_val_data = data.iloc[train_val_idx]
        test_data = data.iloc[test_idx]
        break  # Only take the first split
    
    # Step 4: Further split train_val into train and validation based on groups
    train_val_groups = train_val_data[subject_col]
    train_val_conditions = train_val_data[condition_col]
    
    # Use another GroupKFold to get the train/validation split
    group_kfold_val = GroupKFold(n_splits=int(1 / val_size))
    for train_idx, val_idx in group_kfold_val.split(train_val_data, y=train_val_conditions, groups=train_val_groups):
        train_data = train_val_data.iloc[train_idx]
        val_data = train_val_data.iloc[val_idx]
        break  # Only take the first split
    
    return train_data, val_data, test_data


def simple_split(data, test_size=0.2, val_size=0.2, random_state=42):
    # Step 1: Split into initial train+val and test sets
    train_val_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)
    
    # Step 2: Further split train_val into train and validation sets
    train_data, val_data = train_test_split(train_val_data, test_size=val_size / (1 - test_size), random_state=random_state)
    
    return train_data, val_data, test_data


# Preprocessing
def preprocess_pretraining_datasets(train_data, val_data, test_data,
                        timestamp_column = "datetime",
                        input_columns =['acc_l2_mean','hrv_cvsd','eda_tonic_mean','eda_phasic_mean'],
                        target_columns = [],
                        id_columns = ['source_id'],
                        context_length = 512):
    
    relevant_columns = [timestamp_column]+id_columns+input_columns+target_columns
    train_data = train_data.loc[:,relevant_columns].copy()
    val_data = val_data.loc[:,relevant_columns].copy()
    test_data = test_data.loc[:,relevant_columns].copy()
    
    tsp = TimeSeriesPreprocessor(
        timestamp_column=timestamp_column,
        id_columns=id_columns,
        input_columns=input_columns,
        target_columns=input_columns if len(target_columns) == 0 else target_columns,
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
    test_dataset = PretrainDFDataset(
        tsp.preprocess(test_data),
        id_columns=id_columns,
        timestamp_column="datetime",
#         observable_columns=forecast_columns,
#         target_columns=forecast_columns,
        context_length=context_length,
#         prediction_length=forecast_horizon,
    )
    return tsp, train_dataset, valid_dataset, test_dataset


def preprocess_finetuning_datasets(train_data, val_data, test_data,
                        timestamp_column = "datetime",
                        input_columns =['acc_l2_mean','hrv_cvsd','eda_tonic_mean','eda_phasic_mean'],
                        target_columns = 'binary_stress',
                        # target_columns = ['binary_stress'],
                        id_columns = ['subject_id'],
                        context_length = 512):
    
    # relevant_columns = [timestamp_column]+id_columns+input_columns+target_columns
    relevant_columns = [timestamp_column]+id_columns+input_columns+[target_columns]
    train_data = train_data.loc[:,relevant_columns].copy()
    val_data = val_data.loc[:,relevant_columns].copy()
    test_data = test_data.loc[:,relevant_columns].copy()
    
    tsp = TimeSeriesPreprocessor(
        timestamp_column=timestamp_column,
        id_columns=id_columns,
        observable_columns=input_columns,
        target_columns= [target_columns],
        context_length=context_length,
        scaling=True,
    )
    tsp.train(train_data)
    
    train_dataset = ClassificationDFDataset(
        tsp.preprocess(train_data),
        id_columns=id_columns,
        timestamp_column = timestamp_column,
        input_columns=input_columns,
        label_column=target_columns,
        context_length=context_length,
    #     prediction_length=forecast_horizon,
    )
    valid_dataset = ClassificationDFDataset(
        tsp.preprocess(val_data),
        id_columns=id_columns,
        timestamp_column = timestamp_column,
        input_columns=input_columns,
        label_column=target_columns,
        context_length=context_length,
    #     prediction_length=forecast_horizon,
    )
    test_dataset = ClassificationDFDataset(
        tsp.preprocess(test_data),
        id_columns=id_columns,
        timestamp_column = timestamp_column,
        input_columns=input_columns,
        label_column=target_columns,
        context_length=context_length,
    #     prediction_length=forecast_horizon,
    )
    return tsp, train_dataset, valid_dataset, test_dataset