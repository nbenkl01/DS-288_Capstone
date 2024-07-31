import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import flirt.with_
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from scipy.stats import kendalltau
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime, timedelta
import pytz
from tqdm import tqdm

class EmpLoader:
    """
    A class to handle the loading and processing of Empatica WESAD dataset.

    Attributes:
    - WESAD_path: Path to the WESAD dataset.
    - predictors: Predictor variable for model training.
    - target_survey: The survey from which to extract target values.
    - target: The specific survey target variable.
    """
    def __init__(self, WESAD_path, predictors = 'hrv_rmssd', target_survey='PANAS', target='Stressed', subjects = ''):
        self.WESAD_path = WESAD_path
        self.predictors = predictors
        self.target_survey = target_survey
        self.target = target
        self.subjects = [subject for subject in os.listdir(WESAD_path) if subject.startswith('S')] if subjects == '' else subjects
        self.X = []
        self.y = []

    def load_data(self):
        """
        Load and process data for each subject, extracting features and targets.
        """
        for subject in tqdm(self.subjects):
            loader, data = self.load_emp_data(subject)
            
            labeled_df = assign_conditions(data, loader.get_condition_intervals())
            scored_df = assign_scores(labeled_df, loader, self.target_survey, [self.target] if isinstance(self.target, str) else self.target)
            
            if isinstance(self.predictors, str):
                self.extract_vector(scored_df)
            else:
                raise NotImplementedError('Support for multiple predictor variables not implemented yet')

    def load_emp_data(self, subject):
        zip_file_path = f"{self.WESAD_path}{subject}/{subject}_E4_Data.zip"
            
        data = flirt.with_.empatica(zip_file_path,
                                    window_length=180,
                                    window_step_size=1,
                                    hrv_features=True,
                                    eda_features=False,
                                    acc_features=False)
        
        loader = SurveyDataLoader(self.WESAD_path, subject)
        loader.get_condition_intervals()
        return loader, data
        
    def extract_vector(self, scored_df):
        """
        Extract features and target values into lists.
        
        Parameters:
        - scored_df: DataFrame with scored conditions.
        """
        for condition in scored_df.condition.unique():
            if isinstance(condition, str):
                subset = scored_df.loc[scored_df.condition == condition, [self.predictors, self.target]]
                embedding = list(subset[[self.predictors]].values.ravel())
                self.X.append(embedding)
                label = subset[self.target].unique()[0]
                self.y.append([label])

class SurveyDataLoader:
    """
    A class to load and parse survey data for each subject.

    Attributes:
    - WESAD_path: Path to the WESAD dataset.
    - subject: Subject ID.
    """
    def __init__(self, WESAD_path, subject):
        self.survey_path = f"{WESAD_path}{subject}/{subject}_quest.csv"
        self.respiban_path = f"{WESAD_path}{subject}/{subject}_respiban.txt"
        self.survey_data = pd.read_csv(self.survey_path, header=None, sep=';')
        self.protocol_conditions = []
        self.acceptible_protocol_conditions = ['Base', 'TSST', 'Medi 1', 'Fun', 'Medi 2']
        self.condition_intervals = {}
        self.questionnaires = {
            'PANAS': [],
            'STAI': [],
            'SAM': [],
            'SSSQ': []
        }
        self._parse_data()

    def _parse_data(self):
        """
        Parse survey data including protocol conditions, time intervals, and questionnaires.
        """
        self.protocol_conditions = self.survey_data.iloc[1, 1:].dropna().tolist()
        self.protocol_conditions = [cond for cond in self.protocol_conditions if cond in self.acceptible_protocol_conditions]

        start_times = self.survey_data.iloc[2, 1:].dropna().tolist()
        end_times = self.survey_data.iloc[3, 1:].dropna().tolist()
        self.condition_intervals = {
            condition: {'start': start, 'end': end}
            for condition, start, end in zip(self.protocol_conditions, start_times, end_times)
        }
        
        self._parse_respiban_datetime()
        self._convert_condition_intervals_to_utc()
        self._parse_questionnaires()

    def _parse_questionnaires(self):
        """
        Parse and extract responses from various questionnaires.
        """
        # PANAS
        panas_items = [
            'Active', 'Distressed', 'Interested', 'Inspired', 'Annoyed', 'Strong', 'Guilty',
            'Scared', 'Hostile', 'Excited', 'Proud', 'Irritable', 'Enthusiastic', 'Ashamed', 'Alert',
            'Nervous', 'Determined', 'Attentive', 'Jittery', 'Afraid', 'Stressed', 'Frustrated', 'Happy',
            'Sad'
        ]
        panas_data = self.survey_data[self.survey_data[0].str.contains('PANAS', na=False)]
        for idx in range(len(self.protocol_conditions)):
            responses = panas_data.iloc[idx, 1:].dropna().tolist()
            if len(responses) > len(panas_items):
                responses = responses[:-2]  # Exclude extra columns if they exist
            self.questionnaires['PANAS'].append(dict(zip(panas_items, responses)))

        # STAI
        stai_items = [
            'I feel at ease', 'I feel nervous', 'I am jittery', 'I am relaxed', 'I am worried', 'I feel pleasant'
        ]
        stai_data = self.survey_data[self.survey_data[0].str.contains('STAI', na=False)]
        for idx in range(len(self.protocol_conditions)):
            responses = stai_data.iloc[idx, 1:].dropna().tolist()
            self.questionnaires['STAI'].append(dict(zip(stai_items, responses)))

        # SAM
        sam_items = ['Valence', 'Arousal']
        sam_data = self.survey_data[self.survey_data[0].str.contains('DIM', na=False)]
        for idx in range(len(self.protocol_conditions)):
            responses = sam_data.iloc[idx, 1:].dropna().tolist()
            self.questionnaires['SAM'].append(dict(zip(sam_items, responses)))

        # SSSQ (only after stress condition)
        sssq_items = [
            'I was annoyed', 'I was angry', 'I was irritated',
            'I was committed to attaining my performance goals', 'I wanted to succeed on the task',
            'I was motivated to do the task', 'I reflected about myself',
            'I was worried about what other people think of me', 'I felt concerned about the impression I was making'
        ]
        sssq_data = self.survey_data[self.survey_data[0].str.contains('SSSQ', na=False)]
        
        responses = sssq_data.iloc[0, 1:].dropna().tolist()
        responses = [self.questionnaires['PANAS'][self.protocol_conditions.index('TSST')]['Annoyed']] + responses
        self.questionnaires['SSSQ'].append(dict(zip(sssq_items, responses)))
        
    def _parse_respiban_datetime(self):
        """
        Parse date and time information from the RESPIBAN file.
        """
        with open(self.respiban_path, 'r') as file:
            lines = file.readlines()

        header_lines = []
        for line in lines:
            if line.startswith("# EndOfHeader"):
                break
            header_lines.append(line.strip("# ").strip())

        header_info_str = header_lines[1]
        header_info_json = json.loads(header_info_str)
        device_info = header_info_json[list(header_info_json.keys())[0]]

        date_str = device_info["date"]
        time_str = device_info["time"]
        date_time_str = f"{date_str} {time_str}"
        date_time_obj = datetime.strptime(date_time_str, "%Y-%m-%d %H:%M:%S.%f")

        self.start_datetime = date_time_obj
    
    def _convert_condition_intervals_to_utc(self):
        """
        Convert condition intervals to UTC timezone.
        """
        utc_tz = pytz.UTC
        start_datetime_utc = pytz.timezone("Etc/GMT-2").localize(self.start_datetime).astimezone(utc_tz)

        for condition, times in self.condition_intervals.items():
            start_minutes = float(times['start'])
            end_minutes = float(times['end'])

            start_time_utc = start_datetime_utc + timedelta(minutes=start_minutes)
            end_time_utc = start_datetime_utc + timedelta(minutes=end_minutes)

            self.condition_intervals[condition] = {
                'start': start_time_utc,
                'end': end_time_utc
            }

    def get_protocol_conditions(self):
        """
        Get the list of protocol conditions.
        """
        return self.protocol_conditions

    def get_condition_intervals(self):
        """
        Get the start and end times of each condition.
        """
        return self.condition_intervals

    def get_questionnaire_responses(self, questionnaire_type):
        """
        Get responses from a specific questionnaire.

        Parameters:
        - questionnaire_type: Type of questionnaire to retrieve ('PANAS', 'STAI', 'SAM', 'SSSQ').

        Returns:
        - List of responses for the specified questionnaire.
        """
        if questionnaire_type not in self.questionnaires:
            raise ValueError(f"Invalid questionnaire type: {questionnaire_type}")
        return self.questionnaires[questionnaire_type]

def assign_conditions(df, condition_periods):
    """
    Assign a 'condition' column to the DataFrame based on the datetime index and given condition periods.

    Parameters:
    - df (pd.DataFrame): DataFrame with a datetime index.
    - condition_periods (dict): Dictionary with condition periods where each condition has 'start' and 'end' times.

    Returns:
    - pd.DataFrame: DataFrame with an additional 'condition' column.
    """
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        raise ValueError("Index of the DataFrame must be a datetime index.")

    df['condition'] = pd.NA

    for condition, times in condition_periods.items():
        start_time = pd.to_datetime(times['start'])
        end_time = pd.to_datetime(times['end'])
        
        mask = (df.index >= start_time) & (df.index <= end_time)
        df.loc[mask, 'condition'] = condition

    return df

def assign_scores(df, loader, survey, columns):
    """
    Assign scores to the DataFrame based on survey responses.

    Parameters:
    - df (pd.DataFrame): DataFrame with a datetime index.
    - loader (SurveyDataLoader): Survey data loader object.
    - survey (str): The type of survey from which to get scores.
    - columns (list): List of columns to include from the survey data.

    Returns:
    - pd.DataFrame: DataFrame with assigned scores.
    """
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        raise ValueError("Index of the DataFrame must be a datetime index.")
    
    scores_df = pd.DataFrame(loader.get_questionnaire_responses(survey))[columns]
    scores_df['condition'] = loader.protocol_conditions
    
    labeled_df = df.reset_index().merge(scores_df, on='condition', how='left')

    return labeled_df

def preprocess_data(X, y):
    """
    Preprocess data by padding sequences and normalizing.

    Parameters:
    - X (list): List of feature sequences.
    - y (list): List of target values.

    Returns:
    - tuple: Processed feature tensor and target tensor.
    """
    max_len = max(len(x) for x in X)
    X_padded = [np.pad(x, (0, max_len - len(x)), 'constant') for x in X]
    scaler = StandardScaler()
    X_padded = scaler.fit_transform(np.array(X_padded))
    X_tensor = torch.tensor(X_padded, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    return X_tensor, y_tensor

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.

    Parameters:
    - X (list): List of feature sequences.
    - y (list): List of target values.
    - test_size (float): Proportion of data to use for testing.
    - random_state (int): Seed for random number generator.

    Returns:
    - tuple: Training and testing tensors for features and targets.
    """
    max_len = max(len(x) for x in X)
    X_padded = [np.pad(x, (0, max_len - len(x)), 'constant') for x in X]
    scaler = StandardScaler()
    X_padded = scaler.fit_transform(np.array(X_padded))
    
    X_train_np, X_test_np, y_train, y_test = train_test_split(X_padded, y, test_size=test_size, random_state=random_state)
    
    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    X_test = torch.tensor(X_test_np, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    
    return X_train, X_test, y_train, y_test

def create_dataloaders(X_train, y_train, X_test, y_test, batch_size=2):
    """
    Create DataLoader objects for training and testing datasets.

    Parameters:
    - X_train (torch.Tensor): Training feature tensor.
    - y_train (torch.Tensor): Training target tensor.
    - X_test (torch.Tensor): Testing feature tensor.
    - y_test (torch.Tensor): Testing target tensor.
    - batch_size (int): Batch size for DataLoader.

    Returns:
    - tuple: DataLoaders for training and testing datasets.
    """
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def define_model(input_dim, hidden_dim, model_type = 'simple', num_heads=None, num_layers=None):
    """
    Define a regression model based on the specified type.

    Parameters:
    - input_dim (int): Number of input features.
    - hidden_dim (int): Dimension of the hidden layer.
    - model_type (str): Type of model to define ('simple', 'transformer', 'itransformer', 'lstm').
    - num_heads (int, optional): Number of attention heads (for transformer models).
    - num_layers (int, optional): Number of layers (for transformer and LSTM models).

    Returns:
    - nn.Module: The defined model.
    """
    class SimpleRegressionModel(nn.Module):
        def __init__(self, input_dim, hidden_dim):
            super(SimpleRegressionModel, self).__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_dim, 1)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x

    class TransformerRegressionModel(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_heads, num_layers):
            super(TransformerRegressionModel, self).__init__()
            self.embedding = nn.Linear(input_dim, hidden_dim)
            encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.fc_out = nn.Linear(hidden_dim, 1)

        def forward(self, x):
            x = self.embedding(x)
            x = self.transformer_encoder(x)
            # x = x.mean(dim=0)
            x = self.fc_out(x)
            return x

    class iTransformerRegressionModel(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_heads, num_layers):
            super(iTransformerRegressionModel, self).__init__()
            self.embedding = nn.Linear(input_dim, hidden_dim)
            self.depthwise_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, groups=hidden_dim, padding=1)
            self.pointwise_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
            self.attention = nn.MultiheadAttention(hidden_dim, num_heads)
            self.fc_layers = nn.Sequential(*[nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
            self.fc_out = nn.Linear(hidden_dim, 1)

        def forward(self, x):
            x = self.embedding(x).permute(1, 2, 0)
            x = self.depthwise_conv(x)
            x = self.pointwise_conv(x)
            x = x.permute(2, 0, 1)
            x, _ = self.attention(x, x, x)
            for fc in self.fc_layers:
                x = F.relu(fc(x))
            x = x.mean(dim=0)
            x = self.fc_out(x)
            return x

    class LSTMRegressionModel(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_layers):
            super(LSTMRegressionModel, self).__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, 1)

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            print(lstm_out)
            lstm_out = lstm_out[:, -1, :]
            out = self.fc(lstm_out)
            return out

    if model_type == 'simple':
        return SimpleRegressionModel(input_dim, hidden_dim)
    elif model_type == 'transformer':
        if num_heads is None or num_layers is None:
            raise ValueError("num_heads and num_layers must be specified for transformer model.")
        return TransformerRegressionModel(input_dim, hidden_dim, num_heads, num_layers)
    elif model_type == 'itransformer':
        if num_heads is None or num_layers is None:
            raise ValueError("num_heads and num_layers must be specified for iTransformer model.")
        return iTransformerRegressionModel(input_dim, hidden_dim, num_heads, num_layers)
    elif model_type == 'lstm':
        if num_layers is None:
            raise ValueError("num_layers must be specified for LSTM model.")
        return LSTMRegressionModel(input_dim, hidden_dim, num_layers)
    else:
        raise ValueError("Invalid model type specified.")

def train_model(model, dataloader, num_epochs=10, learning_rate=0.001):
    """
    Train the model using the provided DataLoader and save loss for each epoch.

    Parameters:
    - model (nn.Module): The model to train.
    - dataloader (DataLoader): DataLoader for training data.
    - num_epochs (int): Number of epochs for training.
    - learning_rate (float): Learning rate for optimizer.

    Returns:
    - list: A list of epoch losses.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    loss_results = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * batch_X.size(0)
        
        epoch_loss = running_loss / len(dataloader.dataset)
        loss_results.append(epoch_loss)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
    
    return loss_results

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on test data and return predictions and metrics.

    Parameters:
    - model (nn.Module): The trained model.
    - X_test (torch.Tensor): Testing feature tensor.
    - y_test (torch.Tensor): Testing target tensor.

    Returns:
    - dict: Dictionary containing predictions, actual values, and MSE.
    """
    model.eval()
    with torch.no_grad():
        predictions = model(X_test).squeeze().numpy()
        actuals = y_test.numpy()
    
    rmse = root_mean_squared_error(actuals, predictions)
    tau, _ = kendalltau(actuals, predictions)
    return {'predictions': predictions, 'actuals': actuals, 'rmse': rmse, 'tau': tau}

def visualize_results(results_dict):
    """
    Visualize the results for different models by plotting predictions against actual values,
    and display MSE and Kendall's tau.

    Parameters:
    - results_dict (dict): Dictionary with model names as keys and evaluation results as values.
    """
    plt.figure(figsize=(15, 5))
    for i, (model_name, results) in enumerate(results_dict.items(), 1):
        plt.subplot(1, len(results_dict), i)
        plt.scatter(range(len(results['actuals'])), results['actuals'], color='red', label='Actuals', alpha=0.5)
        plt.scatter(range(len(results['predictions'])), results['predictions'], color='blue', label='Predictions', alpha=0.5)
        plt.xlabel('Sample Index')
        plt.ylabel('Stress Value (Target)')
        plt.title(f'{model_name} Predicted vs Actual\n'
                  f'(RMSE: {results["rmse"]:.4f}, '
                  f'Kendall\'s Tau: {results["tau"]:.4f})'
                  )
        plt.legend()
    plt.tight_layout()
    plt.show()

def visualize_loss(loss_results_dict):
    """
    Visualize the training loss over epochs for different models.

    Parameters:
    - loss_results_dict (dict): Dictionary with model names as keys and loss lists as values.
    """
    plt.figure(figsize=(10, 5))
    for model_name, loss_results in loss_results_dict.items():
        plt.plot(range(1, len(loss_results) + 1), loss_results, label=model_name)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.show()

def main():
    """
    Main function to execute the entire pipeline: loading data, training, and evaluation.
    """
    emp_data = EmpLoader(WESAD_path="../../Data:Code Sources/WESAD/", 
                        predictors='hrv_rmssd',
                        target_survey='PANAS', 
                        target='Stressed')
    emp_data.load_data()
    X_train, X_test, y_train, y_test = split_data(emp_data.X, emp_data.y)
    train_loader, _ = create_dataloaders(X_train, y_train, X_test, y_test)
    input_dim = X_train.size(1)
    hidden_dim = 10
    model = define_model(input_dim, hidden_dim)
    loss_results = train_model(model, train_loader)
    visualize_loss({
        'MLP': loss_results,
        })
    results = evaluate_model(model, X_test, y_test)
    visualize_results({
            'MLP': results,
        })