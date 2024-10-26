import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import pytz
from tqdm import tqdm
import flirt
import CompetencyDemo.utils as utils

#Unlabelled
def load_unlabelled_e4data(unlabelled_dir, subset = None):

    # For all .zip files in unlabelled_dir
    zip_files = [f for f in os.listdir(unlabelled_dir) if f.endswith('.zip')]
    if subset:
        zip_files = [f for f in os.listdir(unlabelled_dir) if f.endswith('.zip') and ('_SPD_' in f or 'WEEE_' in f or 'PPG' in f or 'ue4w' in f) ]
        random.shuffle(zip_files)
        zip_files = zip_files[:subset]
    unreadable_files = []
    collected_data = pd.DataFrame([])
    
    for zip_file in tqdm(zip_files, desc="Processing zip files"):
        try:
            zip_file_path = os.path.join(unlabelled_dir, zip_file)

            # Load the Empatica E4 data using the flirt library
            data = flirt.with_.empatica(zip_file_path)
            data = data.reset_index(names = ['datetime'])
            data['source_id']=zip_file
            split_zf = zip_file.split('_')
            if 'unlabelled' in split_zf:
                dataset = split_zf[split_zf.index('unlabelled')+2]
            else:
                dataset = split_zf[0]
            data['dataset']=dataset
            data = data.set_index(['source_id','datetime'])
            if collected_data.empty:
                collected_data = data.copy()
            else:
                collected_data = pd.concat([collected_data, data])

        except:
            unreadable_files.append(zip_file)

    return collected_data, unreadable_files



# Labelled
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

    df['condition'] = np.nan

    for condition, times in condition_periods.items():
        start_time = pd.to_datetime(times['start'])
        end_time = pd.to_datetime(times['end'])
        
        mask = (df.index >= start_time) & (df.index <= end_time)
        df.loc[mask, 'condition'] = condition

    return df

## WESAD
def load_WESAD_e4data(WESAD_path, subset = None):
    
    unreadable_files = []
    collected_data = pd.DataFrame([])
    
    subjects = [s for s in os.listdir(WESAD_path) if s not in ['.DS_Store', 'wesad_readme.pdf','labelled']]
    if subset:
        subjects = subjects[:subset]
        
    for subject in tqdm(subjects):
        try:
            zip_file_path = os.path.join(WESAD_path,subject,f"{subject}_E4_Data.zip")
            data = flirt.with_.empatica(zip_file_path)
            loader = utils.SurveyDataLoader(WESAD_path, subject)
            data = assign_conditions(data.copy(), loader.condition_intervals)
            data = data.reset_index(names = ['datetime'])
            data['subject_id']=subject
            data = data.set_index(['subject_id','datetime'])
            if collected_data.empty:
                collected_data = data.copy()
            else:
                collected_data = pd.concat([collected_data, data])
        except:
            unreadable_files.append(f"{subject}_E4_Data.zip")
            
    return collected_data, unreadable_files

## Stressed Nurses
def create_condition_intervals(file_path, target_id):
    # Define time zone for Eastern Time
    eastern = pytz.timezone('US/Eastern')
    utc = pytz.utc
    
    # Read the Excel file
    df = pd.read_excel(file_path)
    
    # Filter rows by the given ID
    id_filtered_df = df[df['ID'] == target_id]
    
    condition_intervals = {}
    sorted_intervals = []

    # Iterate over each row to create condition intervals
    for _, row in id_filtered_df.iterrows():
        start_time = row['Start time']
        end_time = row['End time']
        stress = row['Stress level']
        date = str(row['date']).split(' ')[0]
        
        # Combine date with time to form a full datetime string
        start_str = f"{date} {start_time}"
        end_str = f"{date} {end_time}"

        # Parse the start and end times as Eastern time
        start_dt = eastern.localize(datetime.strptime(start_str, '%Y-%m-%d %H:%M:%S'))
        end_dt = eastern.localize(datetime.strptime(end_str, '%Y-%m-%d %H:%M:%S'))

        # Handle case where the end time crosses into the next day
        if end_dt < start_dt:
            end_dt += timedelta(days=1)

        # Convert to UTC
        start_utc = start_dt.astimezone(utc)
        end_utc = end_dt.astimezone(utc)

        # Create the condition interval entry
        interval_key = f"{row['ID']}_{date}-{str(start_time).split(':')[0]}__Stress{stress}"
        condition_intervals[interval_key] = {
            'start': start_utc,
            'end': end_utc,
        }

        # Add the intervals to a list for sorting later
        sorted_intervals.append((interval_key, start_utc, end_utc))
    
    # Sort intervals by start time
    sorted_intervals.sort(key=lambda x: x[1])
    
    # Add additional intervals: from start of day to first condition, and between conditions
    for idx, (key, start_utc, end_utc) in enumerate(sorted_intervals):
        # Get the start date from the first condition's start time
        current_day = start_utc.date()

        # Add interval from start of day to first condition
        if idx == 0:
            start_of_day = datetime.combine(current_day, datetime.min.time()).replace(tzinfo=utc)
            if start_of_day < start_utc:
                condition_intervals[f"{target_id}_{current_day}_StartOfDay_Rest"] = {
                    'start': start_of_day,
                    'end': start_utc
                }

        # Add interval from the end of the current condition to the start of the next condition
        if idx < len(sorted_intervals) - 1:
            next_start_utc = sorted_intervals[idx + 1][1]
            if end_utc < next_start_utc:
                condition_intervals[f"{target_id}_Post_{'_'.join(key.split('_')[:-1])}_Rest"] = {
                    'start': end_utc,
                    'end': next_start_utc
                }

        # If this is the last interval, add an interval to the end of the day
        if idx == len(sorted_intervals) - 1:
            end_of_day = datetime.combine(end_utc.date(), datetime.max.time()).replace(tzinfo=utc)
            if end_utc < end_of_day:
                condition_intervals[f"{target_id}_{end_utc.date()}_EndOfDay_Rest"] = {
                    'start': end_utc,
                    'end': end_of_day
                }

    return condition_intervals

def load_nurses_e4data(nurse_path, subset = None):
    
    unreadable_files = []
    collected_data = pd.DataFrame([])
    
    nurses = [str(n) for n in os.listdir(nurse_path) if n not in ['.DS_Store', 'wesad_readme.pdf','labelled']]
    if subset:
        nurses = nurses[:subset]
        
    for nurse in tqdm(nurses):
        nurse_condition_intervals = create_condition_intervals(os.path.join(nurse_path, '../SurveyResults.xlsx'), nurse)
        nurse_dir = os.path.join(nurse_path, nurse)
        zip_filenames = [name for name in os.listdir(nurse_dir)]
        if subset:
            zip_filenames = zip_filenames[:subset]
        for zip_file_name in zip_filenames:
            zip_file_path = os.path.join(nurse_dir,zip_file_name)
            try:
                data = flirt.with_.empatica(zip_file_path)
                data = assign_conditions(data.copy(), nurse_condition_intervals)
                data['condition'] = data.condition.apply(lambda x: x.split('_')[-1] if type(x)==str else x)
                data = data.reset_index(names = ['datetime'])
                data['subject_id']=nurse
                data = data.set_index(['subject_id','datetime'])
                if collected_data.empty:
                    collected_data = data.copy()
                else:
                    collected_data = pd.concat([collected_data, data])
            except:
                unreadable_files.append(zip_file_path)
            
    return collected_data, unreadable_files