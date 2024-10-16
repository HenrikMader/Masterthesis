import pandas as pd
import pandas as pd
from pm4py.algo.transformation.log_to_features import algorithm as log_to_features
import torch
import pandas as pd
from torch.utils.data import TensorDataset
from datetime import datetime
import pm4py
import os
import numpy as np
import time



device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')






'''

    Maybe we have a problem here?

'''
def creatingInitialEmbedding(xesDataframe, features_to_index):

    ## Does this always create the same embedding? Check this!
    categorical_indices = []
    for i in range(0,xesDataframe.shape[0]):   
        helperDf = xesDataframe.reset_index()
        valueFromPrefix = helperDf.loc[i, "concept:name"]
        valueFromPrefix = "concept:name_" + valueFromPrefix.replace(" ", "")
        if (valueFromPrefix in features_to_index):
            index = features_to_index[valueFromPrefix]
            categorical_indices.append(index)
    timestamps = pd.to_datetime(xesDataframe["time:timestamp"])
    timestamps = timestamps.reset_index(drop=True)
    start_time = timestamps.min()
    elapsed_time = timestamps - start_time
    elapsed_time_minutes = (elapsed_time.dt.total_seconds()) / (60 * 60)

    mean_val = elapsed_time_minutes.mean()
    std_val = elapsed_time_minutes.std()

    standardized = (elapsed_time_minutes - mean_val) / std_val

    min_time = elapsed_time_minutes.min()
    max_time = elapsed_time_minutes.max()
    scaled_time = (elapsed_time_minutes - min_time) / (max_time - min_time)



    return categorical_indices, elapsed_time_minutes


def z_standardize_non_zero(timestamps_list):
    # Convert list to NumPy array
    timestamps_array = np.array(timestamps_list, dtype=float)
    
    # Extract non-zero values
    non_zero_values = timestamps_array[timestamps_array > 0]
    
    # Compute mean and standard deviation of non-zero values
    mean = np.mean(non_zero_values)
    std = np.std(non_zero_values)
    
    print(f"Mean: {mean}, Standard Deviation: {std}")
    
    # Create a mask for non-zero values
    mask = timestamps_array > 0
    
    # Apply Z-standardization to non-zero values
    timestamps_array[mask] = (timestamps_array[mask] - mean) / std
    
    return timestamps_array

def createSlidingWindowEmbedding(slidingWindowNumber, arrayInitialEmbedding, elapsedTimeMinutes):

    allWindows = []
    allTimeWindows = []
    for i in range(len(arrayInitialEmbedding)):
        listWindow = [0] * (slidingWindowNumber - 1)
        listTime = [0] * (slidingWindowNumber - 1)
        list_time_in_day = [0] * (slidingWindowNumber - 1)
        all_weekdays = [0] * (slidingWindowNumber - 1)
        all_time_before_event = [0] * (slidingWindowNumber - 1)
        if (i < slidingWindowNumber):
            listWindow[slidingWindowNumber - i -1:] = arrayInitialEmbedding[0:i + 1]
            listTime[slidingWindowNumber -i - 1:] = (elapsedTimeMinutes[0:i + 1]).reset_index(drop=True)
            allWindows.append(listWindow)
            allTimeWindows.append(listTime)
        else:
            listWindow = arrayInitialEmbedding[i - slidingWindowNumber + 1:i + 1]
            listTime = (elapsedTimeMinutes[i - slidingWindowNumber + 1:i + 1]).tolist()
            allWindows.append(listWindow)
            allTimeWindows.append(listTime)
    return allWindows, allTimeWindows

def creatingTensorsTrainingAndTesting(wholeDataFrame, featuresDfInput, sliding_window):
    array_values_all = []
    array_time_all = []
    array_label_all = []

    grouped = wholeDataFrame['case:concept:name'].unique()
    for case_name in grouped:
        filtered_df = wholeDataFrame[wholeDataFrame['case:concept:name'] == case_name]

        initialList, listTime = creatingInitialEmbedding(filtered_df, featuresDfInput)
        arrayListWindow, listTimeWindow = createSlidingWindowEmbedding(sliding_window, initialList, listTime)


        for j in range(len(arrayListWindow)):
            '''and (j != 0) add this if first case should be skipped '''
            if (pd.notna(filtered_df.reset_index().loc[j, "remain_time"])) and j != 0:
                
                result = [filtered_df.reset_index().loc[j,"remain_time"]]
                
                array_label_all.append(result)

                min_val = np.min(listTimeWindow[j])
                max_val = np.max(listTimeWindow[j])

                array_time_all.append(listTimeWindow[j])
                array_values_all.append(arrayListWindow[j])

    features_tensor = torch.tensor(array_values_all, dtype=torch.long).to(device)
    labels_tensor = torch.tensor(array_label_all, dtype=torch.long).to(device)
    timestamps_tensor = torch.tensor(array_time_all, dtype=torch.float).to(device)

    dataset = TensorDataset(features_tensor, timestamps_tensor, labels_tensor)
    return dataset


