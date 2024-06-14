import pandas as pd
import pandas as pd
from pm4py.algo.transformation.log_to_features import algorithm as log_to_features
import torch
import pandas as pd
from torch.utils.data import TensorDataset
from datetime import datetime
import pm4py



device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')




'''

    Other features to add: Time between activities -> elapsed time
    Time within week
    Time within day
'''



## Problem: Think about small sliding Windows
def creatingInitialDataframeOneHot(dataFrame, featuresForMapping):

    new_event_dict = {key.replace('concept:name_', ''): value for key, value in featuresForMapping.items()}
    columns = list(new_event_dict.keys())
    rows = list(dataFrame.index)
    df_new = pd.DataFrame(0, index=rows, columns=columns)

    # Index of current row and current row
    for idx, row in dataFrame.iterrows():
        concept_name = row['concept:name']
        ## We could also replace with -1
        concept_name = concept_name.replace(" ", "")
        df_new.loc[idx, concept_name] = 1

    timestamps = pd.to_datetime(dataFrame["time:timestamp"])
    timestamps = timestamps.reset_index(drop=True)
    start_time = timestamps.min()
    elapsed_time = timestamps - start_time
    elapsed_time_minutes = (elapsed_time.dt.total_seconds())

    df_new["elapsed_time"] = elapsed_time_minutes
    #df_new["remain_time"] = dataFrame["remain_time"]

    target = dataFrame["remain_time"]

    return df_new, target

## 1 dataset = 1 input. Haben einen timestamp dazu
def createSlidingWindow(slidingWindowNumber, initialDataframe):
    allSlidingWindows = []
    for idx, row in initialDataframe.iterrows():
        if (idx == 0):
            print("Sth")
        columns = list(initialDataframe.columns)
        rows = range(slidingWindowNumber)
        df_new = pd.DataFrame(0, index=rows, columns=columns)
        if (idx < slidingWindowNumber - 1):
            df_new.iloc[-idx - 1: , : ] = initialDataframe.iloc[0:idx +1 , :]
        else:
            df_new = initialDataframe.iloc[idx-slidingWindowNumber + 1:idx + 1, :]
        allSlidingWindows.append(df_new)
    return allSlidingWindows

def creatingTensorAndTrainingOneHot(wholeDataFrame, featuresDfInput, sliding_window):
    tensorListValues = []
    tensorListTargets = []
    grouped_df = wholeDataFrame.groupby('case:concept:name').first().reset_index()

    for case_name in grouped_df['case:concept:name']:
        filtered_df = wholeDataFrame[wholeDataFrame['case:concept:name'] == case_name]

        initialDataframe, targets = creatingInitialDataframeOneHot(filtered_df, featuresDfInput)
        arraySlidingWindow = createSlidingWindow(sliding_window, initialDataframe)

        for df, target in zip(arraySlidingWindow, targets):
            # Convert dataframe to NumPy array
            data_array = df.values 
            tensor = torch.tensor(data_array, dtype=torch.float32)
            tensorListValues.append(tensor)

            # Assuming target is a scalar value
            target_tensor = torch.tensor(target, dtype=torch.float32)
            tensorListTargets.append(target_tensor)

    # Stack tensors along a new dimension (assuming each tensor is of shape (sequence_length, num_features))
    fullTensorValues = torch.stack(tensorListValues, dim=0).to(device)
    fullTensorTargets = torch.stack(tensorListTargets, dim=0).to(device)

    # Create TensorDataset with features and targets
    dataset = TensorDataset(fullTensorValues, fullTensorTargets)

    return dataset




    ## Input: (batch_size, seq_len, input_size)





def creatingInitialEmbedding(xesDataframe, features_to_index):
    ## Does this always create the same embedding? Check this!
    categorical_indices = []
    # Dropping first and last element like in Paper of Tax
    #xesDataframe = xesDataframe.drop(xesDataframe.index[-1])
    #if (not xesDataframe.empty):
    #    xesDataframe = xesDataframe.drop(xesDataframe.index[0])
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
    elapsed_time_minutes = (elapsed_time.dt.total_seconds())
    timeInDay = timestamps.dt.time
    def time_to_hours(time_obj):
        total_hours = (time_obj.hour +
                    time_obj.minute / 60 +
                    time_obj.second / 3600 +
                    time_obj.microsecond / (3600 * 1_000_000))
        return total_hours
    

    # Convert to an integer (you might want to round, floor, or ceil depending on your use case)
    minutes_list = [time_to_hours(time_str) for time_str in timeInDay]

    timestamp_weekday = timestamps.dt.weekday + 1

    return categorical_indices, elapsed_time_minutes, minutes_list, timestamp_weekday

# Give the elapsed time here
def createSlidingWindowEmbedding(slidingWindowNumber, arrayInitialEmbedding, elapsedTimeMinutes, time_in_day, timestamps_weekday):

    allWindows = []
    allTimeWindows = []
    all_time_in_day_windows = []
    all_weekdays_windows = []
    for i in range(len(arrayInitialEmbedding)):
        listWindow = [0] * (slidingWindowNumber - 1)
        listTime = [0] * (slidingWindowNumber - 1)
        list_time_in_day = [0] * (slidingWindowNumber - 1)
        all_weekdays = [0] * (slidingWindowNumber - 1)
        if (i < slidingWindowNumber):
            listWindow[slidingWindowNumber - i -1:] = arrayInitialEmbedding[0:i + 1]
            listTime[slidingWindowNumber -i - 1:] = (elapsedTimeMinutes[0:i + 1]).reset_index(drop=True)
            list_time_in_day[slidingWindowNumber -i - 1:] = (time_in_day[0:i + 1])
            all_weekdays[slidingWindowNumber -i - 1:] = timestamps_weekday[0:i + 1]
            allWindows.append(listWindow)
            allTimeWindows.append(listTime)
            all_time_in_day_windows.append(list_time_in_day)
            all_weekdays_windows.append(all_weekdays)
        ## Irgendetwas stimmt hier nicht
        else:
            listWindow = arrayInitialEmbedding[i - slidingWindowNumber + 1:i + 1]
            listTime = (elapsedTimeMinutes[i - slidingWindowNumber + 1:i + 1]).tolist()
            list_time_in_day = (time_in_day[i - slidingWindowNumber + 1:i + 1])
            all_weekdays = (timestamps_weekday[i - slidingWindowNumber + 1:i + 1]).tolist()
            allWindows.append(listWindow)
            allTimeWindows.append(listTime)
            all_time_in_day_windows.append(list_time_in_day)
            all_weekdays_windows.append(all_weekdays)
    return allWindows, allTimeWindows, all_time_in_day_windows, all_weekdays_windows


def creatingTensorsTrainingAndTesting(wholeDataFrame, featuresDfInput, sliding_window):
    array_values_all = []
    array_time_all = []
    array_label_all = []
    array_time_in_day = []
    array_weekday = []
    grouped_df = wholeDataFrame.groupby('case:concept:name').first().reset_index()

    for case_name in grouped_df['case:concept:name']:
        filtered_df = wholeDataFrame[wholeDataFrame['case:concept:name'] == case_name]
        initialList, listTime, timeInDay, timestamps_weekday = creatingInitialEmbedding(filtered_df, featuresDfInput)
        arrayListWindow, listTimeWindow, listTimeInDayWindows, all_weekday_windows = createSlidingWindowEmbedding(sliding_window, initialList, listTime, timeInDay, timestamps_weekday)


        ## Check if this means that whole trace is not looked at!
        # Do this with simple trace, where the first event is nan and the others not
        for j in range(len(arrayListWindow)):
            if pd.notna(filtered_df.reset_index().loc[j,"remain_time"]):
                result = [filtered_df.reset_index().loc[j,"remain_time"]]
                array_label_all.append(result)
                array_time_all.append(listTimeWindow[j])
                array_values_all.append(arrayListWindow[j])
                array_time_in_day.append(listTimeInDayWindows[j])
                # Why sometimes to list an sometimes not
                array_weekday.append(all_weekday_windows[j])

    features_tensor = torch.tensor(array_values_all, dtype=torch.long).to(device)
    labels_tensor = torch.tensor(array_label_all, dtype=torch.float).to(device)
    timestamps_tensor = torch.tensor(array_time_all, dtype=torch.long).to(device)
    time_in_day_tensor = torch.tensor(array_time_in_day, dtype=torch.long).to(device)
    time_in_week = torch.tensor(array_weekday, dtype=torch.long).to(device)



    dataset = TensorDataset(features_tensor, timestamps_tensor, time_in_day_tensor, time_in_week, labels_tensor)
    return dataset



'''
pathForTesting = "./helper.xes"
trainDataFrame = pm4py.read.read_xes(pathForTesting)
_, featuresDf = (log_to_features.apply(trainDataFrame, parameters={"str_ev_attr": ["concept:name"]}))


features_to_index = {feature:idx for idx, feature in enumerate(featuresDf)}

features_to_index = {key: value + 1 for key, value in features_to_index.items()}
'''


#initialDataframe, targets = creatingInitialDataframeOneHot(trainDataFrame, features_to_index)
#createSlidingWindow(40, initialDataframe, targets)

#def creatingTensorAndTrainingOneHot(wholeDataFrame, featuresDfInput, sliding_window):
#creatingTensorAndTrainingOneHot(trainDataFrame, features_to_index, 40)

