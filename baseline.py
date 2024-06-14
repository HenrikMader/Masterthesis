from pm4py.algo.transformation.log_to_features import algorithm as log_to_features    
import pm4py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



#### Filter all Nan Values

def calculateArrayAverageTime(trainDataFrame, testDataFrame):

    #grouped_train = trainDataFrame.groupby('case:concept:name')
    #grouped_test = testDataFrame.groupby('case:concept:name')

    #def drop_last(group):
    #    return group.iloc[:-1]
    
    #trainDataFrame = grouped_train.apply(drop_last).reset_index(drop=True)
    #testDataFrame = grouped_test.apply(drop_last).reset_index(drop=True)

    predictionsOfRemainingTimeOnTrain = []

    helper = pm4py.get_prefixes_from_log(trainDataFrame, length=1000)
    helper_two = helper.groupby('@@index_in_trace')['remain_time'].mean()
    print(helper_two)

    ## Wieso nichtmehr alle dabei?
    helper_test = pm4py.get_prefixes_from_log(testDataFrame, length=1000)


    predictionOfRemainingTimeOnTest = [] 

    # For training samples
    for index, row in helper_test.iterrows():
        #print(helper_two)
        if (row["@@index_in_trace"] <= helper_two.idxmax()):
            remaingTimeIndex = helper_two.loc[row["@@index_in_trace"]]
            predictionOfRemainingTimeOnTest.append(remaingTimeIndex)
        else:
            predictionOfRemainingTimeOnTest.append(0)

    # For testing samples
    for index, row in helper.iterrows():
        remaingTimeIndex = helper_two.loc[row["@@index_in_trace"]]
        predictionsOfRemainingTimeOnTrain.append(remaingTimeIndex)
    
    return predictionOfRemainingTimeOnTest, predictionsOfRemainingTimeOnTrain



trainDataFrame = pm4py.read.read_xes("./BPIC/RequestForPayment_train.xes")
testDataFrame = pm4py.read.read_xes("./BPIC/RequestForPayment_test.xes")
predictionTest, predictionTrian = calculateArrayAverageTime(trainDataFrame, testDataFrame)

#helper = pm4py.get_prefixes_from_log(testDataFrame, length=1000)
#helperGrouped = helper.groupby("case:concept:name").max("@@index_in_trace")
#cases = helperGrouped[helperGrouped["@@index_in_trace"] > 15]
#testDataFrame = testDataFrame[testDataFrame[""]]


absolute_errors = np.abs(predictionTest - testDataFrame["remain_time"])  # Compute the absolute errors


mae = np.mean(absolute_errors)  # Ignoring Nan values

print("Mean Absolute Error (MAE):", mae)

#calculateArrayAverageTime(trainDataFrame, testDataFrame)
