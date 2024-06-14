import torch
import numpy as np
import pm4py
from sklearn.linear_model import LinearRegression
from pm4py.algo.transformation.log_to_features import algorithm as log_to_features
from creatingOneHotEncoding import creatingTensorsTrainingAndTesting
from torch import nn
import pickle
from allVariables import batch_size, num_epochs, learning_rate, embedding_dim, slidingWindow, helperDataset, path_train, path_test, device


path_train = './BPIC/BPI_Challenge_2012_train.xes'
path_test = './BPIC/BPI_Challenge_2012_test.xes'

if (__name__ == "__main__"):

    trainDataFrame = pm4py.read.read_xes(path_train)
    testDataFrame = pm4py.read.read_xes(path_test)
    _, featuresDf = log_to_features.apply(trainDataFrame, parameters={"str_ev_attr": ["concept:name"]})
    input_len = len(featuresDf)

    features_to_index = {feature:idx for idx, feature in enumerate(featuresDf)}

    features_to_index = {key: value + 1 for key, value in features_to_index.items()}

    datasetTraining = creatingTensorsTrainingAndTesting(trainDataFrame, features_to_index, sliding_window=slidingWindow)

    #combined_tensor = torch.cat((datasetTraining.tensors[0], datasetTraining.tensors[1]), dim=1)

    inputValueTime = datasetTraining.tensors[1].detach().numpy()
    prediction = datasetTraining.tensors[4].detach().numpy()

    rf = LinearRegression()

    
    '''
        Training
    '''
    rf.fit(inputValueTime, prediction) 


    with open('./generatingModels/regression_2012.pickle', 'wb') as f:
        pickle.dump(rf, f)

    '''
        Testing
    '''
    datasetTesting = creatingTensorsTrainingAndTesting(testDataFrame, features_to_index, sliding_window=slidingWindow)
    inputValueTimeTesting = datasetTesting.tensors[1].detach().numpy()


    all_results = rf.predict(inputValueTimeTesting)

    total_absolute_error = 0
    number_of_samples = len(all_results)

    true_label = np.repeat(datasetTesting.tensors[2].detach().numpy(), inputValueTimeTesting.shape[1])

    for i in range(number_of_samples):
        total_absolute_error += np.abs(all_results[i] - true_label[i])

    mae = total_absolute_error / number_of_samples

    print("MAE")
    print(mae)

'''
    For inferencing
'''
def predictionOnTensorLinearRegression(inputTime):

    with open('./generatingModels/regression_2012.pickle', 'rb') as f:
        rf = pickle.load(f)

    all_results = rf.predict(inputTime)

    return all_results