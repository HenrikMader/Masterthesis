import torch
import numpy as np
import pm4py
from sklearn.ensemble import RandomForestRegressor
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

    combined_tensor = torch.cat((datasetTraining.tensors[0], datasetTraining.tensors[1]), dim=1)

    prediction = datasetTraining.tensors[4].detach().numpy()

    rf = RandomForestRegressor(n_estimators=200, random_state=42, warm_start=True)


    '''
        Training
    '''
    rf.fit(combined_tensor, prediction) 


    with open('./generatingModels/randomForest_2012.pickle', 'wb') as f:
        pickle.dump(rf, f)

    '''
        Testing
    '''
    datasetTesting = creatingTensorsTrainingAndTesting(testDataFrame, features_to_index, sliding_window=slidingWindow)
    #embedded_feature_testing = embedding(datasetTesting.tensors[0])
    combined_tensor_testing = torch.cat((datasetTesting.tensors[0], datasetTesting.tensors[1]), dim=1)

    all_results = rf.predict(combined_tensor_testing)

    total_absolute_error = 0
    number_of_samples = len(all_results)

    true_label = np.repeat(datasetTesting.tensors[2].detach().numpy(), combined_tensor_testing.shape[1])

    for i in range(number_of_samples):
        total_absolute_error += np.abs(all_results[i] - true_label[i])

    mae = total_absolute_error / number_of_samples

    ## Why is the MAE so weird here?
    print("MAE")
    print(mae)

'''
    For inferencing
'''
def predictionOnTensorValueAndTime(inputValue, inputTime):
    X_combined = torch.cat((inputValue, inputTime), dim=-1)

    with open('./generatingModels/randomForest_2012.pickle', 'rb') as f:
        rf = pickle.load(f)

    all_results = rf.predict(X_combined)

    return all_results