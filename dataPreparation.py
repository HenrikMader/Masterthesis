import pm4py
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
from pm4py.algo.transformation.log_to_features import algorithm as log_to_features
from encoding import creatingTensorsTrainingAndTesting
from sklearn.model_selection import KFold
import os
import time
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def load_data(path_train, path_val, path_test, slidingWindow):
    train_df = pm4py.read.read_xes(path_train)
    val_df = pm4py.read.read_xes(path_val)
    test_df = pm4py.read.read_xes(path_test)
    _, features_df = log_to_features.apply(train_df, parameters={"str_ev_attr": ["concept:name"]})
    feature_to_index = {feature: idx + 1 for idx, feature in enumerate(features_df)}
    print("Features to index")
    print(feature_to_index)

    train_dataset = creatingTensorsTrainingAndTesting(train_df, feature_to_index, sliding_window=slidingWindow)
    torch.save(train_dataset, "./train.pt")
    test_dataset = creatingTensorsTrainingAndTesting(test_df, feature_to_index, sliding_window=slidingWindow)
    torch.save(test_dataset, "./test.pt")
    val_dataset =  creatingTensorsTrainingAndTesting(val_df, feature_to_index, sliding_window=slidingWindow)
    torch.save(val_dataset, "./val.pt")
    
    return train_dataset, val_dataset, test_dataset, len(features_df), feature_to_index


def prepare_data_ensemble(meta_features, meta_var, meta_labels, batch_size, shuffle):
    meta_features = torch.tensor(meta_features)
    meta_var = torch.tensor(meta_var)
    meta_labels = torch.tensor(meta_labels)

    dataset = TensorDataset(meta_features, meta_var, meta_labels)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader


def prepare_data_ensemble_no_uncertainty(meta_features, meta_labels, batch_size, shuffle):
    meta_features = torch.tensor(meta_features)
    meta_labels = torch.tensor(meta_labels)
    dataset = TensorDataset(meta_features, meta_labels)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader


