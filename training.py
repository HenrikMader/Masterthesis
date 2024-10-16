import torch
from sklearn.metrics import mean_absolute_error
import numpy as np
from torch import nn, optim
import pm4py
import pickle
import matplotlib.pyplot as plt
from torch.nn.utils import clip_grad_norm_
from regression_uncertainties import epistemic_uncertainty, prediction_avg, aleatoric_uncertainty, uncertainty, prediction_avg_log
from torch.utils.data import DataLoader, TensorDataset, Subset
import time
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')



def regression_loss(mean, true):
    return torch.mean(torch.abs((true - mean)))



## MAE vs RMSE (But always sum as reduction)
def criterionHans(mean, true, log_var):
    precision = torch.exp(-log_var)

    return torch.mean(torch.sum((2 * precision) ** .5 * torch.abs(true - mean) + log_var / 2, 1), 0)


'''
    TODO: Implement Early stopping
'''
class EarlyStopping:
    def __init__(self, path, patience=40, delta=1e-8):
        """
        Args:
            patience (int): How long to wait after last time the monitored metric improved.
                            Default: 5
            delta (float): Minimum change in the monitored metric to qualify as an improvement.
                           Default: 0
            path (str): Path to save the model checkpoint.
        """
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decreases.'''
        torch.save(model, self.path)
        self.val_loss_min = val_loss



loss_func_gaussian = nn.GaussianNLLLoss()


def train_no_uncertainty(num_epochs, model, train_dataloader, val_dataloader, learning_rate):
        model_name = type(model).__name__
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        early_stopping = EarlyStopping(path=model_name, patience=num_epochs)

        for epoch in range(num_epochs):
            model.train()
            total_train_loss = 0

            for batch_idx, (inputs_value, inputs_time, targets) in enumerate(train_dataloader):
                inputs_value, inputs_time, targets = \
                    inputs_value.to(device), inputs_time.to(device), targets.to(device)
                optimizer.zero_grad()
                output, regularizer = model(inputs_value, inputs_time)
                regularizer = regularizer.to(device)
                loss = regression_loss(output, targets) + regularizer
                loss.backward()
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1)
                optimizer.step()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_dataloader)
            if (epoch > 5):
                print(f"Epoch: {epoch + 1}; Average Training Loss: {avg_train_loss:.4f}")
                train_losses.append(avg_train_loss)

            model.eval()
            total_val_loss = 0

            with torch.no_grad():
                for batch_idx, (inputs_value, inputs_time, targets) in enumerate(val_dataloader):
                    inputs_value, inputs_time, targets = \
                        inputs_value.to(device), inputs_time.to(device), targets.to(device)
                    output, _ = model(inputs_value, inputs_time)
                    #regularizer = model.regularizer().to(device)
                    loss = regression_loss(output, targets) #+ regularizer
                    total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(val_dataloader)
            if (epoch > 5):
                val_losses.append(avg_val_loss)
                print(f"Epoch: {epoch + 1}; Average Val Loss: {avg_val_loss:.4f}")

            early_stopping(avg_val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break
            #if avg_val_loss < best_val_loss:
            #    best_val_loss = avg_val_loss
            #    torch.save(model, f"bestModelNoUnc_{model_name}")


        #torch.save(model, f"lastModelNoUnc_{model_name}")
        model = torch.load(model_name)

        #plt.show()

        return model

def train(num_epochs, model, train_dataloader, val_dataloader, learning_rate):
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        best_val_loss = float('inf')
        model_name = type(model).__name__
        early_stopping = EarlyStopping(path=model_name, patience=num_epochs)

        train_losses_avg = []
        val_losses_avg = []

        for epoch in range(num_epochs):
            model.train()
            total_train_loss = 0

            for batch_idx, (inputs_value, inputs_time, targets) in enumerate(train_dataloader):
                inputs_value, inputs_time, targets = inputs_value.to(device), inputs_time.to(device), targets.to(device)

                optimizer.zero_grad()
                output, log_var, regularizer = model(inputs_value, inputs_time)
                regularizer = regularizer.to(device)
                loss = criterionHans(output, targets, log_var) + regularizer


                loss.backward()
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1)
                optimizer.step()
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_dataloader)
            
            if (epoch > 5):
                print(f"Epoch: {epoch + 1}; Average Training Loss: {avg_train_loss:.4f}")
                train_losses_avg.append(avg_train_loss)

            
            model.eval()
            total_val_loss = 0

            with torch.no_grad():
                for batch_idx, (inputs_value, inputs_time, targets) in enumerate(val_dataloader):
                    inputs_value, inputs_time, targets = inputs_value.to(device), inputs_time.to(device), targets.to(device)

                    output, log_var, _ = model(inputs_value, inputs_time)

                    loss = criterionHans(output, targets, log_var)
                    total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(val_dataloader)
            
            if (epoch > 5):
                print(f"Epoch: {epoch + 1}; Average Val Loss: {avg_val_loss:.4f}")
                val_losses_avg.append(avg_val_loss)

            ## Implementation of early stopping
            early_stopping(avg_val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break

        #torch.save(model, f"lastModel_{model_name}")
        model = torch.load(model_name)

        #plt.show()

        return model

## Get the activity before
def createSet_baselineModel(pathTrain):
    # Load the data


    ### Check if right
    trainDataFrame = pm4py.read.read_xes(pathTrain)

    # Group the data by the 'concept:name' column
    groupedTrainDataFrame = trainDataFrame.groupby("concept:name")
    
    # Calculate the mean and standard deviation for 'remain_time' for each group
    mean_series = groupedTrainDataFrame["remain_time"].mean()
    std_series = groupedTrainDataFrame["remain_time"].var()  # For standard deviation


    # Combine the mean and std into a dictionary with tuples
    data_dict = {key: (mean_series[key], std_series[key]) for key in mean_series.index}

    return data_dict



def createSet_prefix_baseline(pathTrain):
    
    train_dataFrame = pm4py.read.read_xes(pathTrain)




    helper = pm4py.get_prefixes_from_log(train_dataFrame, length=1000)
    groupedTrainDataFrame = helper.groupby("@@index_in_trace")

    mean_series = groupedTrainDataFrame["remain_time"].mean()
    std_series = groupedTrainDataFrame["remain_time"].var()
    print("mean series")
    print(mean_series)
    print("std series")
    print(std_series)

    std_series = std_series.fillna(0)

    print(std_series)
    time.sleep(3)

    data_dict = {key: (mean_series[key], std_series[key]) for key in mean_series.index}

    return data_dict
'''
    Currently only aleatoric uncertainty!
'''


def generate_meta_features(model, dataloader, device):
    model.train()
    allResults = []
    allTargets = []
    allVariancesEpistemic = []
    allVariancesAleatoric = []
    allVariancesOverall = []

    with torch.no_grad():
        for inputs_value, inputs_time, targets in dataloader:
            inputs_value, inputs_time, targets = inputs_value.to(device), inputs_time.to(device), targets.to(device)
            # Not the standard deviation currently!
            out_mean, out_var = model.sample(125, inputs_value.size(0), inputs_value, inputs_time)
            E = epistemic_uncertainty(out_mean)
            predictions = prediction_avg_log(out_mean)
            A = aleatoric_uncertainty(out_var)
            uncertainties = uncertainty(out_mean, out_var)



            # Convert tensors to numpy arrays
            E = E.detach().cpu().numpy()
            A = A.detach().cpu().numpy()


            ## Is this necessary?
            predictions = predictions.flatten().detach().cpu().numpy()
            allVariances = uncertainties.detach().cpu().numpy()
            targets = targets.detach().cpu().numpy()

            # Append predictions to the list
            allResults.append(predictions)
            allTargets.append(targets)
            allVariancesEpistemic.append(E)
            allVariancesAleatoric.append(A)
            allVariancesOverall.append(allVariances)
    
    allResults_combined = torch.tensor(np.concatenate(allResults)).unsqueeze(1)
    allTargets_combined = torch.tensor(np.concatenate(allTargets))
    allVariancesEpistemic_combined = torch.tensor(np.concatenate(allVariancesEpistemic))
    allVariancesAleatoric_combined = torch.tensor(np.concatenate(allVariancesAleatoric))
    allVariances_combined = torch.tensor(np.concatenate(allVariancesOverall))


    

    return allResults_combined, allVariances_combined, allTargets_combined, allVariancesAleatoric_combined, allVariancesEpistemic_combined



def generate_meta_features_no_unct(model, dataloader, device):
    model.to(device)
    model.eval()
    meta_features = []
    meta_labels = []

    with torch.no_grad():
        for inputs_value, inputs_time, targets in dataloader:
            inputs_value, inputs_time, targets = inputs_value.to(device), inputs_time.to(device), targets.to(device)
            output, _ = model(inputs_value, inputs_time)
            #output = torch.clamp(output, min=0)
            meta_features.append(output.cpu().cpu().numpy())
            meta_labels.append(targets.cpu().cpu().numpy())


    print(np.vstack(meta_features))
    print(np.vstack(meta_labels))
    return np.vstack(meta_features), np.vstack(meta_labels)



def generateMeta_features_baseline(dictionary_baseline, features_to_index, dataloader):
    # Prepare dictionary_baseline and inverse dictionary
    dictionary_baseline = {key.replace(" ", ""): value for key, value in dictionary_baseline.items()}
    inverse_dict = {v: k for k, v in features_to_index.items()}

    # Initialize lists to accumulate results
    all_predictions = []
    all_variances = []
    all_targets = []



    with torch.no_grad():
        for (inputs_value, inputs_time, targets) in dataloader:

            # Extract labels
            activity_before_current = inputs_value[:, -2]

            # Map labels to string and trim them
            activity_before_current = [inverse_dict[item.item()] for item in activity_before_current]

            # Get rid of the concept name
            activity_before_current_trimmed = [label[13:] for label in activity_before_current]

            # Extract the first and second values from the tuples corresponding to the mapped labels
            prediction_from_dictionary = [dictionary_baseline.get(label, (0,))[0] for label in activity_before_current_trimmed]
            variance_from_dictionary = [dictionary_baseline.get(label, (0,))[1] for label in activity_before_current_trimmed]

            # Append results to lists
            all_predictions.extend(prediction_from_dictionary)
            all_variances.extend(variance_from_dictionary)
            all_targets.extend(targets)

    # Convert accumulated lists to tensors
    prediction_tensor = torch.tensor(all_predictions, dtype=torch.float32).unsqueeze(1)
    variance_tensor = torch.tensor(all_variances, dtype=torch.float32).unsqueeze(1)
    #variance_tensor = torch.log(variance_tensor)
    all_targets = torch.tensor(all_targets, dtype=torch.float32).unsqueeze(1)


    return prediction_tensor, variance_tensor, all_targets


def generateMeta_features_prefix_baseline(dictionary_prefix_baseline, dataloader):
    all_predictions = []
    all_variances = []
    all_targets = []

    with torch.no_grad():
        for (inputs_value, inputs_time, targets) in dataloader:
            targetsHelper = targets.flatten()
            count_non_zero_per_row = torch.count_nonzero(inputs_value, dim=1)
            all_targets.extend(targets)

            for index, count in enumerate(count_non_zero_per_row):
                count = count.item()
                if count in dictionary_prefix_baseline:
                    value = dictionary_prefix_baseline[count]

                    all_predictions.extend([value[0]])
                    all_variances.extend([value[1]])
                    #all_targets.extend(targets)
                else:
                    all_predictions.extend([0])
                    all_variances.extend([0])


    all_predictions = torch.tensor(all_predictions, dtype=torch.float32).unsqueeze(1)
    all_variances = torch.tensor(all_variances, dtype=torch.float32).unsqueeze(1)
    all_targets = torch.tensor(all_targets, dtype=torch.float32).unsqueeze(1)

    return all_predictions, all_variances, all_targets




def train_ensemble(num_epochs, model, train_dataloader, val_dataloader, loss_func, learning_rate, log_var = False):
        model_name = type(model).__name__
        early_stopping = EarlyStopping(path=model_name)
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            model.train()
            total_train_loss = 0

            for batch_idx, (meta_mean, meta_var, targets) in enumerate(train_dataloader):
                meta_mean, meta_var, targets = meta_mean.to(device), meta_var.to(device), targets.to(device)


                optimizer.zero_grad()

                output = model(meta_mean, meta_var).to(device)
                loss = regression_loss(output, targets)
                loss.backward()
                #torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1)
                optimizer.step()

                total_train_loss += loss.item()
            

            avg_train_loss = total_train_loss / len(train_dataloader)
            if (epoch % 5 == 0):
                print(f"Epoch: {epoch + 1}; Average train Loss: {avg_train_loss:.4f}")
                train_losses.append(avg_train_loss)

            model.eval()
            total_val_loss = 0

            with torch.no_grad():
                for batch_idx, (meta_mean, meta_var, targets) in enumerate(val_dataloader):
                    meta_mean, meta_var, targets = meta_mean.to(device), meta_var.to(device), targets.to(device)

                    output = model(meta_mean, meta_var)
                    loss = regression_loss(output, targets)
                    total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(val_dataloader)
            
            if (epoch % 5 == 0):
                print(f"Epoch: {epoch + 1}; Average Val Loss: {avg_val_loss:.4f}")
                val_losses.append(avg_val_loss)

            early_stopping(avg_val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break


        #torch.save(model, "lastModel_ensemble")
        model = torch.load(model_name)
        return model

def train_ensemble_no_uncertainty(num_epochs, model, train_dataloader, val_dataloader, loss_func, learning_rate):
        model_name = type(model).__name__
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        early_stopping = EarlyStopping(path=model_name)

        for epoch in range(num_epochs):
            model.train()
            total_train_loss = 0

            for batch_idx, (meta_mean, targets) in enumerate(train_dataloader):


                meta_mean, targets = meta_mean.to(device), targets.to(device)


                optimizer.zero_grad()

                output = model(meta_mean).to(device)
                loss = regression_loss(output, targets)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_dataloader)
            train_losses.append(avg_train_loss)
            if (epoch % 5 == 0):
                print(f"Epoch: {epoch + 1}; Average train Loss: {avg_train_loss:.4f}")

            model.eval()
            total_val_loss = 0

            with torch.no_grad():
                for batch_idx, (meta_mean, targets) in enumerate(val_dataloader):
                    meta_mean, targets = meta_mean.to(device), targets.to(device)

                    output = model(meta_mean)
                    loss = regression_loss(output, targets)
                    total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(val_dataloader)
            val_losses.append(avg_val_loss)
            
            if (epoch % 30 == 0):
                print(f"Epoch: {epoch + 1}; Average Val Loss: {avg_val_loss:.4f}")

            early_stopping(avg_val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break

        model = torch.load(model_name)

        return model