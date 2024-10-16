from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import torch
from regression_uncertainties import epistemic_uncertainty, prediction_avg, aleatoric_uncertainty, uncertainty
import time
from sklearn.metrics import mean_absolute_error
import pandas as pd

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')


def test_loop_no_uncertainty(dataloader: DataLoader, model: torch.nn.Module):
        model.eval()
        total_absolute_error = 0
        numberOfSamples = 0
        index_dict = {}
        allResults = []
        allTargets = []

        with torch.no_grad():
            for (inputs_value, inputs_time, targets) in dataloader:
                inputs_value, inputs_time, targets = inputs_value.to(device), inputs_time.to(device), targets.to(device)
                pred, _ = model(inputs_value, inputs_time)

                #pred = torch.clamp(pred, min=0)
                #variance = torch.exp(variance)
                pred = pred.flatten().cpu()
                targets = targets.flatten().cpu()
                #variance = torch.exp(variance)
                allResults = np.concatenate((allResults, pred))
                allTargets = np.concatenate((allTargets, targets))

                absolute_diff = torch.abs(targets.squeeze() - pred.squeeze())
                
                total_absolute_error += torch.sum(absolute_diff).item()
                
                numberOfSamples += targets.numel()



        # Create an array of indices
        indices = np.arange(len(allResults))


        # Adding labels and title
        plt.xlabel('Index')
        plt.ylabel('Value')

        #plt.show()

        mae = total_absolute_error / numberOfSamples
        print(f'Mean Absolute Error: {mae:.4f}')
        
        return mae

def test_random_forest_with_uncertainty(randomForestModel, testLoader, featuresToIndex):
    # Lists to store features and labels
    X_test = []
    y_test = []
    
    # Iterate through the DataLoader
    for batch_idx, (inputs_value, inputs_time, targets) in enumerate(testLoader):
        # Count non-zero items in inputs_value
        non_zero_counts = torch.sum(inputs_value != 0, dim=1).cpu().numpy()  # Shape: (batch_size,)
        
        # Take the last entry from inputs_value
        lastItem = inputs_value[:, -1].cpu().numpy()
        n_classes = len(featuresToIndex)
        one_hot_encoded = np.eye(n_classes)[np.array(lastItem) - 1]

        # Take the last entry from inputs_time
        last_entry_times = inputs_time[:, -1].unsqueeze(dim=1).cpu().numpy()  # Shape: (batch_size,)

        # Combine one-hot encoded features and time-related features
        #batch_features = np.hstack([one_hot_encoded, last_entry_times])

        #final_data = pd.concat([one_hot_encoded, last_entry_times], axis=1)

        combined_features = np.concatenate((one_hot_encoded, last_entry_times), axis=1)


        
        # Append features and labels to lists
        X_test.append(one_hot_encoded)
        y_test.append(targets.cpu().numpy())
    
    # Convert lists to numpy arrays for testing
    X_test = np.concatenate(X_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)
    
    # Get predictions from each tree in the random forest
    all_tree_predictions = np.array([tree.predict(X_test) for tree in randomForestModel.estimators_])

    # Calculate the mean prediction across all trees (this is the final prediction)
    predictions = np.mean(all_tree_predictions, axis=0)

    # Calculate uncertainty (standard deviation of predictions across trees)
    uncertainty = np.std(all_tree_predictions, axis=0)



    percentiles = np.percentile(uncertainty, np.arange(5, 101, 10))  # 5%, 10%, 20%, ..., 100%
    
    # Initialize list to store MAE for each percentile range
    mae_percentiles = []

    mae = mean_absolute_error(y_test, predictions)
    
    # Print results
    print("Mean Absolute Error (MAE):", mae)
    print("Uncertainty (Standard Deviation of Predictions):", uncertainty)


    '''
        See how good the correlation is!
    '''
    # Calculate MAE for each percentile range
    for i in range(len(percentiles)):
        if i == 0:
            mask = uncertainty <= percentiles[i]
        else:
            mask = (uncertainty > percentiles[i-1]) & (uncertainty <= percentiles[i])
        
        if np.any(mask):
            mae = mean_absolute_error(y_test[mask], predictions[mask])
            mae_percentiles.append((percentiles[i], mae))
        else:
            mae_percentiles.append((percentiles[i], np.nan))
    
    # Print results
    print("Percentiles of Uncertainty and Corresponding MAE:")
    for percentile, mae in mae_percentiles:
        print(f"Percentile <= {percentile:.2f}%: MAE = {mae:.4f}")



    # Optional: Calculate prediction intervals (e.g., 95% interval)
    
    return mae, uncertainty


def test_loop(dataloader: DataLoader, model: torch.nn.Module, printing = False):
        model.train()
        total_absolute_error = 0
        numberOfSamples = 0
        index_dict = {}
        allResults = []
        allTargets = []
        allVariancesEpistemic = []
        allVariancesAleatoric = []
        allVariancesOverall = []

        ## Look at process step
        dictionaryMAE = {}
        dictionaryUncertainty = {}
        dictionaryCounter = {}

        counter = 3
        with torch.no_grad():
            for (inputs_value, inputs_time, targets) in dataloader:
                inputs_value, inputs_time, targets = inputs_value.to(device), inputs_time.to(device), targets.to(device)

                non_zero_counts = torch.count_nonzero(inputs_value, dim=1)


                # Sample from the model
                out_mean, out_var = model.sample(125, inputs_value.size(0), inputs_value, inputs_time)

                E = epistemic_uncertainty(out_mean)
                predictions = prediction_avg(out_mean)
                A = aleatoric_uncertainty(out_var)
                uncertainties = uncertainty(out_mean, out_var)

                # Convert tensors to numpy arrays
                E = E.detach().cpu().numpy()
                A = A.detach().cpu().numpy()
                predictions = predictions.flatten().detach().cpu().numpy()
                allVariances = uncertainties.detach().cpu().numpy()
                targets = targets.detach().cpu().numpy()


                helper = np.array(targets.flatten(), dtype=np.float32)
                helper2 = np.array(predictions, dtype=np.float32)
                helper3 = allVariances.flatten()
                helper4 = non_zero_counts.detach().cpu().numpy()

                absolute_error = np.abs(helper - helper2)


                for j in range(len(helper4)):
                    step = helper4[j]
                    if step not in dictionaryMAE:
                        dictionaryMAE[step] = 0.0
                        dictionaryUncertainty[step] = 0.0
                        dictionaryCounter[step] = 0
                    dictionaryMAE[step] += absolute_error[j]  # Assuming each index corresponds to a batch step
                    dictionaryUncertainty[step] += helper3[j]  # Sum variances
                    dictionaryCounter[step] += 1


                # Append predictions to the list
                allResults.append(predictions)
                allTargets.append(targets)
                allVariancesEpistemic.append(E)
                allVariancesAleatoric.append(A)
                allVariancesOverall.append(allVariances)

        # Concatenate all arrays into a single array
        allResults_combined = np.concatenate(allResults)
        allTargets_combined = np.concatenate(allTargets)
        allVariancesEpistemic_combined = np.concatenate(allVariancesEpistemic)
        allVariancesAleatoric_combined = np.concatenate(allVariancesAleatoric)
        allVariances_combined = np.concatenate(allVariancesOverall)
        allTargets_combined = allTargets_combined.flatten()
        allVariancesEpistemic_combined = allVariancesEpistemic_combined.flatten()

        indices = np.arange(len(allResults_combined))

        ########### Plot the uncertainties and MAE ###########


        # Initialize a new dictionary to store averages
        average_dict_mae = {}
        average_dict_unc = {}


        # Iterate through the keys in sum_dict
        for key in dictionaryCounter:
            if key in dictionaryMAE:
                # Compute the average for each step
                average = dictionaryMAE[key] / dictionaryCounter[key]
                average_dict_mae[key] = average


        for key in dictionaryCounter:
            if key in dictionaryUncertainty:
                # Compute the average for each step
                average = dictionaryUncertainty[key] / dictionaryCounter[key]
                average_dict_unc[key] = average


            



        # Extract the keys and values
        keys = list(dictionaryCounter.keys())
        values1 = [average_dict_mae[key] for key in keys]
        values2 = [average_dict_unc[key] for key in keys]
        ##plt.show()






        #######################################################

        #plt.show()

        
        def calculate_mae(predictions, targets):
            return np.mean(np.abs(predictions - targets))
        
        
        
        '''
            Stuff for Overall
        '''
        print("Overall")
        # Calculate the percentiles for the variance
        percentiles = np.percentile(allVariances_combined, np.arange(5, 101, 5))  # 10%, 20%, ..., 100%
        print(percentiles)
        # Calculate MAE for each percentile range
        mae_percentiles = []
        for percentile in percentiles:
            mask = allVariances_combined <= percentile
            mask = mask.flatten()
            
            if np.any(mask):
                mae = calculate_mae(allResults_combined[mask], allTargets_combined[mask])
                mae_percentiles.append((percentile, mae))
            else:
                mae_percentiles.append((percentile, np.nan))

        for (percentile, mae) in mae_percentiles:
                print(f"MAE for variance below {percentile:.2f}: {mae:.4f}")


        if (printing == True):

            percentiles_values = [p for p, _ in mae_percentiles]
            mae_values = [mae for _, mae in mae_percentiles]

            percentiles_values = np.arange(5, 101, 5)
            print(percentiles_values)
            percentiles_values = 100 - percentiles_values
            mae_values = mae_values[::-1]
            percentiles_values = percentiles_values[::-1]


            x_min, x_max = np.min(percentiles_values), np.max(percentiles_values)
            x_range = x_max - x_min

            # Calculate the height of the y-axis range
            y_min, y_max = np.min(mae_values), np.max(mae_values)
            y_range = y_max - 0

            # Calculate the total area of the bounding box
            total_area = x_range * y_range



            auc = np.trapz(mae_values, percentiles_values)

            print(f"AUC (Area under the curve): {auc}")
            print(f"whole area {total_area}")
            print(f"normalized {auc / total_area}")




            

            # Plot the Accuracy-Rejection curve
            plt.figure(figsize=(10, 6))
            plt.plot(percentiles_values, mae_values, marker='o', linestyle='-', color='b')
            plt.xlabel('Rejection in %')
            plt.ylabel('Mean Absolute Error (MAE)')
            plt.title('Accuracy-Rejection Curve')
            plt.grid(True)
            plt.show()

        mae = calculate_mae(allResults_combined, allTargets_combined)
        print("MAE overall")
        print(mae)
        return mae



def test_loop_average(dataloader):
    total_absolute_error = 0.0
    total_elements = 0
    
    with torch.no_grad():
        for (inputs_value, targets) in dataloader:
            # Calculate the average across dim=1 for each batch
            average = inputs_value.mean(dim=1, keepdim=True)
            
            # Calculate the absolute error for each element
            absolute_error = torch.abs(average - targets)
            
            # Accumulate the total absolute error
            total_absolute_error += absolute_error.sum().item()
            
            # Accumulate the total number of elements
            total_elements += targets.numel()

    # Calculate the overall MAE across all elements
    maeOverall = total_absolute_error / total_elements
    print("MAE overall average:", maeOverall)
    
    return maeOverall





def test_loop_ensemble(dataloader: DataLoader, model: torch.nn.Module, log_var = False):
        model.eval()
        total_absolute_error = 0
        numberOfSamples = 0
        index_dict = {}
        allResults = []
        allTargets = []

        #dataloader = transform_data(dataloader, shuffle=False)

        with torch.no_grad():
            for (mean, var, targets) in dataloader:
                
                mean, var, targets = mean.to(device), var.to(device), targets.to(device)

                pred = model(mean, var)
                pred = pred.flatten().cpu()
                targets = targets.flatten().cpu()
                allResults = np.concatenate((allResults, pred))
                allTargets = np.concatenate((allTargets, targets))

                absolute_diff = torch.abs(targets.squeeze() - pred.squeeze())
                
                total_absolute_error += torch.sum(absolute_diff).item()
                
                numberOfSamples += targets.numel()

        mae = total_absolute_error / numberOfSamples
        print(f'Mean Absolute Error: {mae:.4f}')
        
        return mae



def test_loop_ensemble_no_uncertainty(dataloader: DataLoader, model: torch.nn.Module):
        model.eval()
        total_absolute_error = 0
        numberOfSamples = 0
        index_dict = {}
        allResults = []
        allTargets = []

        with torch.no_grad():
            for (mean, targets) in dataloader:
                mean, targets = mean.to(device), targets.to(device)
                
                pred = model(mean)
                pred = pred.flatten().cpu()
                targets = targets.flatten().cpu()
                #variance = torch.exp(variance)
                allResults = np.concatenate((allResults, pred))
                allTargets = np.concatenate((allTargets, targets))

                absolute_diff = torch.abs(targets.squeeze() - pred.squeeze())
                
                total_absolute_error += torch.sum(absolute_diff).item()
                
                numberOfSamples += targets.numel()



        # Create an array of indices
        indices = np.arange(len(allResults))


        #plt.show()

        mae = total_absolute_error / numberOfSamples
        print(f'Mean Absolute Error: {mae:.4f}')
        
        return mae

#### Doesnt work, since I am not allowed to see test dataset
def test_loop_weighted_average(dataloader):
    total_absolute_differences = 0
    total_samples = 0
    
    with torch.no_grad():
        for (mean, var, targets) in dataloader:
            # Sum the variances across models
            totalVariance = var.sum(dim=1, keepdim=True)


            # Compute the weight for each model as the inverse of the normalized variance
            varianceEachModel = var / (totalVariance)
            varianceEachModel = varianceEachModel + 1e-6
            inverseEachModel = 1.0 / varianceEachModel

            # Normalize weights so that they sum to 1
            weights = inverseEachModel / inverseEachModel.sum(dim=1, keepdim=True)

            # Compute the weighted average
            weightedAverage = (mean * weights).sum(dim=1, keepdim=True)
            # Calculate absolute differences between weighted average and targets
            absolute_differences = torch.abs(weightedAverage - targets)
            
            # Accumulate total absolute differences
            total_absolute_differences += absolute_differences.sum()
            total_samples += targets.size(0)  # Accumulate the number of samples
    
    # Compute overall MAE as the total absolute differences divided by the total number of samples
    maeOverall = total_absolute_differences / total_samples
    print("MAE overall weighted average", maeOverall)
    return maeOverall



def test_loop_min_uncertainty(dataloader):
    total_absolute_differences = 0
    total_samples = 0
    
    with torch.no_grad():
        for (mean, var, targets) in dataloader:
            _, min_variance_indices = torch.min(var, dim=1)
            min_variance_indices = min_variance_indices.unsqueeze(1)
            selected_predictions = torch.gather(mean, dim=1, index=min_variance_indices)
            absolute_differences = torch.abs(selected_predictions - targets)
            
            total_absolute_differences += absolute_differences.sum()
            total_samples += targets.size(0)  # Accumulate the number of samples in this batch
    
    maeOverall = total_absolute_differences / total_samples
    print("MAE overall min uncertainty", maeOverall)
    return maeOverall


def test_baseline(dictionary_baseline, features_to_index, dataloader, printing=False):
    # Prepare dictionary_baseline and inverse dictionary

    dictionary_baseline = {key.replace(" ", ""): value for key, value in dictionary_baseline.items()}
    inverse_dict = {v: k for k, v in features_to_index.items()}

    # Initialize variables to accumulate MAE
    total_loss = 0
    num_batches = 0

    allPrediction = []
    allTarget = []
    allUncertainty = []

    with torch.no_grad():
        for (inputs_value, inputs_time, targets) in dataloader:
            # Extract labels
            activity_before_current = inputs_value[:, -2]

            # Map labels to string and trim them
            activity_before_current = [inverse_dict[item.item()] for item in activity_before_current]

            ## Get rid of the concept name
            activity_before_current_trimmed = [label[13:] for label in activity_before_current]

            # Extract the first values from the tuples corresponding to the mapped labels
            prediction_from_dictionary = [dictionary_baseline.get(label, (0,))[0] for label in activity_before_current_trimmed]

            # Convert to tensor and ensure it has the correct shape
            prediction = torch.tensor(prediction_from_dictionary, device=inputs_value.device)
            prediction = prediction.unsqueeze(1)

            # Calculate the MAE for this batch
            batch_loss = torch.abs(prediction - targets).mean()

            uncertainty_from_dictionary = [dictionary_baseline.get(label, (1,))[0] for label in activity_before_current_trimmed]

            # Convert to tensor and ensure it has the correct shape
            uncertainty_from_dictionary = torch.tensor(uncertainty_from_dictionary, device=inputs_value.device)
            uncertainty_from_dictionary = uncertainty_from_dictionary.unsqueeze(1)

            allPrediction.append(prediction)
            allUncertainty.append(uncertainty_from_dictionary)
            allTarget.append(targets)


            
            # Accumulate the loss and count the batches
            total_loss += batch_loss.item()
            num_batches += 1

    
    def calculate_mae(predictions, targets):
            return np.mean(np.abs(predictions - targets))
    

    combinedPrediction = torch.cat(allTarget, dim=0).cpu().numpy()
    combinedTarget = torch.cat(allPrediction, dim=0).cpu().numpy()

    allUncertainty = torch.cat(allUncertainty, dim=0).cpu().numpy()

    percentiles = np.percentile(allUncertainty, np.arange(5, 101, 5))  # 10%, 20%, ..., 100%
    print(percentiles)
    # Calculate MAE for each percentile range
    mae_percentiles = []
    for percentile in percentiles:
        mask = allUncertainty <= percentile
        mask = mask.flatten()
        
        if np.any(mask):
            mae = calculate_mae(combinedPrediction[mask], combinedTarget[mask])
            mae_percentiles.append((percentile, mae))
        else:
            mae_percentiles.append((percentile, np.nan))


    ## Calculate the AUC_ROC

    if (printing == True):

            percentiles_values = [p for p, _ in mae_percentiles]
            mae_values = [mae for _, mae in mae_percentiles]

            percentiles_values = np.arange(5, 101, 5)
            print(percentiles_values)
            percentiles_values = 100 - percentiles_values
            mae_values = mae_values[::-1]
            percentiles_values = percentiles_values[::-1]

            x_min, x_max = np.min(percentiles_values), np.max(percentiles_values)
            x_range = x_max - x_min

            # Calculate the height of the y-axis range
            y_min, y_max = np.min(mae_values), np.max(mae_values)
            y_range = y_max - 0

            # Calculate the total area of the bounding box
            total_area = x_range * y_range



            auc = np.trapz(mae_values, percentiles_values)

            print(f"AUC (Area under the curve): {auc}")
            print(f"whole area {total_area}")
            print(f"normalized {auc / total_area}")




            

            # Plot the Accuracy-Rejection curve
            plt.figure(figsize=(10, 6))
            plt.plot(percentiles_values, mae_values, marker='o', linestyle='-', color='b')
            plt.xlabel('Rejection in %')
            plt.ylabel('Mean Absolute Error (MAE)')
            plt.title('Accuracy-Rejection Curve')
            plt.grid(True)
            plt.show()




    # Print MAE for each percentile range
    for (percentile, mae) in mae_percentiles:
        print("here")
        print(f"MAE for variance below {percentile:.2f}: {mae:.4f}")





    time.sleep(5)

    # Compute the average MAE
    mae_overall = total_loss / num_batches
    print("MAE over all batches:", mae_overall)

    return mae_overall



def test_baseline_prefix(dictionary_baseline_prefix, dataloader):
    overAllError = 0
    numberItems = 0

    allPredictions = []
    allTargets = []
    allUncertainty = []
    with torch.no_grad():
        for (inputs_value, inputs_time, targets) in dataloader:
            targetsHelper = targets.flatten()
            count_non_zero_per_row = torch.count_nonzero(inputs_value, dim=1)

            for index, count in enumerate(count_non_zero_per_row):
                count = count.item()
                if count in dictionary_baseline_prefix:
                    value = dictionary_baseline_prefix[count]

                    allPredictions.append(torch.tensor(value[0]).unsqueeze(0))
                    allUncertainty.append(torch.tensor(value[1]).unsqueeze(0))
                    allTargets.append(torch.tensor(targetsHelper[index].item()).unsqueeze(0))

                    overAllError += np.abs(value[0] - targetsHelper[index].item())
                    numberItems = numberItems + 1
    



    mae = overAllError / numberItems
    print("MAE baseline prefix")
    print(mae)


    combinedPrediction = torch.cat(allTargets, dim=0).cpu().numpy()
    combinedTarget = torch.cat(allPredictions, dim=0).cpu().numpy()

    allUncertainty = torch.cat(allUncertainty, dim=0).cpu().numpy()

    percentiles = np.percentile(allUncertainty, np.arange(5, 101, 10))  # 10%, 20%, ..., 100%
    print("percentiles here")
    print(percentiles)
    # Calculate MAE for each percentile range
    mae_percentiles = []
    for percentile in percentiles:
        mask = allUncertainty <= percentile
        mask = mask.flatten()
        
        if np.any(mask):
            mae = calculate_mae(combinedPrediction[mask], combinedTarget[mask])
            mae_percentiles.append((percentile, mae))
        else:
            mae_percentiles.append((percentile, np.nan))

    print("mae percentiles")
    print(mae_percentiles)

    # Print MAE for each percentile range
    for (percentile, mae) in mae_percentiles:
        print("here")
        print(f"MAE for variance below {percentile:.2f}: {mae:.4f}")





    time.sleep(5)



    return mae



def calculate_mae(predictions, targets):
            return np.mean(np.abs(predictions - targets))

