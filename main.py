import torch
from torch.utils.data import DataLoader, Subset
from dataPreparation import load_data, prepare_data_ensemble, prepare_data_ensemble_no_uncertainty
from training import train, generate_meta_features, train_ensemble, train_ensemble_no_uncertainty, createSet_baselineModel, generateMeta_features_baseline, createSet_prefix_baseline, generateMeta_features_baseline, train_no_uncertainty, generate_meta_features_no_unct
from testing import test_loop, test_loop_ensemble, test_loop_ensemble_no_uncertainty, test_loop_min_uncertainty, test_loop_weighted_average, test_baseline, test_baseline_prefix, test_loop_no_uncertainty, test_loop_average
from models import Ensemble, EnsembleNoUncertainty, SimpleRegressionNoUncertainty, SimpleRegressionUncertainty, Net, NetNoUnc, SimpleEnsemble, SimpleEnsembleNoUncertainty, BranchForEveryInput, StochasticCNN_1D, StochasticCNN_1DNoUnc
import numpy as np
import math
from sklearn.ensemble import RandomForestRegressor
import pickle
from torch import nn, optim
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import KFold
torch.cuda.empty_cache()


'''
    Configurations, hyperparameters and others
'''
path_train = "./RawData/Sepsis Cases_train.xes"
path_val = "./RawData/Sepsis Cases_val.xes"
path_test = "./RawData/Sepsis Cases_test.xes"




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device")
print(device)
learning_rate_lstm = 0.005
batch_size = 512
slidingWindow = 30


num_epochs_lstm = 20
hidden_size = 16

num_layers = 1


# For ensemble
num_models = 3
num_epochs_ensemble = 451
learning_rate_ensemble = 0.001
learning_rate_ensemble_LP = 0.001
num_epochs_regression = 451
loss_func = nn.L1Loss()
# Load data

num_epochs_cnn = 20
num_filters = 32
filter_size = 3
stride = 2
learning_rate_cnn = 0.001

number_of_runs = 5


train_dataset, val_dataset, test_dataset, num_features_df, features_to_index = load_data(path_train, path_val, path_test, slidingWindow)



printing = True
learning_rate = [0.01, 0.001, 0.0001]

for h in range(len(learning_rate)):
    averageMAE_no_no_array = []
    averageMAE_trained_no_array = []
    mae_weighted_average_array = []
    mae_ensemble_no_no_reg_array = []
    mae_ensemble_no_uncertainty_reg_array = []
    mae_ensemble_both_reg_array = []
    mae_ensemble_no_no_mlp_array = []
    mae_ensemble_no_uncertainty_mlp_array = []
    mae_ensemble_both_mlp_array = []
    mae_ensemble_no_no_mlp_simple_array = []
    mae_ensemble_no_uncertainty_mlp_simple_array = []
    mae_ensemble_both_mlp_simple_array = []


    cnn_average_no = []
    cnn_average_with = []


    lstm_average_no = []
    lstm_average_with = []

    baseline_activity_array = []

    print("run number", h)
    with open("results_cnn_lstm_baseline.txt", "a") as file:
        file.write(f"Run number {h} with {learning_rate[h]}" + '\n')



    for j in range(number_of_runs):

        torch.cuda.empty_cache()

        print("Learning rate")
        print(h)

        learning_rate_ensemble = learning_rate[h]
        learning_rate_ensemble_LP = learning_rate[h]

        learning_rate_branch_gate = learning_rate[h]

        

        embedding_dim = round(math.sqrt(num_features_df))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        '''
            Load Models here. On the first itteration, we train the models. Afterwards, we only do inferencing to save computing ressources.
        '''
        if (j == 0):
            cnn_model = StochasticCNN_1D(num_features_df, embedding_dim, seq_len=slidingWindow)
            cnn_model = train(num_epochs_cnn, cnn_model, train_loader, val_loader, learning_rate_cnn)
            mae_cnn = test_loop(test_loader, cnn_model, printing=printing)

            
            # Train LSTM
            cnn_model_no_unc = StochasticCNN_1DNoUnc(num_features_df, embedding_dim, seq_len=slidingWindow, Bayes=False)
            cnn_model_no_unc = train_no_uncertainty(num_epochs_cnn, cnn_model_no_unc, train_loader, val_loader, learning_rate_cnn)
            mae_cnn_no_unc = test_loop_no_uncertainty(test_loader, cnn_model_no_unc)


            print("LSTM")
            lstm_model = Net(num_features_df, hidden_size, num_layers, embedding_dim).to(device)
            lstm_model =  train(num_epochs_lstm, lstm_model, train_loader, val_loader, learning_rate_lstm)
            mae_lstm = test_loop(test_loader, lstm_model, printing=printing)

            # Train LSTM
            lstm_model_no_unc = NetNoUnc(num_features_df, hidden_size, num_layers, embedding_dim).to(device)
            lstm_model_no_unc =  train_no_uncertainty(num_epochs_lstm, lstm_model_no_unc, train_loader, val_loader, learning_rate_lstm)
            mae_lstm_no_uncertainty = test_loop_no_uncertainty(test_loader, lstm_model_no_unc)
        else:
            cnn_model = StochasticCNN_1D(num_features_df, embedding_dim, seq_len=slidingWindow)
            cnn_model = torch.load("./StochasticCNN_1D") #train(num_epochs_cnn, cnn_model, train_loader, val_loader, learning_rate_cnn)
            mae_cnn = test_loop(test_loader, cnn_model, printing=printing)

            
            # Train LSTM
            cnn_model_no_unc = StochasticCNN_1DNoUnc(num_features_df, embedding_dim, seq_len=slidingWindow, Bayes=False)
            cnn_model_no_unc = torch.load("./StochasticCNN_1DNoUnc") ##train_no_uncertainty(num_epochs_cnn, cnn_model_no_unc, train_loader, val_loader, learning_rate_cnn)
            mae_cnn_no_unc = test_loop_no_uncertainty(test_loader, cnn_model_no_unc)


            print("LSTM")
            lstm_model = Net(num_features_df, hidden_size, num_layers, embedding_dim).to(device)
            lstm_model =  torch.load("./Net") # #train(num_epochs_lstm, lstm_model, train_loader, val_loader, learning_rate_lstm)
            mae_lstm = test_loop(test_loader, lstm_model, printing=printing)

            # Train LSTM
            lstm_model_no_unc = NetNoUnc(num_features_df, hidden_size, num_layers, embedding_dim).to(device)
            lstm_model_no_unc =  torch.load("./NetNoUnc") # #train_no_uncertainty(num_epochs_lstm, lstm_model_no_unc, train_loader, val_loader, learning_rate_lstm)
            mae_lstm_no_uncertainty = test_loop_no_uncertainty(test_loader, lstm_model_no_unc)

        cnn_average_with.append(mae_cnn)

        lstm_average_with.append(mae_lstm)
        lstm_average_no.append(mae_lstm_no_uncertainty)

        baseline_dict = createSet_baselineModel(path_train)
        mae_baseline = test_baseline(baseline_dict, features_to_index, test_loader, printing=printing)
        #with open("results_cnn_lstm_baseline.txt", "a") as file:
        #    file.write(f"baseline activity {mae_baseline}" + '\n')

        baseline_activity_array.append(mae_baseline)

        baseline_dict_two = createSet_prefix_baseline(path_train)
        mae_baseline_two = test_baseline_prefix(baseline_dict_two, test_loader)
        #with open("results_cnn_lstm_baseline.txt", "a") as file:
        #    file.write(f"baseline prefix {mae_baseline_two}" + '\n')




        cnn_average_no.append(mae_cnn_no_unc)


        # Set train loaders to false to avoid mixing up the datasets during generation of META Features
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)



        '''
            For ensemble not trained on uncertainty
        '''

        no_uncertainty_cnn_train_features, no_uncertainty_train_labels = generate_meta_features_no_unct(cnn_model_no_unc, train_loader, device)
        no_uncertainty_lstm_train_features, _ = generate_meta_features_no_unct(lstm_model_no_unc, train_loader, device)
        no_uncertainty_baseline_train_features, _, allTargets = generateMeta_features_baseline(baseline_dict, features_to_index, train_loader)
        #no_uncertainty_baseline_prefix_train_features, _, _  = generateMeta_features_baseline(baseline_prefix_dict, train_loader)


        if np.array_equal(no_uncertainty_train_labels, allTargets):
            print("The training labels are exactly the same.")
        else:
            print("The training labels are different.")


        no_uncertainty_train_meta_features = np.hstack((no_uncertainty_lstm_train_features, no_uncertainty_baseline_train_features, no_uncertainty_cnn_train_features)) #, no_uncertainty_baseline_prefix_train_features))

        #time.sleep(4)


        ### Give the holdout dataset here, because ensemble is validated on other dataset
        no_uncertainty_cnn_val_features, no_uncertainty_val_labels = generate_meta_features_no_unct(cnn_model_no_unc, val_loader, device)
        no_uncertainty_lstm_val_features, _ = generate_meta_features_no_unct(lstm_model_no_unc, val_loader, device)
        no_uncertainty_baseline_val_features, _, _ = generateMeta_features_baseline(baseline_dict, features_to_index, val_loader)
        #no_uncertainty_baseline_prefix_val_features, _, _ = generateMeta_features_baseline(baseline_prefix_dict, val_loader)
        no_uncertainty_val_meta_features = np.hstack((no_uncertainty_lstm_val_features, no_uncertainty_baseline_val_features, no_uncertainty_cnn_val_features)) #, no_uncertainty_baseline_prefix_val_features))


        no_uncertainty_cnn_test_features, no_uncertainty_test_labels = generate_meta_features_no_unct(cnn_model_no_unc, test_loader, device)
        no_uncertainty_lstm_test_features, _ = generate_meta_features_no_unct(lstm_model_no_unc, test_loader, device)
        no_uncertainty_baseline_test_features, _, _ = generateMeta_features_baseline(baseline_dict, features_to_index, test_loader)
        #no_uncertainty_baseline_prefix_test_features, _, _ = generateMeta_features_baseline(baseline_prefix_dict, test_loader)
        no_uncertainty_test_meta_features = np.hstack((no_uncertainty_lstm_test_features, no_uncertainty_baseline_test_features, no_uncertainty_cnn_test_features)) #, no_uncertainty_baseline_prefix_test_features))



        helper_labels_no = no_uncertainty_test_labels
        residuals_no = np.abs(no_uncertainty_test_meta_features - helper_labels_no)



        first_column_better_no = np.sum((residuals_no[:, 0] < residuals_no[:, 1]) & (residuals_no[:, 0] < residuals_no[:, 2]))
        second_column_better_no = np.sum((residuals_no[:, 1] < residuals_no[:, 0]) & (residuals_no[:, 1] < residuals_no[:, 2]))
        third_column_better_no = np.sum((residuals_no[:, 2] < residuals_no[:, 0]) & (residuals_no[:, 2] < residuals_no[:, 1]))



        time.sleep(4)

        correlation_matrix_residuals_no = np.corrcoef(residuals_no, rowvar=False)
        print(correlation_matrix_residuals_no)




        indices_no = np.arange(residuals_no.shape[0])







        correlation_matrix_predictions_no = np.corrcoef(no_uncertainty_test_meta_features, rowvar=False)
        print(correlation_matrix_predictions_no)
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_matrix_predictions_no, annot=True, cmap='coolwarm', cbar=True)
        plt.title('Correlation Matrix predictions')
        #plt.show()
        if (printing == True):
            plt.show()




        '''
            For ensemble which is not trained on uncertainty
        '''


        no_uncertainty_train_dataloader_ensemble = prepare_data_ensemble_no_uncertainty(no_uncertainty_train_meta_features, no_uncertainty_train_labels, shuffle=False, batch_size=batch_size)
        no_uncertainty_val_dataloader_ensemble = prepare_data_ensemble_no_uncertainty(no_uncertainty_val_meta_features, no_uncertainty_val_labels, shuffle=False, batch_size=batch_size)
        no_uncertainty_test_dataloader_ensemble = prepare_data_ensemble_no_uncertainty(no_uncertainty_test_meta_features, no_uncertainty_test_labels, shuffle=False, batch_size=batch_size)

        ensemble_no_unc = EnsembleNoUncertainty(num_models).to(device)
        ensemble_no_unc = train_ensemble_no_uncertainty(num_epochs_ensemble, ensemble_no_unc, no_uncertainty_train_dataloader_ensemble, no_uncertainty_val_dataloader_ensemble, loss_func, learning_rate_ensemble)
        mae_ensemble_no_no_mlp = test_loop_ensemble_no_uncertainty(no_uncertainty_test_dataloader_ensemble, ensemble_no_unc)

        ensemble_no_unc_simple = SimpleEnsembleNoUncertainty(num_models).to(device)
        ensemble_no_unc_simple = train_ensemble_no_uncertainty(num_epochs_ensemble, ensemble_no_unc_simple, no_uncertainty_train_dataloader_ensemble, no_uncertainty_val_dataloader_ensemble, loss_func, learning_rate_ensemble)
        mae_ensemble_no_no_mlp_simple = test_loop_ensemble_no_uncertainty(no_uncertainty_test_dataloader_ensemble, ensemble_no_unc_simple)





        ensemble_no_unc_reg = SimpleRegressionNoUncertainty(num_models).to(device)
        ensemble_no_unc_reg = train_ensemble_no_uncertainty(num_epochs_regression, ensemble_no_unc_reg, no_uncertainty_train_dataloader_ensemble, no_uncertainty_val_dataloader_ensemble, loss_func, learning_rate_ensemble_LP)
        mae_ensemble_no_no_reg = test_loop_ensemble_no_uncertainty(no_uncertainty_test_dataloader_ensemble, ensemble_no_unc_reg)




        averageMAE_no_no = test_loop_average(no_uncertainty_test_dataloader_ensemble)

        '''
            For ensembles which are trained on uncertainty
        '''
        cnn_train_features, cnn_train_var, train_labels, allVariances_aleatoric_train_cnn, allVariances_epistemic_train_cnn = generate_meta_features(cnn_model, train_loader, device)
        lstm_train_features, lstm_train_var, train_labels_test_lstm, allVariances_aleatoric_train_lstm, allVariances_epistemic_train_lstm = generate_meta_features(lstm_model, train_loader, device)
        baseline_train_features, baseline_train_var, all_targets_pre = generateMeta_features_baseline(baseline_dict, features_to_index, train_loader)

        if np.array_equal(train_labels, all_targets_pre):
            print("The training no uncertainty labels are exactly the same.")
        else:
            print("The training labels are different.")

        if np.array_equal(train_labels, train_labels_test_lstm):
            print("The training no uncertainty labels are exactly the same LSTM and CNN!.")
        else:
            print("The training labels are different LSTM AND CNN!.")


        baseline_train_var = torch.nan_to_num(baseline_train_var, nan=1e5)
        #baseline_prefix_train_features, baseline_prefix_train_var, all_targets = generateMeta_features_baseline(baseline_prefix_dict, train_loader)
        #baseline_prefix_train_var = torch.nan_to_num(baseline_prefix_train_var, nan=0.0)
        train_meta_features = np.hstack((lstm_train_features, baseline_train_features, cnn_train_features)) #, baseline_prefix_train_features))
        train_meta_var = np.hstack((lstm_train_var, baseline_train_var, cnn_train_var)) #, baseline_prefix_train_var))

        


        cnn_val_features, cnn_val_var, val_labels, allVariances_aleatoric_val_cnn, allVariances_epistemic_val_cnn = generate_meta_features(cnn_model, val_loader, device)
        lstm_val_features, lstm_val_var, _, allVariances_aleatoric_val_lstm, allVariances_epistemic_val_lstm = generate_meta_features(lstm_model, val_loader, device)
        baseline_val_features, baseline_val_var, _ = generateMeta_features_baseline(baseline_dict, features_to_index, val_loader)
        baseline_val_var = torch.nan_to_num(baseline_val_var, nan=1e5)
        #baseline_prefix_val_features, baseline_prefix_val_var, _ = generateMeta_features_baseline(baseline_prefix_dict, val_loader)
        #baseline_prefix_val_var = torch.nan_to_num(baseline_prefix_val_var, nan=0.0)
        val_meta_features = np.hstack((lstm_val_features, baseline_val_features, cnn_val_features)) #, baseline_prefix_val_features))
        val_meta_var = np.hstack((lstm_val_var, baseline_val_var, cnn_val_var)) #, baseline_prefix_val_var))
        #val_meta_var_epistemic = np.hstack((allVariances_epistemic_val_lstm, baseline_val_var )) #, baseline_prefix_val_var))
        #val_meta_var_aleatoric = np.hstack((allVariances_aleatoric_val_lstm, baseline_val_var )) #, baseline_prefix_val_var))


        cnn_test_features, cnn_test_var, test_labels, allVariances_aleatoric_test_cnn, allVariances_epistemic_test_cnn = generate_meta_features(cnn_model, test_loader, device)
        lstm_test_features, lstm_test_var, _, allVariances_aleatoric_test_lstm, allVariances_epistemic_test_lstm = generate_meta_features(lstm_model, test_loader, device)
        baseline_test_features, baseline_test_var, _ = generateMeta_features_baseline(baseline_dict, features_to_index, test_loader)
        baseline_test_var = torch.nan_to_num(baseline_test_var, nan=1e5)
        #baseline_prefix_test_features, baseline_prefix_test_var, _ = generateMeta_features_baseline(baseline_prefix_dict, test_loader)
        #baseline_prefix_test_var = torch.nan_to_num(baseline_prefix_test_var, nan=0.0)
        test_meta_features = np.hstack((lstm_test_features, baseline_test_features, cnn_test_features)) #, baseline_prefix_test_features))
        test_meta_var = np.hstack((lstm_test_var, baseline_test_var, cnn_test_var)) #, baseline_prefix_test_var))
        #test_meta_var_epistemic = np.hstack((allVariances_epistemic_test_lstm, baseline_test_var )) #, baseline_prefix_test_var))
        #test_meta_var_aleatoric = np.hstack((allVariances_aleatoric_test_lstm, baseline_test_var )) #, baseline_prefix_test_var))



        ## Both variance
        train_dataloader_ensemble = prepare_data_ensemble(train_meta_features, train_meta_var, train_labels, batch_size=batch_size, shuffle=False)
        val_dataloader_ensemble = prepare_data_ensemble(val_meta_features, val_meta_var, val_labels, batch_size=batch_size, shuffle=False)
        test_dataloader_ensemble = prepare_data_ensemble(test_meta_features, test_meta_var, test_labels, batch_size=batch_size, shuffle=False)

        #train_dataloader_ensemble_epistemic = prepare_data_ensemble(train_meta_features, train_meta_var_epistemic, train_labels, batch_size=batch_size, shuffle=False)
        #val_dataloader_ensemble_epistemic  = prepare_data_ensemble(val_meta_features, val_meta_var_epistemic, val_labels, batch_size=batch_size, shuffle=False)
        #test_dataloader_ensemble_epistemic  = prepare_data_ensemble(test_meta_features, test_meta_var_epistemic, test_labels, batch_size=batch_size, shuffle=False)

        #train_dataloader_ensemble_aleatoric = prepare_data_ensemble(train_meta_features, train_meta_var_aleatoric, train_labels, batch_size=batch_size, shuffle=False)
        #val_dataloader_ensemble_aleatoric  = prepare_data_ensemble(val_meta_features, val_meta_var_aleatoric, val_labels, batch_size=batch_size, shuffle=False)
        #test_dataloader_ensemble_aleatoric  = prepare_data_ensemble(test_meta_features, test_meta_var_aleatoric, test_labels, batch_size=batch_size, shuffle=False)


        ## Is expecting variance!
        mae_weighted_average = test_loop_weighted_average(test_dataloader_ensemble)
        #with open("results_cnn_lstm_baseline.txt", "a") as file:
        #    file.write(f"MAE weighted average Uncretainty {mae_weighted_average}" + '\n')


        '''
            If epistemic uncertainty is interesting
        '''



        '''
        print("ensemble with epistemic")
        ensemble = Ensemble(num_models).to(device)
        ensemble = train_ensemble(num_epochs_ensemble, ensemble, train_dataloader_ensemble_epistemic, val_dataloader_ensemble_epistemic, loss_func, learning_rate_ensemble)
        mae_ensemble_epistemic_mlp = test_loop_ensemble(test_dataloader_ensemble_epistemic, ensemble)
        with open("results_cnn_lstm_baseline.txt", "a") as file:
            file.write(f"MAE Ensemble MLP epistemic {mae_ensemble}" + '\n')



        ensemble = SimpleRegressionUncertainty(num_models).to(device)
        ensemble = train_ensemble(num_epochs_regression, ensemble, train_dataloader_ensemble_epistemic, val_dataloader_ensemble_epistemic, loss_func, learning_rate_ensemble_LP)
        mae_ensemble_epistemic_reg = test_loop_ensemble(test_dataloader_ensemble_epistemic, ensemble)
        with open("results_cnn_lstm_baseline.txt", "a") as file:
            file.write(f"MAE Ensemble regression epistemic {mae_ensemble}" + '\n')
        '''

        '''
            Ensemble with both variances (aleatoric + epistemic)
        '''


        print("ensemble with both uncertainty")
        ensemble_w_unc = Ensemble(num_models).to(device)
        ensemble_w_unc = train_ensemble(num_epochs_ensemble, ensemble_w_unc, train_dataloader_ensemble, val_dataloader_ensemble, loss_func, learning_rate_ensemble)
        mae_ensemble_both_mlp = test_loop_ensemble(test_dataloader_ensemble, ensemble_w_unc)

        
        ensemble_w_simple = SimpleEnsemble(num_models).to(device)
        ensemble_w_simple = train_ensemble(num_epochs_ensemble, ensemble_w_simple, train_dataloader_ensemble, val_dataloader_ensemble, loss_func, learning_rate_ensemble)
        mae_ensemble_both_mlp_simple = test_loop_ensemble(test_dataloader_ensemble, ensemble_w_simple)



        ensemble_w_reg = SimpleRegressionUncertainty(num_models).to(device)
        ensemble_w_reg = train_ensemble(num_epochs_regression, ensemble_w_reg, train_dataloader_ensemble, val_dataloader_ensemble, loss_func, learning_rate_ensemble_LP)
        mae_ensemble_both_reg = test_loop_ensemble(test_dataloader_ensemble, ensemble_w_reg)

        '''
        print("ensemble with aleatoric")
        ensemble = Ensemble(num_models).to(device)
        ensemble = train_ensemble(num_epochs_ensemble, ensemble, train_dataloader_ensemble, val_dataloader_ensemble_aleatoric, loss_func, learning_rate_ensemble)
        mae_ensemble_aleatoric_mlp = test_loop_ensemble(test_dataloader_ensemble_aleatoric, ensemble)
        with open("results_cnn_lstm_baseline.txt", "a") as file:
            file.write(f"MAE Ensemble MLP aleatoric {mae_ensemble}" + '\n')



        ensemble = SimpleRegressionUncertainty(num_models).to(device)
        ensemble = train_ensemble(num_epochs_regression, ensemble, train_dataloader_ensemble, val_dataloader_ensemble_aleatoric, loss_func, learning_rate_ensemble_LP)
        mae_ensemble = test_loop_ensemble(test_dataloader_ensemble_aleatoric, ensemble)
        with open("results_cnn_lstm_baseline.txt", "a") as file:
            file.write(f"MAE Ensemble regression aleatoric {mae_ensemble}" + '\n')
        '''



        train_dataloader_ensemble_no_uncertainty = prepare_data_ensemble_no_uncertainty(train_meta_features, train_labels, batch_size=batch_size, shuffle=False)
        val_dataloader_ensemble_no_uncertainty = prepare_data_ensemble_no_uncertainty(val_meta_features, val_labels, batch_size=batch_size, shuffle=False)
        test_dataloader_ensemble_no_uncertainty = prepare_data_ensemble_no_uncertainty(test_meta_features, test_labels, batch_size=batch_size, shuffle=False)




        averageMAE_trained_no = test_loop_average(test_dataloader_ensemble_no_uncertainty)

        ## No good results_cnn_lstm_baseline yet
        ensemble_no_uncertainty_trained = EnsembleNoUncertainty(num_models).to(device)
        ensemble_no_uncertainty_trained = train_ensemble_no_uncertainty(num_epochs_ensemble, ensemble_no_uncertainty_trained, train_dataloader_ensemble_no_uncertainty, val_dataloader_ensemble_no_uncertainty, loss_func, learning_rate_ensemble)
        mae_ensemble_no_uncertainty_mlp = test_loop_ensemble_no_uncertainty(test_dataloader_ensemble_no_uncertainty, ensemble_no_uncertainty_trained)


        ensemble_no_uncertainty_simple_trained = SimpleEnsembleNoUncertainty(num_models).to(device)
        ensemble_no_uncertainty_simple_trained = train_ensemble_no_uncertainty(num_epochs_ensemble, ensemble_no_uncertainty_simple_trained, train_dataloader_ensemble_no_uncertainty, val_dataloader_ensemble_no_uncertainty, loss_func, learning_rate_ensemble)
        mae_ensemble_no_uncertainty_mlp_simple = test_loop_ensemble_no_uncertainty(test_dataloader_ensemble_no_uncertainty, ensemble_no_uncertainty_simple_trained)




        ensemble_no_uncertainty_reg_trained = SimpleRegressionNoUncertainty(num_models).to(device)
        ensemble_no_uncertainty_reg_trained = train_ensemble_no_uncertainty(num_epochs_regression, ensemble_no_uncertainty_reg_trained, train_dataloader_ensemble_no_uncertainty, val_dataloader_ensemble_no_uncertainty, loss_func, learning_rate_ensemble_LP)
        mae_ensemble_no_uncertainty_reg = test_loop_ensemble_no_uncertainty(test_dataloader_ensemble_no_uncertainty, ensemble_no_uncertainty_reg_trained)




        
        


        averageMAE_no_no_array.append(averageMAE_no_no)
        averageMAE_trained_no_array.append(averageMAE_trained_no)
        mae_weighted_average_array.append(mae_weighted_average)
        mae_ensemble_no_no_reg_array.append(mae_ensemble_no_no_reg)
        mae_ensemble_no_uncertainty_reg_array.append(mae_ensemble_no_uncertainty_reg)
        mae_ensemble_both_reg_array.append(mae_ensemble_both_reg)
        mae_ensemble_no_no_mlp_array.append(mae_ensemble_no_no_mlp)
        mae_ensemble_no_uncertainty_mlp_array.append(mae_ensemble_no_uncertainty_mlp)
        mae_ensemble_both_mlp_array.append(mae_ensemble_both_mlp)
        mae_ensemble_no_no_mlp_simple_array.append(mae_ensemble_no_no_mlp_simple)
        mae_ensemble_no_uncertainty_mlp_simple_array.append(mae_ensemble_no_uncertainty_mlp_simple)
        mae_ensemble_both_mlp_simple_array.append(mae_ensemble_both_mlp_simple)


    '''
        Take the average over 5 runs
    '''
    
    averageMAE_no_no_array_average = sum(averageMAE_no_no_array) / len(averageMAE_no_no_array)
    averageMAE_trained_no_array_average = sum(averageMAE_trained_no_array) / len(averageMAE_trained_no_array)
    mae_weighted_average_array_average = sum(mae_weighted_average_array) / len(mae_weighted_average_array)
    mae_ensemble_no_no_reg_array_average = sum(mae_ensemble_no_no_reg_array) / len(mae_ensemble_no_no_reg_array)
    mae_ensemble_no_uncertainty_reg_array_average = sum(mae_ensemble_no_uncertainty_reg_array) / len(mae_ensemble_no_uncertainty_reg_array)
    mae_ensemble_both_reg_array_average = sum(mae_ensemble_both_reg_array) / len(mae_ensemble_both_reg_array)
    mae_ensemble_no_no_mlp_array_average = sum(mae_ensemble_no_no_mlp_array) / len(mae_ensemble_no_no_mlp_array)
    mae_ensemble_no_uncertainty_mlp_array_average = sum(mae_ensemble_no_uncertainty_mlp_array) / len(mae_ensemble_no_uncertainty_mlp_array)
    mae_ensemble_both_mlp_array_average = sum(mae_ensemble_both_mlp_array) / len(mae_ensemble_both_mlp_array)
    mae_ensemble_no_no_mlp_simple_array_average = sum(mae_ensemble_no_no_mlp_simple_array) / len(mae_ensemble_no_no_mlp_simple_array)
    mae_ensemble_no_uncertainty_mlp_simple_array_average = sum(mae_ensemble_no_uncertainty_mlp_simple_array) / len(mae_ensemble_no_uncertainty_mlp_simple_array)
    mae_ensemble_both_mlp_simple_array_average = sum(mae_ensemble_both_mlp_simple_array) / len(mae_ensemble_both_mlp_simple_array)


    cnn_average_no_average = sum(cnn_average_no) / len(cnn_average_no)
    cnn_average_with_average = sum(cnn_average_with) / len(cnn_average_with)


    lstm_average_no_average = sum(lstm_average_no) / len(lstm_average_no)
    lstm_average_with_average = sum(lstm_average_with) / len(lstm_average_with)



    baseline_activity_array_average = sum(baseline_activity_array) / len(baseline_activity_array)


    '''
        Average
    '''
    with open("results_cnn_lstm_baseline.txt", "a") as file:
        file.write(f"Average not trained and not given {averageMAE_no_no_array_average}" + '\n')
    
    with open("results_cnn_lstm_baseline.txt", "a") as file:
        file.write(f"Average trained and not given {averageMAE_trained_no_array_average}" + '\n')
        
    with open("results_cnn_lstm_baseline.txt", "a") as file:
        file.write(f"Average trained and given uncertainty {mae_weighted_average_array_average}" + '\n')



    
    '''
        Regression
    '''
    with open("results_cnn_lstm_baseline.txt", "a") as file:
        file.write(f"Regre not trained and not given {mae_ensemble_no_no_reg_array_average}" + '\n')
    
    with open("results_cnn_lstm_baseline.txt", "a") as file:
        file.write(f"Regre trained and not given {mae_ensemble_no_uncertainty_reg_array_average}" + '\n')
        
    with open("results_cnn_lstm_baseline.txt", "a") as file:
        file.write(f"Regre trained and given uncertainty {mae_ensemble_both_reg_array_average}" + '\n')

    
    '''
        MLP
    '''
    with open("results_cnn_lstm_baseline.txt", "a") as file:
        file.write(f"MLP not trained and not given {mae_ensemble_no_no_mlp_array_average}" + '\n')
    
    with open("results_cnn_lstm_baseline.txt", "a") as file:
        file.write(f"MLP trained and not given {mae_ensemble_no_uncertainty_mlp_array_average}" + '\n')
        
    with open("results_cnn_lstm_baseline.txt", "a") as file:
        file.write(f"MLP trained and given uncertainty {mae_ensemble_both_mlp_array_average}" + '\n')


    
    '''
        MLP
    '''
    with open("results_cnn_lstm_baseline.txt", "a") as file:
        file.write(f"MLP Simple not trained and not given {mae_ensemble_no_no_mlp_simple_array_average}" + '\n')
    
    with open("results_cnn_lstm_baseline.txt", "a") as file:
        file.write(f"MLP Simple trained and not given {mae_ensemble_no_uncertainty_mlp_simple_array_average}" + '\n')
        
    with open("results_cnn_lstm_baseline.txt", "a") as file:
        file.write(f"MLP Simple trained and given uncertainty {mae_ensemble_both_mlp_simple_array_average}" + '\n')


    '''
        Base Models
    '''
    with open("results_cnn_lstm_baseline.txt", "a") as file:
        file.write(f"CNN Average no hetero and no Bayes {cnn_average_no_average}" + '\n')
        
    with open("results_cnn_lstm_baseline.txt", "a") as file:
        file.write(f"CNN Average with Bayes and hetero {cnn_average_with_average}" + '\n')


    with open("results_cnn_lstm_baseline.txt", "a") as file:
        file.write(f"LSTM Average no hetero and no Bayes {lstm_average_no_average}" + '\n')
        
    with open("results_cnn_lstm_baseline.txt", "a") as file:
        file.write(f"LSTM Average with Bayes and hetero {lstm_average_with_average}" + '\n')



    with open("results_cnn_lstm_baseline.txt", "a") as file:
        file.write(f"Average baseline {baseline_activity_array_average}" + '\n')





