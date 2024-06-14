import torch
import pm4py
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from pm4py.algo.transformation.log_to_features import algorithm as log_to_features
from creatingOneHotEncoding import creatingTensorsTrainingAndTesting
from allVariables import batch_size, num_epochs, learning_rate, embedding_dim, slidingWindow, helperDataset, path_train, path_test, device
import torch.nn.functional as F
import torch.nn.init as init
import matplotlib.pyplot as plt

print(f'Using device: {device}')


path_train = "./helper.xes"
path_test = "./helper.xes"
path_val = "./helper.xes"
learning_rate = 0.01
num_epochs = 10000
batch_size = 512

embedding_dim = 5
slidingWindow = 40


##### Question: Why is my loss so high for the nan values?
### Look at the helper dataset

## Batch normalization?
## Cross validation?










#### Plotting of Data!
#### Look if the testing set is actually good!
#### Cross validation () => Ergb. aggregiert

class LSTM(nn.Module):
    def __init__(self, num_features_df, hidden_size, num_layers, value_embedding):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Does this needs to be trained? It should be automatically trained
        self.embedding = nn.Embedding(num_features_df + 1, value_embedding)
        self.lstm = nn.LSTM(value_embedding + 1, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.output_layer = nn.Linear(hidden_size, 1)
    
    def forward(self, X, timeArray): #, timeInDay, timeInWeek):
        ## Look closer at the embedding! Maybe there is something wrong with the dimensionality, when unsqueezed!
        embeddings = self.embedding(X)
        X_combined = torch.cat((embeddings, timeArray.unsqueeze(-1)), dim=-1) #, timeInDay.unsqueeze(-1), timeInWeek.unsqueeze(-1)), dim=-1)
        ## This is not in a CNN
        hidden_states = torch.zeros(self.num_layers, X_combined.size(0), self.hidden_size).to(device)
        cell_states = torch.zeros(self.num_layers, X.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(X_combined, (hidden_states, cell_states))
        out = self.output_layer(out[:, -1])
        return out

# Example usage
# num_features_df, hidden_size, value_embedding, and target_chars need to be defined
# X, timeArray, timeInDay, and timeInWeek are inputs to the forward method

# Assuming we have the following hyperparameters
hidden_size = 8     # number of classes for activity prediction

# Create the model

'''




'''





if (__name__ == "__main__"):


    ## Maybe get a shared layer in here
    num_layers = 1

    trainDataFrame = pm4py.read.read_xes(path_train)
    testDataFrame = pm4py.read.read_xes(path_test)
    valDataFrame = pm4py.read.read_xes(path_val)
    _, featuresDf = (log_to_features.apply(trainDataFrame, parameters={"str_ev_attr": ["concept:name"]}))
    input_len = len(featuresDf)
    model = LSTM(input_len, hidden_size, num_layers, embedding_dim)


    features_to_index = {feature:idx for idx, feature in enumerate(featuresDf)}

    features_to_index = {key: value + 1 for key, value in features_to_index.items()}




    #model = LSTM(input_len, hidden_dim, num_layers, embedding_dim).to(device)
    print(model)

    # MAE loss
    loss_func = nn.L1Loss()
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    '''
        Training
    '''
    dataset = creatingTensorsTrainingAndTesting(trainDataFrame, features_to_index, sliding_window=slidingWindow)
    dataset_val = creatingTensorsTrainingAndTesting(valDataFrame, features_to_index, sliding_window=slidingWindow)

    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)




    def train(num_epochs, model, train_dataloader, val_dataloader, loss_func, optimizer):
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        
        for epoch in range(num_epochs):
            model.train()
            total_train_loss = 0
            
            for batch_idx, (inputs_value, inputs_time, time_in_day_input, time_in_week, targets) in enumerate(train_dataloader):
                inputs_value, inputs_time, time_in_day_input, time_in_week, targets = \
                    inputs_value.to(device), inputs_time.to(device), time_in_day_input.to(device), time_in_week.to(device), targets.to(device)
                
                optimizer.zero_grad()
                output = model(inputs_value, inputs_time) # , time_in_day_input, time_in_week
                loss = loss_func(output, targets)
                loss.backward()

                        ## Problem: Embedding Gradient super small
                optimizer.step()
                
                total_train_loss += loss.item()
                print(f"Epoch: {epoch+1}; Batch {batch_idx+1}; Training Loss: {loss.item():.4f}")


            
            # Compute average training loss
            avg_train_loss = total_train_loss / len(train_dataloader)
            train_losses.append(avg_train_loss)
            print(f"Epoch: {epoch+1}; Average Training Loss: {avg_train_loss:.4f}")

            # Validate the model
            model.eval()
            total_val_loss = 0
            
            with torch.no_grad():
                for batch_idx, (inputs_value, inputs_time, time_in_day_input, time_in_week, targets) in enumerate(val_dataloader):
                    inputs_value, inputs_time, time_in_day_input, time_in_week, targets = \
                        inputs_value.to(device), inputs_time.to(device), time_in_day_input.to(device), time_in_week.to(device), targets.to(device)
                    
                    output = model(inputs_value, inputs_time) # , time_in_day_input, time_in_week
                    loss = loss_func(output, targets)
                    total_val_loss += loss.item()
            
            # Compute average validation loss
            avg_val_loss = total_val_loss / len(val_dataloader)
            val_losses.append(avg_val_loss)
            print(f"Epoch: {epoch+1}; Average Validation Loss: {avg_val_loss:.4f}")

            # Save the best model based on validation loss
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                #torch.save(model, './generatingModels/best_model_pay')
            
            model.train()  # Switch back to training mode

        #torch.save(model, './generatingModels/last_model_pay')


        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.show()


    train(num_epochs, model, train_dataloader, val_dataloader, loss_func, optimizer)
    #model = torch.load('./generatingModels/best_model_pay')


    '''
        Testing
    '''
    datasetTesting = creatingTensorsTrainingAndTesting(testDataFrame, features_to_index, sliding_window=slidingWindow)
    print("Features to index")
    print(features_to_index)
    test_dataloader = DataLoader(datasetTesting, batch_size=batch_size, shuffle=False)

    def test_loop(dataloader, model):
        model.eval()
        total_absolute_error = 0
        numberOfSamples = 0
        ## Nan values are just left out
        with torch.no_grad():
            for (inputs_value, inputs_time, time_in_day_input, time_in_week, targets) in dataloader:
                inputs_value, inputs_time, time_in_day_input, time_in_week, targets = inputs_value.to(device), inputs_time.to(device), time_in_day_input.to(device), time_in_week.to(device), targets.to(device)
                #print("Before prediction")
                #print(inputs_value)
                #print(inputs_time)
                #print(time_in_day_input)
                #print(time_in_week)
                #print(targets)
                pred = model(inputs_value, inputs_time) #, time_in_day_input, time_in_week)
                #print("Prediction")
                #print(pred)
                absolute_diff = torch.abs(targets.squeeze() - pred.squeeze())
                #print("absolute difference")
                #print(absolute_diff)
                total_absolute_error += torch.sum(absolute_diff).item()
                #print("sum")
                #print(total_absolute_error)
                #print(targets.numel())
                numberOfSamples += targets.numel()
                #print(numberOfSamples)
        mae = total_absolute_error / numberOfSamples
        print("Mae")
        print(mae)
        print(f'Mean Absolute Error: {mae:.4f}')
        return mae

    print("Testing Now!")
    test_loop(test_dataloader, model)
