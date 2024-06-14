import torch
import pm4py
from torch import nn, optim
from torch.utils.data import DataLoader
from pm4py.algo.transformation.log_to_features import algorithm as log_to_features
from creatingOneHotEncoding import creatingTensorsTrainingAndTesting
import torch.nn.functional as F
import math
from allVariables import batch_size, num_epochs, learning_rate, embedding_dim, slidingWindow, helperDataset, path_train, path_test, device
import matplotlib.pyplot as plt

print(f'Using device: {device}')

path_train = "./helper.xes"
path_test = "./helper.xes"
path_val = "./helper.xes"
learning_rate = 0.001
num_epochs = 10000
embedding_dim = 5


print("Sliding Window")
slidingWindow = 40



# Things for improvement:
## Make fc1 dynamic
## Make the setup more similar to papers


## Sequence length optional, if I want to calculate the input number dynamically
class CNN(nn.Module):
    def __init__(self, num_features_df, embedding_dim, seq_length):
        super(CNN, self).__init__()
        self.embedding = nn.Embedding(num_features_df + 1, embedding_dim)
        self.conv1 = nn.Conv1d(embedding_dim + 1, out_channels=2, kernel_size=4, stride=2)
        self.conv2 = nn.Conv1d(2, out_channels=3, kernel_size=3, stride=2)
        self.conv3 = nn.Conv1d(3, out_channels=1, kernel_size=2, stride=2)

        def conv_output_size(l_in, kernel_size, stride):
            return math.floor((l_in - kernel_size) / stride) + 1

        L_out1 = conv_output_size(seq_length, 4, 2)
        L_out2 = conv_output_size(L_out1, 3, 2)
        L_out3 = conv_output_size(L_out2, 2, 2)

        # Needs to be calculated dynamically
        self.fc1 = nn.Linear(L_out3, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, X, timeArray): #, timeInDay, timeInWeek):
        embeddings = self.embedding(X)
        #print("Embedding")
        #print(embeddings)
        X_combined = torch.cat((embeddings, timeArray.unsqueeze(-1)), dim=-1) #timeInDay.unsqueeze(-1), timeInWeek.unsqueeze(-1)), dim=-1)
        #print("X_com before")
        #print(X_combined)
        X_combined = X_combined.permute(0, 2, 1) 
        #print("X comb after")
        #print(X_combined)
        ## Relu activation function, since it is common in CNNs
        x = F.relu(self.conv1(X_combined))
        
        x = F.relu(self.conv2(x))
        
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if (__name__ == "__main__"):

    trainDataFrame = pm4py.read.read_xes(path_train)
    testDataFrame = pm4py.read.read_xes(path_test)
    valDataFrame = pm4py.read.read_xes(path_val)
    _, featuresDf = (log_to_features.apply(trainDataFrame, parameters={"str_ev_attr": ["concept:name"]}))
    features_to_index = {feature:idx for idx, feature in enumerate(featuresDf)}

    features_to_index = {key: value + 1 for key, value in features_to_index.items()}

    ## Leave those two variables here. Might get confusing otherwise ##
    number_features_df = len(featuresDf)
    print("Length")
    print(number_features_df)
    print("number features")
    print(number_features_df)

    # embedding_dim + 1 because we have 1 additional dimension from time component
    model = CNN(number_features_df, embedding_dim, slidingWindow)
    print(model)

    # MAE loss
    loss_func = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)

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
    #model = torch.load('./generatingModels/cnn_time_2017')


    datasetTesting = creatingTensorsTrainingAndTesting(testDataFrame, features_to_index, sliding_window=slidingWindow)
    test_dataloader = DataLoader(datasetTesting, batch_size=batch_size, shuffle=False)

    def test_loop(dataloader, model):
        model.eval()
        total_absolute_error = 0
        numberOfSamples = 0
        with torch.no_grad():
            for (inputs_value, inputs_time, time_in_day_input, time_in_week, targets) in dataloader:
                inputs_value, inputs_time, time_in_day_input, time_in_week, targets = inputs_value.to(device), inputs_time.to(device), time_in_day_input.to(device), time_in_week.to(device), targets.to(device)
                pred = model(inputs_value, inputs_time) #, time_in_day_input, time_in_week)
                absolute_diff = torch.abs(targets.squeeze() - pred.squeeze())
                total_absolute_error += torch.sum(absolute_diff).item()
                numberOfSamples += targets.numel()
        mae = total_absolute_error / numberOfSamples
        print(f'Mean Absolute Error: {mae:.4f}')
        return mae

    print("Testing Now!")
    test_loop(test_dataloader, model)
