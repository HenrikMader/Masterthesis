import torch
import os

'''
    All variables for the notebooks
'''

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

batch_size = 1024
num_epochs = 200
learning_rate = 0.00001
embedding_dim = 5
slidingWindow = 40

helperDataset = "./helper.xes"
path_train = "./BPIC/BPI_Challenge_2019_train.xes"
path_test = "./BPIC/BPI_Challenge_2019_test.xes"

path_train_2017 = "./BPIC/BPI_Challenge_2017_train.xes"
path_test_2017 = "./BPIC/BPI_Challenge_2017_test.xes"

path_train_2019 = "./BPIC/BPI_Challenge_2019_train.xes"
path_test_2019 = "./BPIC/BPI_Challenge_2019_test.xes"



#os.system("testingTime.py")
#os.system("regression.py")
#os.system("randomforest.py")
#os.system("CNNTime.py")
#os.system("")


## Important: Change the names for saving the files!