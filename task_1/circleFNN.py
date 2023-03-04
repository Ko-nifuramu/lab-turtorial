import torch
import torch.nn as nn
import numpy as np
import math
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

seed = 32

def create_circle_data(period = 20):
    #data_num = period*100周分
    T = period#周期
    t = np.linspace(0, 100*T, 100*T)
    x = np.cos(2*math.pi*t/T) + np.random.normal(0, 0.05, (T*100))
    y = np.sin(2*math.pi*t/T) + np.random.normal(0, 0.05, (T*100))
    data_sample = np.stack([x, y], axis = 1)
    return data_sample

class DotDataset(torch.utils.data.Dataset):
    def __init__(self, input_data, target_data):
        self.target_data = target_data
        self.input_data = input_data

    def __len__(self):
        return self.input_data.shape[0]

    def __getitem__(self, idx):
        target_datum = torch.from_numpy(self.target_data[idx]).float()
        input_datum = torch.from_numpy(self.input_data[idx]).float()
        return target_datum, input_datum

def create_dataloader(batchsize):
    data_sample = create_circle_data(period = 20)

    indeces = [int(data_sample.shape[0]*n) for n in [0.6, 0.6+0.2]]
    train_data, val_data, test_data = np.split(data_sample, indeces)
    train_data, val_data, test_data = np.split(data_sample, indeces)

    train_input_data = train_data[0 : train_data.shape[0]-1, :]
    train_target_data = train_data[1 : train_data.shape[0], :]
    val_input_data = val_data[0 : val_data.shape[0]-1, :]
    val_target_data = val_data[1 : val_data.shape[0], :]
    test_input_data = test_data[0 : test_data.shape[0]-1, :]
    test_target_data = test_data[1 : test_data.shape[0], :]

    train_dataset = DotDataset(train_input_data, train_target_data)
    val_dataset = DotDataset(val_input_data, val_target_data)
    test_dataset = DotDataset(test_input_data, test_target_data)
    
    train_dataloader = DataLoader(train_dataset, batch_size = batchsize)
    val_dataloader = DataLoader(val_dataset, batch_size = batchsize) 
    test_dataloader = DataLoader(test_dataset, batch_size = batchsize)

    return train_dataloader, val_dataloader, test_dataloader, train_input_data, val_input_data, test_input_data


#input(np, string)
def visualize_data(data, data_name):
    print("data.shape = {}".format(data.shape))
    x,y = np.split(data, 2, axis = 1)
    plt.plot(x, y, label = data_name) 
    plt.savefig('sample_data.png')


def visualize_loss(loss_dict, epochs):
    
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(loss_dict['train'], linestyle="solid")
    ax.plot(loss_dict['val'], linestyle="dashed")
    ax.set_yscale('log')
    ax.legend(['train_loss', 'val_loss'])
    ax.set_xlim(0, epochs)
    ax.set_title('loss')
    fig.savefig('loss.png')


def visualize_test(prediction, targets):
    
    x_pred,y_pred = np.split(prediction, 2, axis = 1)
    x_target,y_target = np.split(targets, 2, axis = 1)
    
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(x_pred, y_pred, linestyle = "dotted")
    ax.plot(x_target,y_target, linestyle = "dotted")
    
    ax.legend(['pred', 'target'])
    
    ax.set_title('compare_pred-target')
    plt.savefig('test.png')


def train(model, criterion, optimizer, epochs, batch_size, lr):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)    
    train_dataloader, val_dataloader, test_dataloader, \
    train_input_data, val_input_data, test_input_data = create_dataloader(batch_size)
    optimizer = optimizer(model.parameters(), lr=lr)

    # 各epochでのloss, accを保存する配列
    loss_dict = {'train': [], 'val': []}

    for epoch in range(epochs):
        #学習
        train_loss = 0
        val_loss = 0
        model.train()
        for inputs, targets in train_dataloader:
            predictions = model(inputs)
            loss = criterion(predictions, targets)

            model.zero_grad()
            loss.backward()
            optimizer.step()
        
        #性能評価(汎化ではない)
        model.eval()
        with torch.no_grad():
            for inputs, targets in train_dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                predictions = model(inputs)
                loss = criterion(predictions, targets)
                train_loss += loss

            for inputs, targets in val_dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                predictions = model(inputs)  
                loss = criterion(predictions, targets)  
                val_loss += loss

            print(train_input_data.shape[0])
            loss_dict["train"].append(train_loss/train_input_data.shape[0])
            loss_dict["val"].append(val_loss/ val_input_data.shape[0])

        print("Epoch: {}/{} ".format(epoch + 1, epochs),
            "Traning Loss: {} ".format(train_loss/train_input_data.shape[0]),
            "Validation Loss: {}".format(val_loss/ val_input_data.shape[0]))    

    visualize_loss(loss_dict, epochs)
    return model

def test(model, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)    
    train_dataloader, val_dataloader, test_dataloader, \
    train_input_data, val_input_data, test_input_data = create_dataloader(batch_size)
    test_loss = 0
    
    model.eval()
    with torch.no_grad():
        for inputs, targets in test_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            predictions = model(inputs)  
            loss = criterion(predictions, targets)
            test_loss += loss
    
    print("test_loss = {}".format(test_loss / test_input_data.shape[0]))
    visualize_test(predictions[:20, :], targets[:20, :])

class FNN(nn.Module):
    def __init__(self):
        super(FNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        out = self.layer1(x)
        return out



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.MSELoss()  # Loss定義
optimizer = torch.optim.Adam
model = FNN().to(device)
data_sample = create_circle_data(100)
visualize_data(data_sample, "sample_data")

model = train(
    model=model, criterion=criterion, optimizer=optimizer, epochs=100, batch_size=100, lr=0.00078
)

test(model = model, batch_size=100)