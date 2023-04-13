
import numpy as np
import pickle

with open("ds", "rb") as fp:   # Unpickling
    ballPathList = pickle.load(fp)

bpL = []
for traj in ballPathList:
    path = []
    for point in traj:
        x = np.asarray(point)
        path.append(x)
    
    y = np.asarray(path)
    bpL.append(y)



trajectories = np.asarray(bpL)

print(trajectories)


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split



X = trajectories[:, :-1, :]
y = trajectories[:, 1:, :]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

import torch
import torch.nn as nn

class LSTMCellModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMCellModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        self.lstm_cell2 = nn.LSTMCell(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, horizon): # FIRST CHANGE FROM WHEN WAS WORKING
        hx = torch.zeros(x.size(0), self.hidden_size)
        cx = torch.zeros(x.size(0), self.hidden_size)
        hx2 = torch.zeros(x.size(0), self.hidden_size)
        cx2 = torch.zeros(x.size(0), self.hidden_size)
        outputs = []

        for i in range(x.size(1)):
            if(i < 5):
                hx, cx = self.lstm_cell(x[:, i, :], (hx, cx))
                hx2, cx2 = self.lstm_cell2(hx, (hx2,cx2))
                outputs.append(self.fc(hx2))
            else:
                if(i % horizon == 0):
                    hx, cx = self.lstm_cell1(x[:, i, :], (hx, cx))
                    hx2, cx2 = self.lstm_cell2(hx, (hx2,cx2))
                    outputs.append(self.fc(hx2))
                else:
                    state = x[:, i, :]
                    c = np.concatenate((outputs[-1].detach().cpu().numpy(), state[:, 3:].detach().cpu().numpy()), axis=1)
                    c = torch.tensor(c).to(device)
                    hx, cx = self.lstm_cell(c, (hx, cx))
                    hx2, cx2 = self.lstm_cell2(hx, (hx2,cx2))
                    outputs.append(self.fc(hx2))
        return torch.stack(outputs, dim=1)

input_size = 9
hidden_size = 512
output_size = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMCellModel(input_size, hidden_size, output_size).to(device)
model.load_state_dict(torch.load("imitation_model.pt"))
from torch.optim import Adam

criterion = nn.MSELoss()



outputList = []

with torch.no_grad():
    train_loss = 0
    for i, (inputs, targets) in enumerate(zip(X_test, y_test)):
        inputs = torch.tensor(inputs[np.newaxis, :, :], dtype=torch.float32)
        new_targets = np.zeros((len(targets), 3))
        for q in range(len(targets)):
            new_targets[q] = targets[q][:3]
        targets = torch.tensor(new_targets[np.newaxis, :, :], dtype=torch.float32)            
        outputs = model(inputs,10000)
        outputList.append(outputs)
        loss = criterion(outputs, targets)

        train_loss += loss.item()

    train_loss /= len(X_test)
    print("Current test loss: ", train_loss)

print(outputList)

