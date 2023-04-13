from scipy.spatial import distance
import numpy as np
import pickle

with open("ds", "rb") as fp:   # Unpickling
    ballPathList = pickle.load(fp)


goalPos = [0,5000,0]

def getAngle(point1, point2):
    vector1 = np.array(point1)
    vector2 = np.array(point2)

    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)

    cos_angle = dot_product / (norm1 * norm2)
    angle_rad = np.arccos(np.clip(cos_angle, -1, 1))
    angle_deg = np.degrees(angle_rad)

    return angle_deg

def calculate_velocity(point1, point2, time):
    """
    Calculate directional velocity given two points and time.

    Args:
    point1 (list): The first point as a list [x1, y1, z1]
    point2 (list): The second point as a list [x2, y2, z2]
    time (float): The time difference between the two points in seconds

    Returns:
    list: The directional velocity as a vector [vx, vy, vz]
    """

    if time <= 0:
        raise ValueError("Time should be greater than 0.")

    x1, y1, z1 = point1
    x2, y2, z2 = point2

    # Calculate the directional velocity components
    vx = (x2 - x1) / time
    vy = (y2 - y1) / time
    vz = (z2 - z1) / time

    # Return the velocity as a vector
    return [vx, vy, vz]

def getVelList(posList):
    velList = [[0,0,0]]
    for i in range(1,len(posList)):
        velList.append(calculate_velocity(posList[i - 1], posList[i], 0.1))
    return velList

for j in range(len(ballPathList)):
    p1PosList = []
    p2PosList = []
    ballPosList = []
    p1BallDistList = []
    p2BallDistList = []
    p1GoalDistList = []
    p2GoalDistList = []
    ballGoalDistList = []
    p1GoalAngleList = []
    p2GoalAngleList = []
    ballGoalAngleList = []
    p1BallAngleList = []
    p2BallAngleList = []
    for i in range(len(ballPathList[j])):
        p1PosList.append(ballPathList[j][i][:3])
        p2PosList.append(ballPathList[j][i][3:6])
        ballPosList.append(ballPathList[j][i][6:])
        p1BallDistList.append(distance.euclidean(p1PosList[-1],ballPosList[-1]))
        p2BallDistList.append(distance.euclidean(p2PosList[-1], ballPosList[-1]))
        p1GoalDistList.append(distance.euclidean(p1PosList[-1], goalPos))
        p2GoalDistList.append(distance.euclidean(p2PosList[-1], goalPos))
        ballGoalDistList.append(distance.euclidean(ballPosList[-1], goalPos))
        p1GoalAngleList.append(getAngle(p1PosList[-1], goalPos))
        p2GoalAngleList.append(getAngle(p2PosList[-1], goalPos))
        ballGoalAngleList.append(getAngle(ballPosList[-1], goalPos))
        p1BallAngleList.append(getAngle(p1PosList[-1], ballPosList[-1]))
        p2BallAngleList.append(getAngle(p2PosList[-1], ballPosList[-1]))
    p1VelList = getVelList(p1PosList)
    p2VelList = getVelList(p2PosList)
    ballVelList = getVelList(ballPosList)
    p1VelList = p1VelList[1:]
    p2VelList = p2VelList[1:]
    ballVelList = ballVelList[1:]
    p1BallDistList = p1BallDistList[1:]
    p2BallDistList = p2BallDistList[1:]
    p1GoalDistList = p1GoalDistList[1:]
    p2GoalDistList = p2GoalDistList[1:]
    ballGoalDistList = ballGoalDistList[1:]
    p1GoalAngleList = p1GoalAngleList[1:]
    p2GoalAngleList = p2GoalAngleList[1:]
    ballGoalAngleList = ballGoalAngleList[1:]
    p1BallAngleList = p1BallAngleList[1:]
    p2BallAngleList = p2BallAngleList[1:]
    ballPathList[j] = ballPathList[j][1:]
    for i in range(len(ballPathList[j])):
        ballPathList[j][i].extend(ballVelList[i])
        ballPathList[j][i].extend(p2VelList[i])
        ballPathList[j][i].append(ballGoalDistList[i])
        ballPathList[j][i].append(ballGoalAngleList[i])
        ballPathList[j][i].append(p2BallDistList[i])
        ballPathList[j][i].append(p2BallAngleList[i])
        ballPathList[j][i].append(p2GoalDistList[i])
        ballPathList[j][i].append(p2GoalAngleList[i])
        ballPathList[j][i].extend(p1VelList[i])
        ballPathList[j][i].append(p1BallDistList[i])
        ballPathList[j][i].append(p1BallAngleList[i])
        ballPathList[j][i].append(p1GoalDistList[i])
        ballPathList[j][i].append(p1GoalAngleList[i])




bpL = []
for traj in ballPathList:
    path = []
    for point in traj:
        x = np.asarray(point)
        path.append(x)

    y = np.asarray(path)
    bpL.append(y)



trajectories = np.asarray(bpL)
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split



X = trajectories[:, :-1, :]
y = trajectories[:, 1:, :]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

import torch
import torch.nn as nn

class LSTMCellModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMCellModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        self.lstm_cell2 = nn.LSTMCell(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, horizon):
        hx = torch.zeros(x.size(0), self.hidden_size).to(device)
        cx = torch.zeros(x.size(0), self.hidden_size).to(device)
        hx2 = torch.zeros(x.size(0), self.hidden_size).to(device)
        cx2 = torch.zeros(x.size(0), self.hidden_size).to(device)

        outputs = []

        for i in range(x.size(1)):
            if(i < -1):
                print()
            else:
                if(i % horizon == 0):
                    hx, cx = self.lstm_cell(x[:, i, :], (hx, cx))
                    hx2, cx2 = self.lstm_cell2(hx, (hx2,cx2))
                    outputs.append(self.fc(hx2))
                else:
                    state = x[:, i, :]
                    prevOut = outputs[-1].detach().cpu().numpy()[0]
                    ballPos = state[:, 6:9].detach().cpu().numpy()[0]
                    vel = calculate_velocity(prevOut,state[:,:3].detach().cpu().numpy()[0],0.1)
                    vel.append(distance.euclidean(prevOut,ballPos))
                    vel.append(getAngle(prevOut,ballPos))
                    vel.append(distance.euclidean(prevOut,goalPos))
                    vel.append(getAngle(prevOut,goalPos))
                    c = np.concatenate((outputs[-1].detach().cpu().numpy(), state[:, 3:21].detach().cpu().numpy(), [vel]), axis=1)
                    c = torch.tensor(c,dtype=torch.float32).to(device)
                    hx, cx = self.lstm_cell(c, (hx, cx))
                    hx2, cx2 = self.lstm_cell2(hx, (hx2,cx2))
                    outputs.append(self.fc(hx2))
        return torch.stack(outputs, dim=1)

input_size = 28
hidden_size = 512
output_size = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = LSTMCellModel(input_size, hidden_size, output_size).to(device)
model.load_state_dict(torch.load("imitation_model.pt"))
criterion = nn.MSELoss()
outputList = []
inputList = []
with torch.no_grad():
    test_loss = 0
    for i, (inputs, targets) in enumerate(zip(X_test, y_test)):
        inputs = torch.tensor(inputs[np.newaxis, :, :], dtype=torch.float32).to(device)
        new_targets = np.zeros((len(targets), 3))
        for q in range(len(targets)):
            new_targets[q] = targets[q][:3]
        targets = torch.tensor(new_targets[np.newaxis, :, :], dtype=torch.float32).to(device)
        outputs = model(inputs, 10000)
        o = outputs.cpu().detach().numpy()
        inp = targets.cpu().detach().numpy()
        inputList.append(inp)
        outputList.append(o)
        loss = criterion(outputs, targets)

        test_loss += loss.item()

    test_loss /= len(X_test)
    print("Current test loss: ", test_loss)
import glob
import shutil
file = open("predPos.txt", "w")
for i in range(len(outputList)):
    list = outputList[i][0]
    for j in list:
        file.write(str(j[0]) + " " + str(j[1]) + " " + str(j[2]) + "\n")
file.close()
file = open("player1Pos.txt", "w")
for i in range(len(inputList)):
    list = inputList[i][0]
    for j in list:
        file.write(str(j[0]) + " " + str(j[1]) + " " + str(j[2]) + "\n")
file.close()
