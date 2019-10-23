import torch
import torch.nn as nn
from torchvision.models import resnet18
import math
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2
from torch.utils.data import DataLoader, TensorDataset


def load_data(batch_size):
    path_imgs = "/home/lab/Input_Images/"
    path_labels = "/home/lab/Pushing_Preprocessed/"

    x = []
    y = []

    for i in range(136323): #136323
        x.append(cv2.imread(path_imgs + str(i) + ".png").reshape(3, 224, 224))
        y.append([np.load(path_labels + str(i) + ".npy", allow_pickle=True)[1]])

    x = np.array(x)
    y = np.array(y)

    randomize = np.arange(len(x))
    np.random.shuffle(randomize)
    x = x[randomize]
    y = y[randomize]

    x_train = x[:100000]
    y_train = y[:100000]

    x_val = x[125000:]
    y_val = y[125000:]
    return x_train, y_train, x_val, y_val

def toDataset(batch_size):
    x_train, y_train, x_val, y_val = load_data(batch_size)

#    tensor_x_train = torch.stack([torch.Tensor(i) for i in x_train])
#    tensor_y_train = torch.stack([torch.Tensor(i) for i in y_train])

#    tensor_x_val = torch.stack([torch.Tensor(i) for i in x_val])
#    tensor_y_val = torch.stack([torch.Tensor(i) for i in y_val])

    tensor_x_train = torch.from_numpy(x_train).byte()
    tensor_y_train = torch.from_numpy(y_train).float()

    tensor_x_val = torch.from_numpy(x_val).byte()
    tensor_y_val = torch.from_numpy(y_val).float()

    print("made it here")
    train_dataset = TensorDataset(tensor_x_train, tensor_y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(tensor_x_val, tensor_y_val)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    return train_dataloader, val_dataloader

def train(model, train_dataloader):
    model.train()
    for i, (train_X, train_Y) in enumerate(train_dataloader):
        train_X = train_X.float().cuda(0)
        train_Y = train_Y.cuda(0)

        images = Variable(train_X)
        labels = Variable(train_Y)

        opt.zero_grad()
        outputs = model(images)

        loss = loss_fn(outputs, labels)
        loss.backward()
        opt.step()
        print(loss, end="\r")
    return loss

def test(model, test_dataloader):
    model.eval()
    test_loss = 0
    correct = 0
    test_loss = 0
    with torch.no_grad():
        for data, target in test_dataloader:
            data, target = data.float().cuda(0), target.cuda(0)
            output = model(data)
            test_loss += loss_fn(output, target).item()
    test_loss /= len(test_dataloader)
    return test_loss


batch_size = 32
maxIter = 1000

torch.cuda.set_device(0)

model = resnet18(pretrained=True)
embedding_size = 1
model.fc = nn.Linear(512, 1)
loss_fn = nn.L1Loss().cuda(0)
opt = torch.optim.Adam(model.parameters(), 0.001)

train_dataloader, val_dataloader = toDataset(batch_size)

min_error = 10
model.cuda(0)

for epoch in range(1000):
    loss = train(model, train_dataloader)
    error = test(model, val_dataloader)
    print ("Epoch [%d/%d], Model Loss: %.8f Error : %.8f Min Error: %.8f" %(epoch+1, maxIter,
                                loss.item(), error, min_error))
    if error < min_error:
        torch.save(model.state_dict(), "weight_inference.pt")
    min_error = min(error, min_error)
