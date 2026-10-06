import torch
import torch.nn as nn
from torchvision import transforms, datasets
import PIL as pl
import torch.optim
from torch.utils.data import DataLoader

def main():
    ...



def load_data():
    training_dataset = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )
    images = []
    labels = []

    for img, label in training_dataset:
        img = torch.flatten(img)
        images.append(img)
        labels.append(label)
    

    return (images, labels)

def get_model():
    model = nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    return model


def train(X, Y, model):

    data_loader = DataLoader(
        dataset=zip(X, Y),
        batch_size=64,
        shuffle=True
    )

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.02)


    for images, labels in data_loader:
        prediction = model(images)
        loss = loss_func(prediction, labels) # I'm not sure about comparing prediction[64, 10] and labels[65], should read CrossEntropyLoss() docs

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



