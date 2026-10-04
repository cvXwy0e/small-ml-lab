import torch
import torch.nn as nn
from torchvision import transforms, datasets
import PIL as pl
import torch.optim as optim


def main():

    # model setup
    X, Y = load_data()
    model = get_model()
    train(X, Y, model)

    # Testing
    testing_dataset = datasets.MNIST(
        root="data",
        train=False,
        download=True,
    )
    # stoped here for today...


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
        images.append(img)
        labels.append(label)
    
    print(images[0], labels[0])
    return (images, labels)

def get_model():
    ...

def train(X, Y, model):
    ...

