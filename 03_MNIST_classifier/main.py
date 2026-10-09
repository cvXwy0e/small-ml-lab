import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim
from torch.utils.data import DataLoader

def main():

    # Model setup
    data_loader = load_data(isTraining=True)
    model = get_model()
    train(data_loader, model)


    # Testing Model
    test_loader = load_data(isTraining=False)
    evaluate(test_loader, model)
    



def load_data(isTraining):
    training_dataset = datasets.MNIST(
        root="data",
        train=isTraining,
        download=True,
        transform=transforms.ToTensor()
    )
    images = []
    labels = []

    for img, label in training_dataset:
        img = torch.flatten(img)
        images.append(img)
        labels.append(label)

    return get_dataloader(images, labels, toShuffle=isTraining)

def get_model():
    # simple 784 -> 128 -> 10 network
    model = nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    return model

def train(data_loader, model):

    # Used CrossEntropyLoss to make classification
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.02)

    for i in range(1,6):
        total_loss = 0
        model.train()
        for images, labels in data_loader:
            prediction = model(images)
            loss = loss_func(prediction, labels)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        avg_loss = total_loss / len(data_loader)
        print(f'Epoch {i}: loss = {avg_loss}')

def get_dataloader(X, Y, batchSize=64, toShuffle=True):
    # turned the dataset into list cuz the shuffle method needs len(DataLoader), that's not in the zip(X,Y)
    return DataLoader(
        dataset=list(zip(X, Y)),
        batch_size=batchSize,
        shuffle=toShuffle
    )

def evaluate(test_loader, model):
    # switching model to evaluating mode
    model.eval()

    correct = 0
    total = 0
    # we don't need to update anything, just testing stuff
    with torch.no_grad():

        for inputs, true_value in test_loader:
            outputs = model(inputs)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == true_value).sum().item()
            total += true_value.size(0)

    accuracy = correct / total
    print(f'Test accuracy: {accuracy:.2%}')

if __name__ == "__main__":
    main()