import torch
import numpy as np
import torch.nn as nn


def load_data():
    hour_studied = torch.tensor([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]], dtype=torch.float32)
    grades = torch.tensor([[10], [25], [28], [30], [31], [40], [55], [75], [85], [92]], dtype=torch.float32)

    return (hour_studied, grades)

def get_model():
    return nn.Linear(1,1)

def main():

    X, Y = load_data()
    model = get_model()

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    

    for _ in range(1000):
        prediction = model(X)
        loss = loss_fn(prediction, Y)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    print(model(torch.tensor([[6.5]])))

    



if __name__ == "__main__":
    main()
