import time
import random
import numpy as np
import pandas as pd

import torch
from torch import nn, optim

from model import AutoEncoder
from unet import UNet
from utilities.dataloader import Dataset
from utilities.data_io import make_sub_areas, make_dataset


random.seed(0)

# TODO: refactor data handling?
topo_grids = np.load('./init_scripts/interpolated_master_grid.npy')
topo_dates = np.load('./init_scripts/topo_dates.npy')
df_waves = pd.read_csv('./init_scripts/interpolated_master_waves.csv')
dataset = make_dataset(topo_grids, topo_dates, df_waves, 24, pairs=True)
random.shuffle(dataset)

train_ratio = 0.7
val_ratio = 0.1
test_ratio = 0.2

train_count = int(len(dataset) * train_ratio)
val_count = int(len(dataset) * val_ratio)
test_count = int(len(dataset) * test_ratio)

train_pairs = dataset[:train_count]
val_pairs = dataset[train_count:train_count+val_count]
test_pairs = dataset[-test_count:]

train_set = Dataset(pairs=train_pairs)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

val_set = Dataset(pairs=val_pairs)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=64, shuffle=True)

test_set = Dataset(pairs=test_pairs)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)

# model = AutoEncoder().double()
model = UNet().double()
print(model)

criterion = nn.MSELoss()

# images, labels = next(iter(train_loader))
# images = images.unsqueeze(1)
# labels = labels.unsqueeze(1)
# logps = model(images)
# loss = criterion(logps, labels)
# print('Before backward pass: \n', model.encoder[0].weight.grad)
# loss.backward()
# print('After backward pass: \n', model.encoder[0].weight.grad)

optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
time0 = time.time()
epochs = 15
for e in range(epochs):
    running_loss = 0
    # for images, labels in train_loader:
    for subarea1, subarea2, hs, tp, direction in train_loader:
        subarea1 = subarea1.unsqueeze(1)
        subarea2 = subarea2.unsqueeze(1)
        hs = hs.unsqueeze(1)
        tp = tp.unsqueeze(1)
        direction = direction.unsqueeze(1)

        # Training pass
        optimizer.zero_grad()

        output = model(subarea1)
        loss = criterion(output, subarea2)

        # This is where the model learns by backpropagating
        loss.backward()

        # And optimizes its weights here
        optimizer.step()

        running_loss += loss.item()
    else:
        print(f"Epoch {e} - Training loss: {running_loss / len(train_loader)}")
print(f"Training Time (in minutes) = {(time.time() - time0) / 60}")

exit()

# TODO: adapt model evaluation from MNIST example
images, labels = next(iter(val_loader))
img = images[0].view(1, 784)
with torch.no_grad():
    log_ps = model(img)

ps = torch.exp(log_ps)
probab = list(ps.numpy()[0])
print(f"Predicted Digit = {probab.index(max(probab))}")

correct_count, all_count = 0, 0
for images, labels in val_loader:
    for i in range(len(labels)):
        img = images[i].view(1, 784)
        with torch.no_grad():
            log_ps = model(img)

        ps = torch.exp(log_ps)
        probab = list(ps.numpy()[0])
        pred_label = probab.index(max(probab))
        true_label = labels.numpy()[i]
        if (true_label == pred_label):
            correct_count += 1
        all_count += 1

print("Number Of Images Tested =", all_count)
print("\nModel Accuracy =", (correct_count / all_count))

torch.save(model, './my_mnist_model.pt')
