import os
import time
import random
# import datetime
import numpy as np
import datetime
from datetime import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torchvision
from torch import nn, optim

from model import AutoEncoder
from utilities.dataloader import Dataset
from utilities.common import get_datetime_from_ymd_string

intervals = (
    ('weeks', 604800),  # 60 * 60 * 24 * 7
    ('days', 86400),    # 60 * 60 * 24
    ('hours', 3600),    # 60 * 60
    ('minutes', 60),
    ('seconds', 1),
    )


def display_time(seconds, granularity=2):
    result = []

    for name, count in intervals:
        value = seconds // count
        if value:
            seconds -= value * count
            if value == 1:
                name = name.rstrip('s')
            result.append("{} {}".format(value, name))
    return ', '.join(result[:granularity])


random.seed(0)
data_dir_path = '/media/mn/WD4TB/topo/survey_dems/xyz_data/FULL'
files = [f for f in os.listdir(data_dir_path) if f.endswith('_FULL.npy')]

ids = [x.replace('_FULL.npy', '') for x in files]

date_ids = []
for x in ids:
    date_ids.append(get_datetime_from_ymd_string(x))
ids = []
labels = []
diffs = []
for i in range(len(date_ids) - 1):
    d0 = sorted(date_ids)[i]
    d1 = sorted(date_ids)[i+1]
    k_d0 = f'{d0.year}{"{:02d}".format(d0.month)}{"{:02d}".format(d0.day)}'
    k_d1 = f'{d1.year}{"{:02d}".format(d1.month)}{"{:02d}".format(d1.day)}'
    diff = d1 - d0
    # print(k_d0, k_d1, diff.days)
    ids.append(k_d0)
    labels.append(k_d1)
    diffs.append(diff.days)

df = pd.DataFrame({'id': ids, 'label': labels, 'difference': diffs})
df = df.sample(frac=1).reset_index(drop=True)
df = df[df.difference < 45]
train_count = int(len(df) * 0.8)

train_ids = df.id.values[:train_count]
train_labels = df.label.values[:train_count]
labels_dict = {}
for i in range(len(train_ids)):
    labels_dict[train_ids[i]] = train_labels[i]
train_set = Dataset(list_ids=train_ids, labels=labels_dict)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

val_ids = df.id.values[train_count:]
val_labels = df.label.values[train_count:]
labels_dict = {}
for i in range(len(val_ids)):
    labels_dict[val_ids[i]] = val_labels[i]
val_set = Dataset(list_ids=val_ids, labels=labels_dict)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=64, shuffle=True)

data_iter = iter(train_loader)
images, labels = data_iter.next()

# figure = plt.figure()
# num_of_images = 60
# for index in range(1, num_of_images + 1):
#     plt.subplot(6, 10, index)
#     plt.axis('off')
#     plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')
# plt.show()

model = AutoEncoder()
print(model)

criterion = nn.MSELoss()
images, labels = next(iter(train_loader))

print(images.shape)
print(labels.shape)
# exit()

logps = model(images)  # log probabilities
loss = criterion(logps, labels)  # calculate the NLL loss


print('Before backward pass: \n', model[0].weight.grad)
loss.backward()
print('After backward pass: \n', model[0].weight.grad)
exit()

optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
time0 = time.time()
epochs = 15
for e in range(epochs):
    running_loss = 0
    for images, labels in train_loader:
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)

        # Training pass
        optimizer.zero_grad()

        output = model(images)
        loss = criterion(output, labels)

        # This is where the model learns by backpropagating
        loss.backward()

        # And optimizes its weights here
        optimizer.step()

        running_loss += loss.item()
    else:
        print(f"Epoch {e} - Training loss: {running_loss / len(train_loader)}")
print(f"Training Time (in minutes) = {(time.time() - time0) / 60}")

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
