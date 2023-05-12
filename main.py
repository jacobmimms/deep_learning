import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import os
import copy
import shutil


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def separate_data():
    for image in os.listdir("images"):
        # if the firist letter of the image name is lowercase, it is a dog image
        if image[0].islower():
            shutil.copy("images/" + image, "class_images/dog/" + image)
        if image[0].isupper():
            shutil.copy("images/" + image, "class_images/cat/" + image)



def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs-1))
        print("-"*10)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                # forward pass
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    if is_inception and phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    # backward pass + optimize if only in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item()*inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss/len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double()/len(dataloaders[phase].dataset)
            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)


    # save the best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), "model.pt")
    return model, val_acc_history



if __name__ == "__main__":
    num_classes = 2
    num_epochs = 10
    batch_size = 10
    learning_rate = 0.001
    model = models.resnet18()
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.fc.parameters(), lr=learning_rate, momentum=0.9)

    # data augmentations 
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
    }
    image_datasets = datasets.ImageFolder("class_images/")
    train_data, test_data = torch.utils.data.random_split(image_datasets, [int(len(image_datasets)*0.8), int(len(image_datasets)*0.2)])
    # apply data transformations to the datasets
    train_data.dataset.transform = data_transforms['train']
    test_data.dataset.transform = data_transforms['val']

    # create dataloaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
    dataloaders = {'train': train_loader, 'val': test_loader}

    train_model(model, dataloaders, criterion, optimizer, num_epochs=num_epochs, is_inception=False)
    