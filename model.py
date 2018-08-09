import torch
from torch import nn, optim
from torchvision import models

from collections import OrderedDict

def create_model(arch, hidden_units , prob_dropout):
    # Create model
    model = eval("models." + arch + "(pretrained=True)")


    model.epochs = 0
    
    # To prevent backprop through parameters freeze parameters
    for param in model.parameters():
        param.requires_grad = False
        
    input_units = model.classifier.in_features

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_units, hidden_units)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(prob_dropout)),
        ('fc2', nn.Linear(hidden_units,102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier
    
    return model

def validation(model, loader, criterion, device):
    with torch.no_grad():
        accuracy = 0
        loss = 0
        for inputs, labels in loader:                            
            inputs, labels = inputs.to(device), labels.to(device)

            output = model.forward(inputs)
            loss += criterion(output, labels).item()

            ps = torch.exp(output)
            equality = (labels.data == ps.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()

        loss = loss/len(loader)
        accuracy = accuracy/len(loader)
      
    return loss, accuracy   