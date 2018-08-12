import torch
from torch import nn, optim
from torchvision import models
import numpy as np
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


def train_model(model, trainloader, learning_rate, epochs, gpu, print_every = 40, validloader=None):
    steps = 0
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    
    model.train()
    # Train model on gpu if available
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if gpu:
        device = "cuda:0"
    else:
        device = "cpu"
    model.to(device)
        
    for e in range(epochs):
        running_loss = 0
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if steps % print_every == 0:
                #print("calculate accuracy")
                
                model.eval()
                
                # Calculate loss and accuracy of validation set
                if validloader:
                    loss_val, accuracy_val = validation(model, validloader, criterion, device)                    
                        
                    print("Epoch: {}/{}\t".format(e+1,epochs),
                          ("Step: {}\t".format(steps)),
                          ("Loss (test): {:.4f}\t".format(running_loss/print_every)),
                         ("Loss (val): {:.4f}\t".format(loss_val)),
                         ("Accuracy (val): {:.4f}\n".format(accuracy_val)))
                else:
                    print("Epoch: {}/{}\t".format(e+1,epochs),
                          ("Loss: {:.4f}".format(running_loss/print_every)))
                running_loss = 0
                
                model.train()
        model.epochs += 1
        
        return model, optimizer
    
    
def save_model(model, optimizer):
    checkpoint = {'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'n_epochs': model.epochs,
                  'optimizer' : optimizer.state_dict}

    torch.save(checkpoint, 'checkpoint.pth')

    print("model saved to checkpoint.pth")
    
    
def load_model(checkpoint, gpu): 
    
    if gpu:
        device = "cuda:0"
    else:
        device = "cpu"  
    
    if device == "cuda:0":
        checkpoint = torch.load(checkpoint, map_location={'cpu': 'cuda:0'})
    else:
        checkpoint = torch.load(checkpoint, map_location={'cuda:0': 'cpu'})
    model = models.densenet121(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
        
    classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(1024, 512)),
    ('relu1', nn.ReLU()),
    ('dropout1', nn.Dropout(0.3)),
    ('fc2', nn.Linear(512,102)),
    ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier
    
    #model.classifier = checkpoint['model_classifier']
    model.load_state_dict(checkpoint['state_dict'])
    
    model.epochs = checkpoint['n_epochs']    
    model.class_to_idx = checkpoint['class_to_idx']    
    
    return model


def predict(image, model, top_k, gpu, cat_to_name=None):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''   
    if gpu:
        device = "cuda:0"
    else:
        device = "cpu"  
    
    # Add additional "image index dimension"
    image = np.expand_dims(image,0)
    
    image_tensor = torch.FloatTensor(image)
    
    model.eval()
    
    with torch.no_grad():
        model.to(device)
        image_tensor = image_tensor.to(device)
        output = model.forward(image_tensor)
        
        ps = torch.exp(output)
        
        probs, probs_index = ps.topk(top_k)
        
        probs, probs_index = probs.tolist()[0], probs_index.tolist()[0]
        
        class_to_idx = model.class_to_idx
        idx_to_class = {y:x for x,y in class_to_idx.items()}
        
        classes = [idx_to_class[x] for x in probs_index] 
        
        # Use a mapping of categories to real names
        if cat_to_name:
            classes = [cat_to_name[str(x)] for x in classes]
    
    model.train()   
    return probs, classes