import torch
from torchvision import datasets, transforms
import argparse
import numpy as np
from PIL import Image
import json

def argparse_train():
    parser = argparse.ArgumentParser()

    parser.add_argument("data_directory", help="set directory to get the data from")
    parser.add_argument("--save_dir", help="set directory to save checkpoints", default = "./")
    parser.add_argument("--arch", help="choose model architecture", default="densenet121")
    parser.add_argument("--prob_dropout",type=float, help="choose dropout probability in hidden layer", default=0.3)
    parser.add_argument("--learning_rate",type=float , help="choose learning rate", default=0.001)
    parser.add_argument("--hidden_units", type=int, help="choose hidden units", default=512)
    parser.add_argument("--epochs", type=int, help="choose epochs", default=10)
    parser.add_argument('--gpu', action='store_true', default=False, dest='gpu', help='set gpu-training to True')


    results = parser.parse_args()
    print('data_directory     = {!r}'.format(results.data_directory))
    print('save_dir     = {!r}'.format(results.save_dir))
    print('arch     = {!r}'.format(results.arch))
    print('prob_dropout     = {!r}'.format(results.prob_dropout))
    print('learning_rate     = {!r}'.format(results.learning_rate))
    print('hidden_units     = {!r}'.format(results.hidden_units))
    print('epochs     = {!r}'.format(results.epochs))
    print('gpu_training     = {!r}'.format(results.gpu))
    
    return results

def argparse_predict():
    parser = argparse.ArgumentParser()

    parser.add_argument("image_path", help="set path to image for prediction")
    parser.add_argument("checkpoint", help="set checkpoint to load the model from")
    parser.add_argument("--top_k", type=int, help="return top K most likely classes", default=5)
    parser.add_argument("--category_names", help="use a mapping of categories to real names", default=None)
    parser.add_argument('--gpu', action='store_true', default=False, dest='gpu', help='set gpu-training to True')

    results = parser.parse_args()
    print('image_path     = {!r}'.format(results.image_path))
    print('checkpoint     = {!r}'.format(results.checkpoint))
    print('top_k     = {!r}'.format(results.top_k))
    print('category_names     = {!r}'.format(results.category_names))
    print('gpu_training     = {!r}'.format(results.gpu))
    
    return results

def load_data(data_directory):
    train_dir = data_directory + '/train'
    valid_dir = data_directory + '/valid'
    test_dir = data_directory + '/test'
    # Define transforms for the training, validation, and testing sets
    data_transforms = {'train': transforms.Compose([transforms.Resize((250,250)),
                                          transforms.RandomCrop((224,224)),
                                          transforms.RandomRotation(20),
                                          transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
                       'valid': transforms.Compose([transforms.Resize((224,224)),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
                       'test': transforms.Compose([transforms.Resize((224,224)),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
                      }

    # Load the datasets with ImageFolder
    image_datasets = {'train': datasets.ImageFolder(train_dir, transform = data_transforms['train']),
                      'valid': datasets.ImageFolder(train_dir, transform = data_transforms['valid']),
                      'test': datasets.ImageFolder(train_dir, transform = data_transforms['test'])
                     }

    # Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size = 64, shuffle=True),
                   'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size = 64, shuffle=True),
                   'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size = 64, shuffle=True),
                  }
    
    return image_datasets, dataloaders

def preprocess_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Load image
    image = Image.open(image_path)
    
    # scale image
    size = 256, 256
    image.thumbnail(size)
    
    # center crop
    width, height = image.size   # Get dimensions
    new_width = 224
    new_height = 224
    left = (width - new_width)/2
    
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2
    image = image.crop((left, top, right, bottom))
    
    np_image = np.array(image)
    
    # Normalize
    means = np.array([0.485, 0.456, 0.406])
    stdev = np.array([0.229, 0.224, 0.225])
    
    np_image = (np_image/255 - means)/stdev
    
    image_transposed = np_image.transpose(2,0,1)
    
    return image_transposed


def import_mapping(category_names):
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
        
    return cat_to_name