from utility import argparse_train, load_data
from model import create_model, train_model, save_model

def main():
    results_argparse = argparse_train()

    image_datasets, dataloaders = load_data(results_argparse.data_directory)

    # Create the model
    model = create_model(results_argparse.arch, results_argparse.hidden_units, results_argparse.prob_dropout)
    
    # Train the model
    model, optimizer = train_model(model, dataloaders['train'], results_argparse.learning_rate, results_argparse.epochs, results_argparse.gpu, print_every = 40, validloader = dataloaders['valid'])
    
    # Save model
    model.class_to_idx = image_datasets['train'].class_to_idx
    save_model(model, optimizer)

if __name__ == "__main__":
    main()