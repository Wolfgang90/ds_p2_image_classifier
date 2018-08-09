import utility
import model

def main():
    results_argparse = utility.argparse_train()

    image_datasets, dataloaders = utility.load_data(results_argparse.data_directory)

    model_nn = model.create_model(results_argparse.arch, results_argparse.hidden_units, results_argparse.prob_dropout)

    #print(model_nn)





if __name__ == "__main__":
    main()