# main runner of the code
import torch 
import torch.nn as nn 
from data_process import * 
from torch.utils.data import TensorDataset, DataLoader 


config= {
    "img_reshape_size": 224, 
    "num_layers": 3,
    "hidden_sizes": [16,32,64],
    "batch_size": 64, 
}


def main():
    # retrieve data 
    data= process_data(reshape_size=config["img_reshape_size"])
    training_data= data["train_data"]
    training_labels= data["train_labels"]
    test_data= data["test_data"]
    test_labels= data["test_labels"]

    # create dataset 
    train_dataset= TensorDataset(training_data,training_labels)
    train_loader= DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    
    test_dataset= TensorDataset(test_data, test_labels)
    test_loader= DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    # calcualte number of classes in the label 
    num_classes= test_labels[-1].item() +1      

    # get model 

    # establish loss function 
    loss_fn= nn.CrossEntropyLoss() 
    optimizer= torch.optim.Adam(params=model.parameter())


if __name__ == "__main__":
    main() 