# main runner of the code
import torch 
import torch.nn as nn 
from data_process import * 
from torch.utils.data import TensorDataset, DataLoader 
from model import * 
from train import train_loop
import time 

config= {
    "img_reshape_size": 224, 
    "num_layers": 3,
    "hidden_dim": [16,32,64],
    "batch_size": 64, 
    "epoch": 20, 
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}


def main():
    # retrieve data 
    print("----Retrieving Data-----")
    data= process_data(reshape_size=config["img_reshape_size"])
    training_data= data["train_data"]
    training_labels= data["train_labels"]
    test_data= data["test_data"]
    test_labels= data["test_labels"]

    
    # create dataset 
    print("------Creating dataloder-----")
    train_dataset= TensorDataset(training_data,training_labels)
    train_loader= DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    
    test_dataset= TensorDataset(test_data, test_labels)
    test_loader= DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    # calcualte number of classes in the label 
    num_classes= test_labels[-1].item() +1      

    # get model 
    model= CNN(num_classes=num_classes, hidden_dim=config["hidden_dim"]) 
    
    # training loop 
    print("-----Training Started-----")
    train_start= time.time() 
    train_loop(train_loader=train_loader, model=model, epochs=config["epoch"], device=config["device"])
    train_end= time.time()
    print(f"------Training took: {round((train_end-train_start),4)} seconds------")
    
    # visualize the predicted sample and the corresponding label 


if __name__ == "__main__":
    main() 