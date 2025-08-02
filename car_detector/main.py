# main runner of the code
import torch 
import torch.nn as nn 
import numpy as np 
from data_process import * 


config= {
    "img_reshape_size": 224, 
    "num_layers": 6,
    "hidden_sizes": [],
    "batch_size": 64, 
}


def main():
    

    # retrieve data 
    data= process_data(reshape_size=config["img_reshape_size"])
    training_data= data["train_data"]
    training_labels= data["train_labels"]
    test_data= data["test_data"]
    test_labels= data["test_labels"]  



if __name__ == "__main__":
    main() 