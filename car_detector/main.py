# main runner of the code
import torch 
import torch.nn as nn 
from data_process import * 
from torch.utils.data import TensorDataset, DataLoader 
from model import * 
from train import train_loop
import time 
import argparse 
from validate import * 
import torchvision.models as torch_models 

'''
Create arg parser 
'''
parser = argparse.ArgumentParser(description="Car Brand Detector Parsers")
parser.add_argument("--epoch", type=int, default=20)
parser.add_argument("--model", type=str, default="CNN")
parser.add_argument("--test_only", type=bool, default=False)

args=parser.parse_args() 

print(f"Args\n{args}")


config= {
    "img_reshape_size": 224, 
    "num_layers": 3,
    "hidden_dim": [64,128,256],
    "batch_size": 64, 
    "epoch": args.epoch, 
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}


def main():
    # retrieve data 
    print("----Retrieving Data-----")
    training_data, training_labels= torch.load("/Users/xiang/Desktop/Car-Brand-Detector/car_detector/data/train_data.pt")
    test_data, test_labels= torch.load("/Users/xiang/Desktop/Car-Brand-Detector/car_detector/data/test_data.pt")


    # create dataset 
    print("------Creating dataloder-----")
    train_dataset= TensorDataset(training_data,training_labels)
    train_loader= DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    
    test_dataset= TensorDataset(test_data, test_labels)
    test_loader= DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)


    


    # calcualte number of classes in the label 
    train_num_classes= (training_labels.max().item())+1  

    
    # get model
    if args.model=="CNN":
        model= CNN(num_classes=train_num_classes, hidden_dim=config["hidden_dim"]) 
    elif args.model=="ResNet":
        # ResNet
        block= ResidualBlock
        model= ResNet(ResidualBlock=ResidualBlock, num_classes=train_num_classes)
    elif args.model=="vgg16":
        model= torch_models.vgg16(pretrained=True) 
        model.classifier[6]=nn.Linear(4096, train_num_classes)
        print("-------Model Configuration-----")
        print(model)



    if (args.test_only):
        print("----Only testing----")
        ckpt= torch.load(f"/Users/xiang/Desktop/Car-Brand-Detector/car_detector/model_ckpt/20 Epoch ResNet.ckpt")
        model.load_state_dict(ckpt["model_state_dict"])
        run_test(model=model, test_loader=test_loader)
        exit() 

    # training loop 
    model= model.to(device)
    print("-----Training Started-----")
    train_start= time.time() 
    train_loop(train_loader=train_loader, model=model, epochs=config["epoch"], device=config["device"])
    train_end= time.time()
    print(f"------Training took: {round((train_end-train_start)/60,4)} minutes------")
    
    # visualize the predicted sample and the corresponding label 


if __name__ == "__main__":
    main() 