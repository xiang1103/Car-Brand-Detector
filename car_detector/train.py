import torch 
import torch.nn as nn 
from model import * 
from tqdm import tqdm, trange 
import numpy as np 
import matplotlib.pyplot as plt 
from settings import * 

def train_loop(train_loader, model, epochs, device): 
    model.train() 

    # establish loss function 
    loss_fn= nn.CrossEntropyLoss() 
    optimizer= torch.optim.Adam(params=model.parameters(), lr=0.001)
    loss_items=list() 
    running_loss= 0.0 
    batches =0 
    for epoch in trange(epochs):
        for images, label in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            images=images.to(device)
            label= label.to(device) 
            # model prediction 
            output= model(images)
            loss= loss_fn(output,label)
            loss_items.append(loss.item())
            running_loss+=loss.item() 

            # optimize 
            optimizer.zero_grad() 
            loss.backward() 
            optimizer.step() 
            batches+=1 

        # compute average loss 
        avg_loss = round(running_loss / batches, 4) 
        print(f"\nEpoch {epoch+1} | Avg Loss: {avg_loss}")


    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f"/Users/xiang/Desktop/Car-Brand-Detector/car_detector/model_ckpt/{epochs} Epoch CNN.ckpt"
    )

    print("----Model Saved----")

    if (visualize_graph):
        # visualize the loss graph 
        visualize_loss(loss_items=loss_items, name=f"{epochs} Epoch CNN")
    


def visualize_loss(loss_items, name):
    plt.clf() 
    plt.plot(loss_items)
    plt.xlabel("Batches")
    plt.ylabel("Loss") 
    plt.grid(True)
    plt.title(name)
    plt.savefig(f"/Users/xiang/Desktop/Car-Brand-Detector/car_detector/results/loss_graphs/{name}.jpg")

    