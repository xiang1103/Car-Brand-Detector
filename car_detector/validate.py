''' 
Test the accuracy of the model 
'''
import torch 
from model import * 
import tqdm 

device= "cpu" 


def save_samples():
    pass 
 
def run_test(model, test_loader):
    model.eval() 
    correct=0 
    total=0 
    for img, label in (test_loader):
        img=img.to(device)
        label= label.to(device)
        with torch.no_grad():
            outputs = model(img)                # shape: [batch_size, num_classes]
        _, predicted = torch.max(outputs, 1)   # take class with highest score
        
        # Update counts
        total += label.size(0)
        correct += (predicted == label).sum().item()

        print(f"Label min: {label.min()} | Label max: {label.max()}")
        print(outputs.shape, predicted[:5], label[:5])

        break 

    accuracy = 100 * correct / total
    print(f"Correct: {correct}")
    print(f"Total: {total}")
    print(f"Test Accuracy: {accuracy:.2f}%") 