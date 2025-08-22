''' 
Test the accuracy of the model 
'''
import torch 
from model import * 
import numpy as np 
from settings import * 
import pickle 

device= "cpu" 


def save_samples(fail_samples:list[tuple], success_samples: list[tuple]) -> None:
    ''' 
    Save success or failed prediction samples 
    The samples are saved as an array of tuple (img, predicted_label, actual label)

    '''
    print("----saving failure and success samples-----")
    with open (f"/Users/xiang/Desktop/Car-Brand-Detector/car_detector/results/correct_samples/{len(success_samples)} {setting_model} Success Samples.pkl", "wb") as f:
        pickle.dump(success_samples,f)
    with open(f"/Users/xiang/Desktop/Car-Brand-Detector/car_detector/results/incorrect_samples/{len(fail_samples)} {setting_model} Fail Samples.pkl", "wb") as f:
        pickle.dump(fail_samples,f)
    
    print("-----finished saving data-----")
    print(f"Length of failed samples: {len(fail_samples)}")
    print(f"Length of success samples: {len(success_samples)}")
 


def run_test(model, test_loader):
    success_samples=[]
    failed_samples=[] 
    model.eval() 
    correct=0 
    total=8025 
    for img, label in (test_loader):
        img=img.to(device)
        label= label.to(device)
        with torch.no_grad():
            outputs = model(img)                # shape: [batch_size, num_classes]
        _, predicted = torch.max(outputs, 1)   # take class with highest score
        
        # Update counts
        correct += (predicted == label).sum().item()

        # save the examples 
        for i in range(len(img)):
            p= np.random.rand()
            if label[i].item()==predicted[i].item():
                if p<0.8:
                    success_samples.append((img[i],label[i].item()))
            else:
                if p<0.05:
                    failed_samples.append((img[i],predicted[i].item(), label[i].item()))
    
    if (save_samples):
        save_samples(fail_samples=failed_samples, success_samples=success_samples)
            

    
    accuracy = 100 * correct / total
    print(f"Correct: {correct}")
    print(f"Total: {total}")
    print(f"Test Accuracy: {accuracy:.2f}%") 