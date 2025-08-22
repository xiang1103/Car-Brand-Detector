from datasets import load_dataset
import torch 
from torchvision import transforms 
import numpy as np 


def process_data(reshape_size=224) -> dict[torch.Tensor]:
    ''' 
    Args: 
    reshape_size: size the images are reshaped into 
    ''' 
    ds = load_dataset("tanganke/stanford_cars") 


    train_data= ds["train"] 
    test_data = ds["test"]  

    # data each is categorized with image and label inside a dictionary 

    # transform the images 
    image_transforms= transforms.Compose([
    transforms.Resize((reshape_size, reshape_size)),  # Resize 
    transforms.ToTensor(),           # Convert to tensor 
    ])

    
    normalize_transform = transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)

    # transform train images and take out grayscale images 
    transformed_train_data= list()  
    skipped_labels= list()  
    for i in range (len(train_data)):
        img= train_data[i]["image"] 
        transformed= image_transforms(img) 
        if transformed.shape == torch.Size([3, 224, 224]):
            transformed_train_data.append(normalize_transform(transformed))
        else:
            skipped_labels.append(i)   

    # get train labels 
    train_labels=[]
    pointer=0 
    for i in range(len(train_data)):
        if pointer<len(skipped_labels):
            if i!= skipped_labels[pointer]:
                train_labels.append(train_data[i]["label"])  # Collect labels 
            else:
                pointer += 1 
        else:
            train_labels.append(train_data[i]["label"])
    
    # 8126 x 3 x 244 x 244  (B x C x H x W)
    train_labels = torch.tensor(train_labels)
    train_data= torch.stack(transformed_train_data)


    # transform test images  
    transformed_test_data= list()  
    skipped_labels= list()  
    for i in range (len(test_data)):
        img= test_data[i]["image"] 
        transformed= image_transforms(img) 
        if transformed.shape == torch.Size([3, 224, 224]):
            transformed_test_data.append(normalize_transform(transformed))
        else:
            skipped_labels.append(i)  

    test_labels=[]
    pointer=0 

    for i in range(len(test_data)):
        if pointer<len(skipped_labels):
            if i!= skipped_labels[pointer]:
                test_labels.append(test_data[i]["label"])  # Collect labels 
            else:
                pointer += 1 
        else:
            test_labels.append(test_data[i]["label"])
    
    # [8025]
    test_labels= torch.tensor(test_labels)
    # 8025 x 3 x 244 x 244
    test_data= torch.stack(transformed_test_data)

    
    return ({
        "train_data": train_data, 
        "test_data": test_data,
        "train_labels":train_labels,
        "test_labels": test_labels
    })


def denormalize(img:torch.Tensor, mean= [0.5,0.5,0.5], std=([0.5,0.5,0.5]) ) -> None:
    ''' 
    img shape: CxHxW i.e.: 3 x 224 x 224 
    Denormalize the img from [-1,1] to [0,1]
    '''
    # define mean and std 
    

    # transform for element wise multiplication with the img tensor 
    mean=torch.tensor(mean).view(3,1,1)
    std= torch.tensor(std).view(3,1,1)

    return img * std + mean


# if __name__ == "__main__":
#     process_data() 


