# Car-Brand-Detector
Toy project on detecting car brands using public datasets. Start with simple models like CNN, ResNet, both training on scratch, and finetuning pretrained models. 
    - Can't try Vision Transformer because the data doesn't contain textual information. With textual information we can try using self-supervised learning, so the model learns to recognize the brand through textual context and output. 


## Data 
The data is [Stanford Car dataset from HuggingFace](https://huggingface.co/datasets/tanganke/stanford_cars). This is pretty old dataset with 8000+ training and testing images for 100 classes. 

# Running experiments  
- Data is by default using Stanford Car Dataset. You can change that in car_detector/main.py 
- python main.py --epoch x --model "CNN"/"ResNet"

## Models 
-[x] Training from scratch CNN   
-[x] Training from scratch ResNet-16   
-[ ] Visualize the heatmap of weights on image to see where the model is paying attention to  
-[ ] Finetuning a pretrained model  
-[ ] Test performance on adversial attacks 