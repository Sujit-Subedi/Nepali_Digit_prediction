import torch.utils
from torch.utils.data import Dataset
import pandas as pd
import torch
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms



class Dataloading():

    def __init__(self,image_dir):
        self.image_dir = image_dir
    """
    Initializes the dataloader class

    Args:
    Image root directory(String).
        - Directory path containing imgae data should be in hiearchy as required by ImageLoader package.
    """


    def Data_load(self):
        """
        Function to make dataset and load into the dataloader 

        Parameter:
        Image Directory

        Returns:

        Tuple containing Train and test data loader that provide tensor value of image for model in batch.

        """
        
        #Compose class composes multiple transformation.
        train_transforms = transforms.Compose([
                                        transforms.ToTensor()]) # Convert that image to tensor and scale it in the range of 0-1.

        test_transforms = transforms.Compose([
                                        transforms.ToTensor()])

        train_dataset = torchvision.datasets.ImageFolder(self.image_dir + '/train', transform = train_transforms)
        test_dataset = torchvision.datasets.ImageFolder(self.image_dir + '/test', transform = test_transforms)


        train_Data_Loader = DataLoader(train_dataset,batch_size=64,shuffle=True)
        test_Data_Loader = DataLoader(test_dataset,batch_size=64,shuffle=True)

        for batch_feature, batch_labels in train_Data_Loader:
            print(batch_feature,batch_labels)
            break
        
        print('----Data loaded Sucessfully to train and test loader ------')

        return train_Data_Loader,test_Data_Loader
    

data = Dataloading('Nepali-Digit')
data.Data_load()





