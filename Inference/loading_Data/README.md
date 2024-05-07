## Creating Dataloaders for Image Classification

This document explains the Python code for creating dataloaders to load and prepare image datasets for training and testing a machine learning model.

**Code Breakdown:**

1. **Imports:**
   - Necessary libraries for working with tensors, datasets, dataloaders, and image transformations are imported.

2. **Dataloading Class:**
   - The `Dataloading` class is defined:
     - `__init__` method:
       - Initializes the class by taking the image directory path as input.
       - Performs basic validation to ensure the directory exists.
     - `load_data` method:
       - Defines the core logic for loading the data:

         - Creates two transformation chains (`train_transforms` and `test_transforms`) using `transforms.Compose`. 
         - The current implementation includes only `ToTensor` to convert images to tensors and scale pixel value range from 0-255 to 0-1.Additional Normalization is not required.
         - Loads training and testing datasets using `ImageFolder`. `ImageFolder` requires secific directory type to make dataset properly.
         - Creates dataloaders for training and testing using `DataLoader`where batch size and shuffling boolean (enabled for training only) is passed.
         - Optionally prints the shapes of a sample batch for debugging purposes.
         - Returns the train and test dataloaders as a tuple.

