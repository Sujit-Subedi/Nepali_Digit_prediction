from Dataloader import Dataloading
import torch.nn as nn




class CNNModel(nn.Module):
    def __init__(self):
        super().__init__() # Inhereite all the class/function under nn.Module
        self.conv1 = nn.Conv2d(1,32,kernel_size=(3,3),stride=1,padding=1) # Padding add one additional pixels value toward the end. Filter and stried tends to reduce the spatial dimension by 1.
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv2d(32,32,kernel_size=(3,3),stride=1,padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2))

        self.flat = nn.Flatten()

        # ----- Feature extraction and pooling layer completed---- passing to neural network/fully connected layer.
        self.fc3 = nn.Linear(8192,512)
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.5)

        self.fc4 = nn.Linear(512,10)

    def forward(self, x):
        # input 3x32x32, output 32x32x32
        x = self.act1(self.conv1(x))
        x = self.drop1(x)
        # input 32x32x32, output 32x32x32
        x = self.act2(self.conv2(x))
        # input 32x32x32, output 32x16x16
        x = self.pool2(x)
        # input 32x16x16, output 8192
        x = self.flat(x)
        # input 8192, output 512
        x = self.act3(self.fc3(x))
        x = self.drop3(x)
        # input 512, output 10
        x = self.fc4(x)
        return x



