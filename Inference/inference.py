import torch
from torchvision import io
from torchvision import transforms
from CNN_model import CNNModel
from PIL import Image
import torch.nn as nn

def Loadmodel(Image_path):
    model = CNNModel()
    model.load_state_dict(torch.load('Digit_classifiaction_model.pth'))
    model.eval()
    image_tensor = Image.open(Image_path)
    input_transforms = transforms.Compose([
                                transforms.Grayscale(),
                                transforms.ToTensor(),
                                transforms.Resize((32,32)),  # Convert to PyTorch tensor
                                transforms.Normalize((0.5,), (0.5,))  # Normalize to [0, 1]
                            ])
    image_tensor = transforms.functional.invert(image_tensor)
    transformed_image = input_transforms(image_tensor).unsqueeze(0)   #transformed_image = transformed_image.unsqueeze(0)
    #transformed_image = transformed_image.to(model.device)

    with torch.no_grad():
        predicted_value = model(transformed_image)
        
        output = torch.nn.functional.softmax(predicted_value, dim=1)

        predicted_class = torch.argmax(output, dim=1)  # Get the class with maximum probability
        # Get the probability of the predicted class
        confidence = output[0, predicted_class] * 100  # Assuming output is a 2D tensor
       
        print(f'Model predicted {predicted_class.item()} with confidence of {confidence.item()}')
    return 
       

value = Loadmodel('nine.png')
print(value)











