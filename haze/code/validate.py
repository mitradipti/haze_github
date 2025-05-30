import torch
from train import Unet, HazyImageDataset  # Import the model and dataset class
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn as nn

# Load the saved model
model = Unet()
model.load_state_dict(torch.load("unet_model_cpu_updated_ntire.pth"))
model.eval()  # Set the model to evaluation mode

# Validation Dataset and DataLoader
transform = transforms.Compose([
    transforms.ToTensor(),
])
hazy_image_dir = r'C:\Users\oishi\Documents\haze\haze\resized_outdoor'
transmission_map_dir = r'C:\Users\oishi\Documents\haze\haze\transmission_maps_ntire_outdoor'

dataset = HazyImageDataset(hazy_image_dir, transmission_map_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=False)  # No shuffling for validation

# Define the loss function
criterion = nn.MSELoss()

# Validation Loop
val_loss = 0.0
with torch.no_grad():  # Disable gradient computation for validation
    for inputs, labels in dataloader:
        inputs, labels = inputs.to("cpu"), labels.to("cpu")  # Ensure data is on CPU
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        val_loss += loss.item()

print(f"Validation Loss: {val_loss / len(dataloader)}")