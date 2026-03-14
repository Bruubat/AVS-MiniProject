import segmentation_models_pytorch as smp
import torch

# 1. Initialize Model (U-Net with ResNet34 backbone)
model = smp.Unet(
    encoder_name="resnet34", 
    encoder_weights="imagenet", 
    in_channels=3, 
    classes=1, 
    activation='sigmoid'
)

def get_unet_model():
    # Definiujemy model U-Net z enkoderem ResNet34
    model = smp.Unet(
        encoder_name="resnet34", 
        encoder_weights="imagenet", 
        in_channels=3, 
        classes=1, 
        activation='sigmoid'
    )
    return model

# 2. Training Loop Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = smp.losses.DiceLoss(smp.losses.BINARY_MODE)

def train_one_epoch(loader):
    model.train()
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

# Usage: 
# dataset = ToothbrushDataset('train_folder', 'ground_truth_folder', transform=train_transform)
# loader = DataLoader(dataset, batch_size=8, shuffle=True)
# train_one_epoch(loader)