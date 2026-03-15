import cv2
import torch
import numpy as np
import os
from src.dataset import train_transform 

def detect_and_flag(image_path, model, threshold=0.40, pixel_threshold=45):
    #Dobry wynik dla 0.40 i 45 - po około 70%
    # DODAJ TĘ LINIĘ (aby wiedzieć czy użyć procesora czy karty graficznej):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    
    if not os.path.exists(image_path):
        print(f"Błąd: Plik {image_path} nie istnieje.")
        return False, None

    raw_img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

    # Teraz 'train_transform' i 'device' będą już rozpoznane:
    input_tensor = train_transform(image=img_rgb)['image'].unsqueeze(0).to(device)
    

    # Inference
    with torch.no_grad():
        prediction = model(input_tensor)
        mask = (prediction > threshold).cpu().numpy().squeeze()
    
    # Decision Logic
    defect_pixel_count = np.sum(mask)
    is_defective = defect_pixel_count > pixel_threshold
    
    status = "REJECT - DEFECTIVE" if is_defective else "PASS - GOOD"
    
    # Visual Output
    if is_defective:
        print(f"Flagged: {image_path} | Pixels: {defect_pixel_count} | Status: {status}")
        # Optionally overlay the mask on the original image for review
    
    return is_defective, mask