import os
import torch
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import shutil

DATA_DIR = "data/raw/sdnet"
MODEL_PATH = "models/baseline_resnet/resnet18_sdnet_baseline.pth"
OUTPUT_DIR = "evaluation/false_positives"

os.makedirs(OUTPUT_DIR, exist_ok=True)

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor()
])

model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

def collect_false_positives():
    count = 0
    for surface in ["Decks", "Pavements", "Walls"]:
        noncracked_dir = os.path.join(DATA_DIR, surface, "Non-cracked")
        for f in os.listdir(noncracked_dir):
            img_path = os.path.join(noncracked_dir, f)
            img = Image.open(img_path).convert("RGB")
            x = transform(img).unsqueeze(0)

            with torch.no_grad():
                pred = model(x).argmax(dim=1).item()

            # False positive: predicted crack on non-crack
            if pred == 1:
                dst = os.path.join(OUTPUT_DIR, f"{surface}_{f}")
                shutil.copy(img_path, dst)
                count += 1

            if count >= 50:
                return

collect_false_positives()
print("False positives collected")
