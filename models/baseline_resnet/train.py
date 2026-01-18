import os
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
import torch.optim as optim
from PIL import Image

# ---------------- CONFIG ----------------
DATA_DIR = "data/raw/sdnet"
BATCH_SIZE = 8
EPOCHS = 5
LR = 1e-4
MAX_SAMPLES = 3000        # speed-up baseline
NUM_WORKERS = 0           # REQUIRED on Windows
# ----------------------------------------


class SDNETBinaryDataset(Dataset):
    def __init__(self, root, transform=None):
        self.samples = []
        self.transform = transform

        for surface in ["Decks", "Pavements", "Walls"]:
            cracked_dir = os.path.join(root, surface, "Cracked")
            noncracked_dir = os.path.join(root, surface, "Non-cracked")

            for f in os.listdir(cracked_dir):
                self.samples.append((os.path.join(cracked_dir, f), 1))

            for f in os.listdir(noncracked_dir):
                self.samples.append((os.path.join(noncracked_dir, f), 0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor()
])

dataset = SDNETBinaryDataset(DATA_DIR, transform)
dataset.samples = dataset.samples[:MAX_SAMPLES]

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

print(f"[INFO] Training samples: {len(dataset)}")

model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for i, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if i % 25 == 0:
            print(
                f"Epoch {epoch+1}/{EPOCHS} | "
                f"Batch {i}/{len(loader)} | "
                f"Loss {loss.item():.4f}"
            )

    print(f"Epoch {epoch+1} COMPLETE | Avg Loss: {total_loss:.4f}\n")

print("Training complete")
torch.save(model.state_dict(), "models/baseline_resnet/resnet18_sdnet_baseline.pth")
print("Model saved")
