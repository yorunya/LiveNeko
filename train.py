"""Fine-tune google/vit-base-patch16-224 on the extracted frames in ./dataset.

dataset/ is produced by sample.py: one subfolder per tag holding the sampled
frames (.jpg). The label set is the names of the subfolders that contain
frames (empty folders are skipped with a warning). The ViT classifier head is
replaced, and the fine-tuned model + processor are saved to ./model.
"""
import glob
import os
import random

import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset, random_split
from transformers import ViTForImageClassification, ViTImageProcessor

DATASET_DIR = "dataset"
MODEL_DIR = "model"
BASE_MODEL = "google/vit-base-patch16-224"

EPOCHS = 5
BATCH_SIZE = 32
LR = 2e-5
VAL_FRACTION = 0.4
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

frame_paths = {}  # label -> sorted image paths
for d in sorted(os.listdir(DATASET_DIR)):
    if not os.path.isdir(os.path.join(DATASET_DIR, d)):
        continue
    paths = sorted(
        p
        for ext in ("jpg", "jpeg", "png")
        for p in glob.glob(os.path.join(DATASET_DIR, d, f"*.{ext}"))
    )
    if paths:
        frame_paths[d] = paths
    else:
        print(f"Warning: {DATASET_DIR}/{d}/ has no frames, skipped")
label_names = sorted(frame_paths)
if not label_names:
    raise SystemExit(f"No frames found in {DATASET_DIR}/ (run sample.py first)")
label2id = {label: i for i, label in enumerate(label_names)}
id2label = {i: label for label, i in label2id.items()}
print(f"Labels: {label2id}")

# Load processor and model with a fresh classifier head
processor = ViTImageProcessor.from_pretrained(BASE_MODEL)
model = ViTForImageClassification.from_pretrained(
    BASE_MODEL, num_labels=len(label_names), ignore_mismatched_sizes=True
)
model.config.label2id = label2id
model.config.id2label = id2label  # saved with the model so pred.py gets real names
model.to(device)

# Load and preprocess the frames
frame_tensors, frame_labels = [], []
for label in label_names:
    for p in frame_paths[label]:
        with Image.open(p) as img:
            img = img.convert("RGB")
            pixel_values = processor(images=img, return_tensors="pt").pixel_values[0]
        frame_tensors.append(pixel_values)
        frame_labels.append(label2id[label])
    print(f"{label}: {len(frame_paths[label])} frames")

dataset = TensorDataset(torch.stack(frame_tensors), torch.tensor(frame_labels))

# Train/val split and training loop
val_size = max(1, int(len(dataset) * VAL_FRACTION))
train_size = len(dataset) - val_size
train_set, val_set = random_split(
    dataset, [train_size, val_size], generator=torch.Generator().manual_seed(SEED)
)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
# print(f"Dataset: {train_size} train / {val_size} val frames")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for pixel_values, y in train_loader:
        pixel_values, y = pixel_values.to(device), y.to(device)
        outputs = model(pixel_values=pixel_values, labels=y)
        outputs.loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += outputs.loss.item() * len(y)

    model.eval()
    correct = 0
    with torch.no_grad():
        for pixel_values, y in val_loader:
            pixel_values, y = pixel_values.to(device), y.to(device)
            preds = model(pixel_values=pixel_values).logits.argmax(-1)
            correct += (preds == y).sum().item()
    print(
        f"Epoch {epoch}/{EPOCHS}  "
        f"train_loss={total_loss / train_size:.4f}  "
        f"val_acc={correct / val_size:.3f}"
    )

# Save fine-tuned model and processor
model.save_pretrained(MODEL_DIR)
processor.save_pretrained(MODEL_DIR)
