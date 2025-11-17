# imports
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedGroupKFold
from PIL import Image
import timm
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# load weights offline
from safetensors.torch import load_file

# cosine annealing with warmup learning rate scheduler
def cosine_annealing_with_warmup(
    optimizer, warmup_epochs, total_epochs, base_lr, final_lr=1e-6
):
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch) / float(max(1, warmup_epochs))

        progress = float(current_epoch - warmup_epochs) / float(
            max(1, total_epochs - warmup_epochs)
        )
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# Global configuration
class Config:
    seed = 42
    n_folds = 5
    img_size = 224
    batch_size = 16
    epochs = 40
    lr = 5e-4
    model_name = "convnext_large.fb_in1k"  # image size 224
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 2
    targets = ["Dry_Green_g", "GDM_g", "Dry_Total_g"]


# Dataset Class
class BiomassDataset(Dataset):
    def __init__(self, df, transform=None, is_test=False):
        self.transform = transform
        self.is_test = is_test
        # check if kaggle/input exists in the terminal
        if os.path.exists("/kaggle/input/csiro-biomass/"):
            self.input_dir = "/kaggle/input/csiro-biomass/"
        # else running locally
        else:
            self.input_dir = "./"
        
        self.df = (
            df.pivot_table(
                index=["image_path", "State", "Pre_GSHH_NDVI", "Height_Ave_cm"],
                columns="target_name",
                values="target",
                aggfunc="first",
            ).reset_index()
            if not is_test
            else df[["image_path"]].drop_duplicates().reset_index(drop=True)
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        img = Image.open(self.input_dir + row.image_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        if self.is_test:
            return img, row["image_path"]

        targets = torch.tensor(
            np.log1p(
                [
                    row["Dry_Green_g"],
                    row["GDM_g"],
                    row["Dry_Total_g"],
                ]
            ),
            dtype=torch.float32,
        )
        return img, targets



# Model Class
class BiomassModel(nn.Module):
    def __init__(self, num_targets=3):
        super().__init__()
        kaggle_dataset = Config.model_name.replace("_", "-").replace(".", "-")
        # try to load model weights as Kaggle dataset
        local_weight_path = f"/kaggle/input/{kaggle_dataset}-weights-offline/{Config.model_name}.safetensors"

        self.backbone = timm.create_model(
            Config.model_name, pretrained=False, num_classes=0
        )

        if os.path.exists(local_weight_path):
            weights = load_file(local_weight_path)
            self.backbone.load_state_dict(weights, strict=False)
            print(f"✅ Loaded {Config.model_name} weights locally (offline).")
        else:
            print(
                f"⚠️ Local weights not found, falling back to pretrained download {Config.model_name}."
            )
            self.backbone = timm.create_model(
                Config.model_name, pretrained=True, num_classes=0
            )
        
        # classification head
        self.classification_head = nn.Sequential(
            nn.Linear(self.backbone.num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_targets),
        )

    def forward(self, img):
        features = self.backbone(img)
        outputs = self.classification_head(features)
        return outputs


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for img, targets in tqdm(loader, desc="Training"):
        img, targets = img.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(img)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for img, targets in tqdm(loader, desc="Validating"):
            img, targets = img.to(device), targets.to(device)
            outputs = model(img)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(loader)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def build_targets_from_predictions(preds):
    green = np.maximum(0, preds[:, 0])
    gdm = np.maximum(0, preds[:, 1])
    total = np.maximum(0, preds[:, 2])
    clover = np.maximum(gdm - green, 0)
    dead = np.maximum(0, total - green - clover)

    return np.stack([clover, dead, green, total, gdm], axis=1)


class WeightedMSELoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights

    def forward(self, pred, target):
        mse = (pred - target) ** 2
        weighted = mse * self.weights.to(pred.device).unsqueeze(0)
        return weighted.mean(dim=1).mean()

set_seed(Config.seed)
# Config.epochs = 1 # debug pipeline


# check if kaggle/input exists in the terminal
if os.path.exists("/kaggle/input/csiro-biomass/"):
    input_dir = "/kaggle/input/csiro-biomass/"
# else we are running locally
else:
    input_dir = "./"

df = pd.read_csv(input_dir + "train.csv")  # 357 unique training images
test_df = pd.read_csv(input_dir + "test.csv")

# Data transforms
train_tf = transforms.Compose(
    [
        transforms.Resize((Config.img_size, Config.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# validation transform: resize and normalize only
val_tf = transforms.Compose(
    [
        transforms.Resize((Config.img_size, Config.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


gkf = StratifiedGroupKFold(n_splits=Config.n_folds)
imgs = df.groupby("image_path").first().reset_index()

folds = gkf.split(X=imgs, y=imgs["State"], groups=imgs["image_path"])

for fold, (train_idx, val_idx) in enumerate(folds):
    current_seed = Config.seed + fold

    print(f"\n============= Fold: {fold+1} | Seed: {current_seed} =============")
    set_seed(current_seed)

    best_val = np.inf
    patience = 5
    patience_counter = 0

    model = BiomassModel().to(Config.device)
    # criterion = WeightedMSELoss(Config.target_weights)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.lr)
    scheduler = cosine_annealing_with_warmup(
        optimizer,
        warmup_epochs=5,
        total_epochs=Config.epochs,
        base_lr=Config.lr,
    )

    train_imgs = imgs.loc[train_idx, "image_path"].values
    val_imgs = imgs.loc[val_idx, "image_path"].values
    train_df = df[df["image_path"].isin(train_imgs)].reset_index(drop=True)
    val_df = df[df["image_path"].isin(val_imgs)].reset_index(drop=True)

    train_ds = BiomassDataset(train_df, transform=train_tf)
    val_ds = BiomassDataset(val_df, transform=val_tf)

    train_dl = DataLoader(
        train_ds,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
    )

    for epoch in range(Config.epochs):
        print(f"Epoch {epoch+1}/{Config.epochs}")
        
        # train only the regression head for the first 5 epochs
        if epoch == 0:
            for param in model.backbone.parameters():
                param.requires_grad = False
        elif epoch == 5:
            print("Unfreezing backbone for fine-tuning")
            for param in model.backbone.parameters():
                param.requires_grad = True
        
        trn_loss = train_epoch(model, train_dl, criterion, optimizer, Config.device)
        val_loss = validate(model, val_dl, criterion, Config.device)
        
        scheduler.step()
        
        print(f"Train: {trn_loss:.4f} | Val: {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), f"best_model_fold_{fold+1}.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break


# load models
models = []
for f in range(Config.n_folds):
    fold_model = BiomassModel().to(Config.device)
    state_dict = torch.load(f"best_model_fold_{f+1}.pth", map_location=Config.device)
    result = fold_model.load_state_dict(state_dict)

    if len(result.missing_keys) == 0 and len(result.unexpected_keys) == 0:
        print(f"✅ Fold {f+1}: Model weights loaded successfully.")
        models.append(fold_model)
    else:
        print(f"⚠️ Fold {f+1}: Some keys did not match!")
        if len(result.missing_keys) > 0:
            print("  Missing keys:", result.missing_keys)
        if len(result.unexpected_keys) > 0:
            print("  Unexpected keys:", result.unexpected_keys)


test_df = pd.read_csv(input_dir + "test.csv")
test_imgs = test_df["image_path"].unique()
test_ds = BiomassDataset(test_df, transform=val_tf, is_test=True)
test_dl = DataLoader(
    test_ds,
    batch_size=Config.batch_size,
    shuffle=False,
    num_workers=Config.num_workers,
)

predictions = []
image_paths = []

# predict with TTA
with torch.no_grad():
    for img, paths in tqdm(test_dl, desc="Testing"):
        img = img.to(Config.device)
        ensemble_preds = []
        
        for model in models:
            model.eval()
            
            # TTA: original + flips
            tta_preds = []
            
            # Original
            tta_preds.append(torch.expm1(model(img)))
            
            # Horizontal flip
            tta_preds.append(torch.expm1(model(torch.flip(img, dims=[3]))))
            
            # Vertical flip
            tta_preds.append(torch.expm1(model(torch.flip(img, dims=[2]))))
            
            # Both flips
            tta_preds.append(torch.expm1(model(torch.flip(img, dims=[2, 3]))))
            
            # Average TTA predictions for this model
            model_pred = torch.stack(tta_preds).mean(dim=0)
            ensemble_preds.append(model_pred.cpu().numpy())
        
        # Average across models
        predictions.append(np.mean(ensemble_preds, axis=0))
        image_paths.extend(paths)

predictions = np.vstack(predictions)

full_predictions = build_targets_from_predictions(predictions)


# Build a quick lookup map from image_path → predicted vector
pred_dict = {path: preds for path, preds in zip(image_paths, full_predictions)}
target_idx = {"Dry_Clover_g": 0, "Dry_Dead_g": 1, "Dry_Green_g": 2, "Dry_Total_g": 3, "GDM_g": 4}
submission_dict = {}

def get_predictions(row):
    preds = pred_dict.get(row["image_path"])
    if preds is not None:
        return preds[target_idx[row["target_name"]]]
    return 0.0

test_df["target"] = test_df.apply(get_predictions, axis=1)
test_df["target"] = np.clip(test_df["target"], 0, None)
submission_df = test_df[["sample_id", "target"]]
submission_df.to_csv("submission.csv", index=False)