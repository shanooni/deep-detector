import torch
import torch.nn as nn
# from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet101, ResNet101_Weights
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os

def load_resnet_model(device):
    """Load a pretrained ResNet and remove final classification layer."""
    weights = ResNet101_Weights.IMAGENET1K_V1
    resnet = resnet101(weights=weights, progress=False)
    model = nn.Sequential(*list(resnet.children())[:-1])  # remove FC layer
    model = model.to(device)
    model.eval()
    return model

def extract_features(model, dataloader, device):
    """Extract deep features from images using pretrained model."""
    features_list, labels_list = [], []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting Features"):
            images = images.to(device)
            outputs = model(images)
            outputs = outputs.view(outputs.size(0), -1)  # flatten from (batch, 2048, 1, 1)
            features_list.append(outputs.cpu().numpy())
            labels_list.append(labels.cpu().numpy())

    features = np.concatenate(features_list)
    labels = np.concatenate(labels_list)

    return features, labels