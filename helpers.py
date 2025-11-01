import os
import numpy as np

def save_features(features, labels, output_dir, prefix="resnet18_v2"):
    """Save extracted features and labels as .npy files."""
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, f"{prefix}_features.npy"), features)
    np.save(os.path.join(output_dir, f"{prefix}_labels.npy"), labels)
    print(f"✅ Saved features to {output_dir}/{prefix}_features.npy")
    print(f"✅ Saved labels to {output_dir}/{prefix}_labels.npy")