from dataloader import get_device, get_transforms, load_dataset
from feature_extractor import load_resnet_model, extract_features
from helpers import save_features

def main(data_dir, output_dir):
    device = get_device()
    transform = get_transforms()
    dataset, dataloader = load_dataset(data_dir, transform)
    model = load_resnet_model(device)

    features, labels = extract_features(model, dataloader, device)
    print(f"Features shape: {features.shape}, Labels shape: {labels.shape}")

    save_features(features, labels, output_dir)

if __name__ == "__main__":
    data_dir = "/Users/shanoonissaka/Documents/school/thesis-project/datasets/images/train"
    main(data_dir, output_dir="./extracted_features")

    """
    To run different pretrained models, modify the import and function calls in feature_extractor.py:
    """