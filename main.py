from dataloader import get_device, get_transforms, load_dataset, load_dataset_with_limits
from feature_extractor import load_resnet_model, extract_features, load_pretrained_model
from helpers import save_features

def main(data_dir, output_dir):
    device = get_device()
    transform = get_transforms()
    # dataset, dataloader = load_dataset(data_dir, transform)
    dataset, dataloader = load_dataset_with_limits(data_dir, transform, max_samples=500, shuffle=True)
    model, model_name = load_pretrained_model(device=device, model_name="efficientnet_b5")

    features, labels = extract_features(model, dataloader, device)
    print(f"Features shape: {features.shape}, Labels shape: {labels.shape}")

    save_features(features, labels, output_dir, prefix=model_name)

if __name__ == "__main__":
    data_dir = "/Users/shanoonissaka/Documents/school/thesis-project/datasets/images/train"
    main(data_dir, output_dir="./features")

    """
    To run different pretrained models, modify the model_name parameter in the main function call.
    Available model names depend on the implementations in pretrainer.py:
    """