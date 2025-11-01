import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_device():
    """Return GPU if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
def get_transforms():
    """Define preprocessing transformations for images."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform


def load_dataset(data_dir, transform, batch_size=32, num_workers=4):
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return dataset, dataloader