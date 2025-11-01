import torch
import torch.nn as nn
# EfficientNet imports
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from torchvision.models import efficientnet_b5, EfficientNet_B5_Weights
from torchvision.models import efficientnet_b6, EfficientNet_B6_Weights
from torchvision.models import efficientnet_b7, EfficientNet_B7_Weights

# DenseNet imports
from torchvision.models import densenet121, DenseNet121_Weights
from torchvision.models import densenet169, DenseNet169_Weights
from torchvision.models import densenet201, DenseNet201_Weights
from torchvision.models import densenet161, DenseNet161_Weights

# ResNet imports
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet34, ResNet34_Weights
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet101, ResNet101_Weights
from torchvision.models import resnet152, ResNet152_Weights

# Inception imports
from torchvision.models import inception_v3, Inception_V3_Weights

# MobileNet imports
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

# ShuffleNet imports
from torchvision.models import shufflenet_v2_x0_5, ShuffleNet_V2_X0_5_Weights
from torchvision.models import shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights
from torchvision.models import shufflenet_v2_x1_5, ShuffleNet_V2_X1_5_Weights
from torchvision.models import shufflenet_v2_x2_0, ShuffleNet_V2_X2_0_Weights

# SqueezeNet imports
from torchvision.models import squeezenet1_0, SqueezeNet1_0_Weights
from torchvision.models import squeezenet1_1, SqueezeNet1_1_Weights

# Vision Transformer imports
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision.models import vit_b_32, ViT_B_32_Weights
from torchvision.models import vit_l_16, ViT_L_16_Weights
from torchvision.models import vit_l_32, ViT_L_32_Weights
from torchvision.models import vit_h_14, ViT_H_14_Weights

# ConvNeXt imports
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torchvision.models import convnext_small, ConvNeXt_Small_Weights
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
from torchvision.models import convnext_large, ConvNeXt_Large_Weights   




# load pretrained model functions and feature extraction remain unchanged
def load_model(device, model_name="resnet18"):
    """Load a specified pretrained model and remove final classification layer."""
    if model_name == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V2
        model = resnet18(weights=weights, progress=False)
    elif model_name == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        model = efficientnet_b0(weights=weights, progress=False)
    elif model_name == "densenet121":
        weights = DenseNet121_Weights.IMAGENET1K_V1
        model = densenet121(weights=weights, progress=False)
    elif model_name == "inception_v3":
        weights = Inception_V3_Weights.IMAGENET1K_V1
        model = inception_v3(weights=weights, progress=False)
    elif model_name == "mobilenet_v2":
        weights = MobileNet_V2_Weights.IMAGENET1K_V1
        model = mobilenet_v2(weights=weights, progress=False)
    elif model_name == "mobilenet_v3_large":
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1
        model = mobilenet_v3_large(weights=weights, progress=False)
    elif model_name == "mobilenet_v3_small":
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
        model = mobilenet_v3_small(weights=weights, progress=False)
    elif model_name == "shufflenet_v2_x1_0":
        weights = ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1
        model = shufflenet_v2_x1_0(weights=weights, progress=False)
    elif model_name == "squeezenet1_0":
        weights = SqueezeNet1_0_Weights.IMAGENET1K_V1
        model = squeezenet1_0(weights=weights, progress=False)
    elif model_name == "squeezenet1_1":
        weights = SqueezeNet1_1_Weights.IMAGENET1K_V1
        model = squeezenet1_1(weights=weights, progress=False)
    elif model_name == "vit_b_16":
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        model = vit_b_16(weights=weights, progress=False)
    elif model_name == "vit_b_32":
        weights = ViT_B_32_Weights.IMAGENET1K_V1
        model = vit_b_32(weights=weights, progress=False)
    elif model_name == "vit_l_16":
        weights = ViT_L_16_Weights.IMAGENET1K_V1
        model = vit_l_16(weights=weights, progress=False)
    elif model_name == "vit_l_32":
        weights = ViT_L_32_Weights.IMAGENET1K_V1
        model = vit_l_32(weights=weights, progress=False)
    elif model_name == "vit_h_14":
        weights = ViT_H_14_Weights.IMAGENET1K_V1
        model = vit_h_14(weights=weights, progress=False)
    elif model_name == "convnext_tiny":
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        model = convnext_tiny(weights=weights, progress=False)
    elif model_name == "convnext_small":
        weights = ConvNeXt_Small_Weights.IMAGENET1K_V1
        model = convnext_small(weights=weights, progress=False)
    elif model_name == "convnext_base":
        weights = ConvNeXt_Base_Weights.IMAGENET1K_V1
        model = convnext_base(weights=weights, progress=False)
    elif model_name == "convnext_large":
        weights = ConvNeXt_Large_Weights.IMAGENET1K_V1
        model = convnext_large(weights=weights, progress=False)
    elif model_name == "efficientnet_b1":
        weights = EfficientNet_B1_Weights.IMAGENET1K_V1
        model = efficientnet_b1(weights=weights, progress=False)
    elif model_name == "efficientnet_b2":
        weights = EfficientNet_B2_Weights.IMAGENET1K_V1
        model = efficientnet_b2(weights=weights, progress=False)
    elif model_name == "efficientnet_b3":
        weights = EfficientNet_B3_Weights.IMAGENET1K_V1
        model = efficientnet_b3(weights=weights, progress=False)
    elif model_name == "efficientnet_b4":
        weights = EfficientNet_B4_Weights.IMAGENET1K_V1
        model = efficientnet_b4(weights=weights, progress=False)
    elif model_name == "efficientnet_b5":
        weights = EfficientNet_B5_Weights.IMAGENET1K_V1
        model = efficientnet_b5(weights=weights, progress=False)
    elif model_name == "efficientnet_b6":
        weights = EfficientNet_B6_Weights.IMAGENET1K_V1
        model = efficientnet_b6(weights=weights, progress=False)
    elif model_name == "efficientnet_b7":
        weights = EfficientNet_B7_Weights.IMAGENET1K_V1
        model = efficientnet_b7(weights=weights, progress=False)
    elif model_name == "efficientnet_b0_v2":
        weights = EfficientNet_B0_Weights.IMAGENET1K_V2
        model = efficientnet_b0(weights=weights, progress=False)
    elif model_name == "efficientnet_b1_v2":
        weights = EfficientNet_B1_Weights.IMAGENET1K_V2
        model = efficientnet_b1(weights=weights, progress=False)
    elif model_name == "efficientnet_b2_v2":
        weights = EfficientNet_B2_Weights.IMAGENET1K_V2
        model = efficientnet_b2(weights=weights, progress=False)
    elif model_name == "efficientnet_b3_v2":
        weights = EfficientNet_B3_Weights.IMAGENET1K_V2
        model = efficientnet_b3(weights=weights, progress=False)
    elif model_name == "efficientnet_b4_v2":
        weights = EfficientNet_B4_Weights.IMAGENET1K_V2
        model = efficientnet_b4(weights=weights, progress=False)
    elif model_name == "efficientnet_b5_v2":
        weights = EfficientNet_B5_Weights.IMAGENET1K_V2
        model = efficientnet_b5(weights=weights, progress=False)
    elif model_name == "efficientnet_b6_v2":
        weights = EfficientNet_B6_Weights.IMAGENET1K_V2
        model = efficientnet_b6(weights=weights, progress=False)
    elif model_name == "efficientnet_b7_v2":
        weights = EfficientNet_B7_Weights.IMAGENET1K_V2
        model = efficientnet_b7(weights=weights, progress=False)
    elif model_name == "densenet169":
        weights = DenseNet169_Weights.IMAGENET1K_V1
        model = densenet169(weights=weights, progress=False)
    elif model_name == "densenet201":
        weights = DenseNet201_Weights.IMAGENET1K_V1
        model = densenet201(weights=weights, progress=False)
    elif model_name == "densenet161":
        weights = DenseNet161_Weights.IMAGENET1K_V1
        model = densenet161(weights=weights, progress=False)
    elif model_name == "resnet34":
        weights = ResNet34_Weights.IMAGENET1K_V2
        model = resnet34(weights=weights, progress=False)
    elif model_name == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V2
        model = resnet50(weights=weights, progress=False)
    elif model_name == "resnet101":
        weights = ResNet101_Weights.IMAGENET1K_V2
        model = resnet101(weights=weights, progress=False)
    elif model_name == "resnet152":
        weights = ResNet152_Weights.IMAGENET1K_V2
        model = resnet152(weights=weights, progress=False)
    elif model_name == "inception_v3":
        weights = Inception_V3_Weights.IMAGENET1K_V1
        model = inception_v3(weights=weights, progress=False)
    elif model_name == "mobilenet_v2":
        weights = MobileNet_V2_Weights.IMAGENET1K_V1
        model = mobilenet_v2(weights=weights, progress=False)
    elif model_name == "mobilenet_v3_large":
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1
        model = mobilenet_v3_large(weights=weights, progress=False)
    elif model_name == "mobilenet_v3_small":
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
        model = mobilenet_v3_small(weights=weights, progress=False)
    elif model_name == "shufflenet_v2_x0_5":
        weights = ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1
        model = shufflenet_v2_x0_5(weights=weights, progress=False)
    elif model_name == "shufflenet_v2_x1_0":
        weights = ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1
        model = shufflenet_v2_x1_0(weights=weights, progress=False)
    elif model_name == "shufflenet_v2_x1_5":
        weights = ShuffleNet_V2_X1_5_Weights.IMAGENET1K_V1
        model = shufflenet_v2_x1_5(weights=weights, progress=False)
    elif model_name == "shufflenet_v2_x2_0":
        weights = ShuffleNet_V2_X2_0_Weights.IMAGENET1K_V1
        model = shufflenet_v2_x2_0(weights=weights, progress=False)
    else:
        raise ValueError(f"Model {model_name} not supported.")
    
    model = nn.Sequential(*list(model.children())[:-1])  # remove FC layer
    model = model.to(device)
    model.eval()
    return model