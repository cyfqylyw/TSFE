import torchvision


def get_network(name, pretrained=False):
    if not pretrained:
        weights = None

    network = {
        "VGG16": torchvision.models.vgg16(weights=weights),
        "VGG16_bn": torchvision.models.vgg16_bn(weights=weights),
        "resnet18": torchvision.models.resnet18(weights=weights),
        "resnet34": torchvision.models.resnet34(weights=weights),
        "resnet50": torchvision.models.resnet50(weights=weights),
        "resnet101": torchvision.models.resnet101(weights=weights),
        "resnet152": torchvision.models.resnet152(weights=weights),
    }
    if name not in network.keys():
        raise KeyError(f"{name} is not a valid network architecture")
    return network[name]
