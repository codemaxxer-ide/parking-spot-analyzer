"""Training-only realistic augmentation; deterministic validation/test transforms."""

from torchvision import transforms
from torchvision.transforms import InterpolationMode

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def build_transforms(*, training: bool, image_size: int = 128):
    if image_size < 32:
        raise ValueError("Image size must be >=32.")
    operations = [transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR)]
    if training:
        operations += [
            transforms.RandomResizedCrop(image_size, scale=(0.90, 1.0), ratio=(0.95, 1.05)),
            transforms.RandomRotation(7, interpolation=InterpolationMode.BILINEAR),
            transforms.ColorJitter(brightness=0.25, contrast=0.20, saturation=0.12, hue=0.015),
            transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 0.8))], p=0.15),
        ]
    operations += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(operations)
