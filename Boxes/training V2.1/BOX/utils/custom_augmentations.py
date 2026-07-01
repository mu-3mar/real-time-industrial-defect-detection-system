import albumentations as A
import cv2


def get_conveyor_augmentations():
    """
    Build Albumentations transform pipeline for realistic industrial conveyor belt imagery.
    Subtle blur augmentations only with low probabilities.
    """
    return A.Compose([
        A.OneOf([
            A.MotionBlur(blur_limit=(5, 15), allow_shifted=True, p=0.5),  # horizontal/vertical standard motion blur
            A.GaussianBlur(blur_limit=(3, 5), sigma_limit=(0.5, 1.5), p=0.5)
        ], p=0.30)  # Apply any blur only 30% of the time
    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels'],
        min_visibility=0.5
    ))


# For standalone use
def apply_conveyor_blur_augment(image):
    """Apply blur augmentations directly to a single image."""
    transform = get_conveyor_augmentations()
    transformed = transform(image=image)
    return transformed['image']
