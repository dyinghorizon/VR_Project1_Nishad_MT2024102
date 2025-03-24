import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from typing import Tuple
import random  # Import the random module

def load_best_parameters(filename="best_watershed_params.txt"):
    """
    Loads the best watershed parameters from a JSON file.

    Args:
        filename (str): The name of the file to load the parameters from.

    Returns:
        dict: The dictionary containing the best parameters, or None if the file
              does not exist or an error occurs.
    """
    try:
        with open(filename, "r") as f:
            best_params = json.load(f)
        print(f"Best parameters loaded from {filename}")
        return best_params
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {filename}.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading parameters from {filename}: {e}")
        return None


def segment_mask_region_watershed(image, opening_kernel_size: Tuple[int, int] = (5, 5), opening_iterations: int = 2,
                                  dilation_iterations: int = 3, distance_threshold_ratio: float = 0.8,
                                  morph_kernel_size: Tuple[int, int] = (5, 5), morph_iterations: int = 1):
    """
    Segments the mask region in an image using the watershed algorithm with tunable parameters.

    Args:
        image (np.ndarray): The input image (color).
        opening_kernel_size (Tuple[int, int]): Kernel size for morphological opening.
        opening_iterations (int): Number of iterations for morphological opening.
        dilation_iterations (int): Number of iterations for dilation.
        distance_threshold_ratio (float): Ratio for thresholding the distance transform.
        morph_kernel_size (Tuple[int, int]): Kernel size for morphological operations on the final mask.
        morph_iterations (int): Number of iterations for morphological operations on the final mask.
    Returns:
        np.ndarray: The segmented mask region (binary image).
        np.ndarray: The image with the segmented mask region overlaid (for visualization).
    """

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Thresholding to create a binary image
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Noise removal using morphological opening
    kernel_opening = np.ones(opening_kernel_size, np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_opening, iterations=opening_iterations)

    # Sure background area
    sure_bg = cv2.dilate(opening, kernel_opening, iterations=dilation_iterations)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, distance_threshold_ratio * dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    # Apply watershed
    markers = cv2.watershed(image, markers)

    # Create the mask
    mask = np.zeros_like(gray)
    mask[markers == -1] = 255
    mask[markers == 1] = 0
    mask[markers > 1] = 255

    # Morphological operations to clean up the mask (optional)
    kernel_morph = np.ones(morph_kernel_size, np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_morph, iterations=morph_iterations)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_morph, iterations=morph_iterations)

    # Overlay the mask on the original image for visualization
    masked_image = image.copy()
    masked_image[mask == 255] = [0, 255, 0]  # Green overlay

    return mask, masked_image


def visualize_segmentation(original_image, mask, masked_image, method, iou_score=None, dice_score=None):
    """
    Visualizes the original image, the segmented mask, and the masked image.

    Args:
        original_image (np.ndarray): The original image.
        mask (np.ndarray): The segmented mask.
        masked_image (np.ndarray): The image with the mask overlaid.
        method (str): The segmentation method used.
        iou_score (float, optional): The IoU score. Defaults to None.
        dice_score (float, optional): The Dice score. Defaults to None.
    """

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB) if len(original_image.shape) == 3 else original_image,
               cmap='gray')
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap="gray")
    title_mask = f"Mask ({method})"
    if iou_score is not None and dice_score is not None:
        title_mask += f"\nIoU: {iou_score:.4f}, Dice: {dice_score:.4f}"
    plt.title(title_mask)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB) if len(masked_image.shape) == 3 else masked_image,
               cmap='gray')
    plt.title("Masked Image")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def evaluate_segmentation(ground_truth_mask, predicted_mask):
    """
    Evaluates the segmentation results using Intersection over Union (IoU) and Dice coefficient.

    Args:
        ground_truth_mask (np.ndarray): The ground truth mask (binary).
        predicted_mask (np.ndarray): The predicted mask (binary).

    Returns:
        tuple: A tuple containing the IoU score and the Dice coefficient.
    """
    intersection = np.logical_and(ground_truth_mask, predicted_mask)
    union = np.logical_or(ground_truth_mask, predicted_mask)
    iou_score = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0

    dice_score = (2 * np.sum(intersection)) / (np.sum(ground_truth_mask) + np.sum(predicted_mask)) if (np.sum(ground_truth_mask) + np.sum(predicted_mask)) > 0 else 0
    return iou_score, dice_score


def load_data(data_dir, num_samples=10):
    """
    Loads a specified number of random images and masks from the dataset directory.

    Args:
        data_dir (str): The path to the root directory of the dataset.
        num_samples (int): The number of image-mask pairs to load.

    Returns:
        list: A list of tuples, where each tuple contains an image (np.ndarray) and its corresponding mask (np.ndarray).
              Returns an empty list if no data is found or an error occurs.
    """
    images_and_masks = []
    face_crop_path = os.path.join(data_dir, "face_crop")
    face_crop_seg_path = os.path.join(data_dir, "face_crop_segmentation")

    if not os.path.exists(face_crop_path) or not os.path.exists(face_crop_seg_path):
        print(f"Error: 'face_crop' or 'face_crop_segmentation' directory not found in {data_dir}")
        return []

    image_names = [
        img_name
        for img_name in os.listdir(face_crop_path)
        if img_name.endswith((".jpg", ".jpeg", ".png"))
    ]

    if not image_names:
        print(f"Error: No images found in {face_crop_path}")
        return []

    # Select random image names
    selected_image_names = random.sample(image_names, min(num_samples, len(image_names)))

    for img_name in selected_image_names:
        img_path = os.path.join(face_crop_path, img_name)
        mask_path = os.path.join(face_crop_seg_path, img_name)

        if os.path.exists(mask_path):
            try:
                img = cv2.imread(img_path)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if img is None or mask is None:
                    print(f"Error: Could not load image or mask: {img_name}")
                    continue
                images_and_masks.append((img, mask))
            except Exception as e:
                print(f"Error processing {img_name}: {e}")

    return images_and_masks


# --- Example Usage ---
if __name__ == "__main__":
    # Load the best parameters from the file
    best_params = load_best_parameters()

    if best_params is None:
        print("Could not load best parameters. Exiting.")
        exit()

    data_dir = "./MSFD"  # Corrected: Now points to the root directory
    
    all_images_and_masks = []
    
    for subdir in os.listdir(data_dir):
        subdir_path = os.path.join(data_dir, subdir)
        if os.path.isdir(subdir_path):
            # Now load data from each subdirectory (e.g., "1", "2")
            images_and_masks = load_data(subdir_path, num_samples=10)
            all_images_and_masks.extend(images_and_masks)
            break # we only need 10 images, so we break after the first subdir

    if not all_images_and_masks:
        print("No data loaded. Check your dataset directory and structure.")
        exit()
    
    # --- Watershed with best parameters on the loaded images---
    for img, gt_mask in all_images_and_masks:
        # Ensure images and masks are valid
        if img is None or gt_mask is None:
            print("Skipping invalid image or mask.")
            continue
        mask_watershed, masked_image_watershed = segment_mask_region_watershed(img, **best_params)
        iou_watershed, dice_watershed = evaluate_segmentation(gt_mask, mask_watershed)
        visualize_segmentation(img, mask_watershed, masked_image_watershed, "Watershed (Best Params)", iou_watershed, dice_watershed)
