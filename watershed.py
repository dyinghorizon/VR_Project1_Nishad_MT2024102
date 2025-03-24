import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Tuple
from sklearn.model_selection import train_test_split
import json  # Import the json module
from itertools import product

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


def visualize_segmentation(original_image, mask, masked_image, method):
    """
    Visualizes the original image, the segmented mask, and the masked image.

    Args:
        original_image (np.ndarray): The original image.
        mask (np.ndarray): The segmented mask.
        masked_image (np.ndarray): The image with the mask overlaid.
        method (str): The segmentation method used.
    """

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB) if len(original_image.shape) == 3 else original_image,
               cmap='gray')
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap="gray")
    plt.title(f"Mask ({method})")
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


def load_data(data_dir):
    """Loads images and masks from the dataset directory."""
    images = []
    masks = []
    
    face_crop_path = os.path.join(data_dir, "face_crop")
    face_crop_seg_path = os.path.join(data_dir, "face_crop_segmentation")

    if os.path.exists(face_crop_path) and os.path.exists(face_crop_seg_path):
        for img_name in os.listdir(face_crop_path):
            if img_name.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(face_crop_path, img_name)
                mask_path = os.path.join(face_crop_seg_path, img_name)

                if os.path.exists(mask_path):
                    try:
                        img = cv2.imread(img_path)
                        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                        if img is None or mask is None:
                            print(f"Error: Could not load image or mask: {img_name}")
                            continue
                        images.append(img)
                        masks.append(mask)
                    except Exception as e:
                        print(f"Error processing {img_name}: {e}")
    return images, masks

def tune_watershed_parameters(images, masks, param_grid):
    """
    Tunes the watershed segmentation parameters using a grid search approach based on IoU and Dice scores.

    Args:
        images (list): List of input images.
        masks (list): List of ground truth masks.
        param_grid (dict): A dictionary defining the parameter grid to search.

    Returns:
        dict: A dictionary containing the best parameters, the corresponding best IoU score, and the best Dice score.
    """
    best_iou = 0
    best_dice = 0
    best_params = {}

    param_names = list(param_grid.keys())
    param_combinations = product(*param_grid.values())

    for params in param_combinations:
        current_params = dict(zip(param_names, params))
        print(f"Testing parameters: {current_params}")

        total_iou = 0
        total_dice = 0
        for img, gt_mask in zip(images, masks):
            # Ensure images and masks are valid
            if img is None or gt_mask is None:
                print("Skipping invalid image or mask.")
                continue
            
            mask, _ = segment_mask_region_watershed(img, **current_params)
            iou, dice = evaluate_segmentation(gt_mask, mask)
            total_iou += iou
            total_dice += dice
        
        avg_iou = total_iou / len(images) if len(images) > 0 else 0
        avg_dice = total_dice / len(images) if len(images) > 0 else 0
        print(f"Average IoU: {avg_iou:.4f}, Average Dice: {avg_dice:.4f}")

        # Using a combined metric (e.g., average of IoU and Dice) for selection
        combined_metric = (avg_iou + avg_dice) / 2

        if combined_metric > (best_iou + best_dice)/2:
            best_iou = avg_iou
            best_dice = avg_dice
            best_params = current_params

    print(f"\nBest Average IoU: {best_iou:.4f}, Best Average Dice: {best_dice:.4f}")
    print(f"Best parameters: {best_params}")
    return {"best_params": best_params, "best_iou": best_iou, "best_dice": best_dice}

def save_best_parameters(best_params, filename="best_watershed_params.txt"):
    """
    Saves the best watershed parameters to a text file in JSON format.

    Args:
        best_params (dict): The dictionary containing the best parameters.
        filename (str): The name of the file to save the parameters to.
    """
    try:
        with open(filename, "w") as f:
            json.dump(best_params, f, indent=4)
        print(f"Best parameters saved to {filename}")
    except Exception as e:
        print(f"Error saving parameters to {filename}: {e}")

# --- Example Usage ---
if __name__ == "__main__":
    data_dir = "./MSFD"  # Corrected: Now points to the root directory
    
    all_images = []
    all_masks = []
    
    for subdir in os.listdir(data_dir):
        subdir_path = os.path.join(data_dir, subdir)
        if os.path.isdir(subdir_path):
            # Now load data from each subdirectory (e.g., "1", "2")
            images, masks = load_data(subdir_path)
            all_images.extend(images)
            all_masks.extend(masks)

    if not all_images or not all_masks:
        print("No data loaded. Check your dataset directory and structure.")
        exit()
    
    # Split data into training and validation sets
    train_images, val_images, train_masks, val_masks = train_test_split(
        all_images, all_masks, test_size=0.2, random_state=42
    )

    # --- Parameter Grid ---
    param_grid = {
        "opening_kernel_size": [(3, 3), (5, 5)],
        "opening_iterations": [1, 2],
        "dilation_iterations": [2, 3],
        "distance_threshold_ratio": [0.7, 0.8],
        "morph_kernel_size": [(3,3), (5,5)],
        "morph_iterations": [1,2]
    }

    # --- Tuning ---
    print("Tuning parameters...")
    tuning_results = tune_watershed_parameters(train_images, train_masks, param_grid)

    # --- Save the best parameters to a file ---
    best_params = tuning_results["best_params"]
    save_best_parameters(best_params)

    # --- Watershed with best parameters on validation set---
    
    total_iou_val = 0
    total_dice_val = 0
    for img, gt_mask in zip(val_images, val_masks):
        # Ensure images and masks are valid
        if img is None or gt_mask is None:
            print("Skipping invalid image or mask.")
            continue
        mask_watershed, masked_image_watershed = segment_mask_region_watershed(img, **best_params)
        visualize_segmentation(img, mask_watershed, masked_image_watershed, "Watershed (Tuned)")
        iou_watershed, dice_watershed = evaluate_segmentation(gt_mask, mask_watershed)
        total_iou_val += iou_watershed
        total_dice_val += dice_watershed
    avg_iou_val = total_iou_val / len(val_images) if len(val_images) > 0 else 0
    avg_dice_val = total_dice_val / len(val_images) if len(val_images) > 0 else 0
    print(f"Average IoU (Watershed Tuned on validation set): {avg_iou_val:.4f}")
    print(f"Average Dice (Watershed Tuned on validation set): {avg_dice_val:.4f}")
