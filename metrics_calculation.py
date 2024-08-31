import os
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def calculate_metrics_ssim_psnr(output_images_path, ground_truth_images_path):
    ssim_measures = []
    psnr_measures = []
    
    # Get the list of image filenames
    output_files = sorted(os.listdir(output_images_path))
    ground_truth_files = sorted(os.listdir(ground_truth_images_path))

    for output_file, ground_truth_file in zip(output_files, ground_truth_files):
        output_image_path = os.path.join(output_images_path, output_file)
        ground_truth_image_path = os.path.join(ground_truth_images_path, ground_truth_file)
        
        # Load images
        output_image = cv2.imread(output_image_path)
        ground_truth_image = cv2.imread(ground_truth_image_path)
        
        # Convert images to grayscale if they are RGB
        if output_image.shape[-1] == 3:
            output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)
            ground_truth_image = cv2.cvtColor(ground_truth_image, cv2.COLOR_BGR2GRAY)
        
        # Ensure image dimensions are appropriate for SSIM calculation
        min_dim = min(output_image.shape[:2])
        win_size = min_dim if min_dim > 7 else 7
        
        # Compute SSIM and PSNR
        try:
            error_ssim, _ = ssim(output_image, ground_truth_image, full=True, win_size=win_size)
            error_psnr = psnr(output_image, ground_truth_image)
            
            ssim_measures.append(error_ssim)
            psnr_measures.append(error_psnr)
        except ValueError as e:
            print(f"Error computing metrics for {output_file}: {e}")
    
    return ssim_measures, psnr_measures

def calculate_UIQM(output_images_path):
    # Placeholder for UIQM calculation function
    return np.zeros(len(os.listdir(output_images_path)))
