import cv2
import numpy as np
import os

def expand_black_region(input_path, output_path, pixels):
    # Load the image
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not read image at {input_path}")
        return

    # To expand the black region (0), we erode the white region (255)
    # A kernel of size (2*pixels + 1) will erode the boundary by 'pixels'
    kernel_size = 2 * pixels + 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # cv2.erode shrinks the white (255) areas, effectively expanding the black (0) areas
    expanded_img = cv2.erode(img, kernel, iterations=1)
    
    # Save the result
    cv2.imwrite(output_path, expanded_img)
    print(f"Saved expanded mask to {output_path}")

if __name__ == "__main__":
    input_file = "test1/expanded_black_1px.png"
    output_file = "test1/expanded_black_30px.png"
    expand_black_region(input_file, output_file, 30)
