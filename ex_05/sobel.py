import cv2
import numpy as np
import sys
import os

def run_sobel(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not read image.")
        return

    # א. המרה לגווני אפור
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    gx_abs = np.abs(gx)
    gy_abs = np.abs(gy)
    
    gx_norm = cv2.normalize(gx_abs, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    gy_norm = cv2.normalize(gy_abs, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    magnitude = np.sqrt(gx**2 + gy**2)
    mag_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    base_name = os.path.splitext(image_path)[0]
    cv2.imwrite(f"{base_name}_grayscale.jpg", gray)
    cv2.imwrite(f"{base_name}_gx.jpg", gx_norm)
    cv2.imwrite(f"{base_name}_gy.jpg", gy_norm)
    cv2.imwrite(f"{base_name}_magnitude.jpg", mag_norm)
    print("All 4 images have been saved.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_sobel(sys.argv[1])
    else:
        print("Usage: python sobel.py <image_path>")