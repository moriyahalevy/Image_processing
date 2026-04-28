import numpy as np
from scipy import signal
from numpy.lib.stride_tricks import sliding_window_view

def initialize_kernel():
    kernel = np.array([
        [-1,  2,  1],
        [-2,  1, -3],
        [ 3,  0, -1]
    ], dtype=np.float32)
    return kernel

def image_get():
    image = np.array([
        [103, 102, 101, 100],
        [104, 103, 102, 101],
        [ 53,  52,  51,  50],
        [ 45,  53,  52,  51]
    ], dtype=np.uint8)
    return image

def loop_correlate_cross(image, kernel):
    img_h, img_w = image.shape
    k_h, k_w = kernel.shape
    
    res_h = img_h - k_h + 1
    res_w = img_w - k_w + 1
    
    result = np.zeros((res_h, res_w), dtype=np.float32)
    
    for i in range(res_h):
        for j in range(res_w):
            patch = image[i:i+k_h, j:j+k_w]
            result[i, j] = np.sum(patch * kernel)
            
    return result

def np_correlate_cross(image, kernel):
    windows = sliding_window_view(image, kernel.shape)
    result = np.sum(windows * kernel, axis=(2, 3))
    return result.astype(np.float32)

def cross_correlate_scipy(image, kernel):
    result = signal.correlate2d(image, kernel, mode='valid')
    return result.astype(np.float32)

def correlations_cross_compare():
    img = image_get()
    ker = initialize_kernel()
    
    res_loop = loop_correlate_cross(img, ker)
    res_np = np_correlate_cross(img, ker)
    res_scipy = cross_correlate_scipy(img, ker)
    
    # בדיקה שכל התוצאות זהות (allclose מתגברת על שגיאות עיגול קטנות)
    check_1 = np.allclose(res_loop, res_np)
    check_2 = np.allclose(res_np, res_scipy)
    
    return check_1 and check_2

if __name__ == "__main__":
    if correlations_cross_compare():
        print("Success! All methods produced the same result.")
        img = image_get()
        ker = initialize_kernel()
        print("The Result Matrix (2x2):")
        print(loop_correlate_cross(img, ker))
    else:
        print("Failure: Results do not match.")