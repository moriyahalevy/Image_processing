import numpy as np

def warp_nearest_neighbor(image: np.ndarray, angle_deg: float, scale_x: float, scale_y: float) -> np.ndarray:
    H, W, C = image.shape
    cx, cy = W / 2.0, H / 2.0

    theta = np.radians(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    T1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])
    RS = np.array([[cos_t * scale_x, -sin_t * scale_x, 0],
                   [sin_t * scale_y,  cos_t * scale_y, 0],
                   [0,                0,                1]])
    T2 = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]])

    M_inv = np.linalg.inv(T2 @ RS @ T1)

    ys, xs = np.indices((H, W))
    coords = np.stack([xs.ravel(), ys.ravel(), np.ones(H*W)])
    
    src_coords = M_inv @ coords
    src_x = np.round(src_coords[0]).astype(int).reshape(H, W)
    src_y = np.round(src_coords[1]).astype(int).reshape(H, W)

    mask = (src_x >= 0) & (src_x < W) & (src_y >= 0) & (src_y < H)

    output = np.zeros_like(image)
    # השמת הערכים ללא לולאות בכלל
    output[mask] = image[src_y[mask], src_x[mask]]

    return output

