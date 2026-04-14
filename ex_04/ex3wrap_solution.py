import numpy as np

def warp_image(image: np.ndarray,
               angle_deg: float,
               scale_x: float,
               scale_y: float) -> np.ndarray:

    H, W, C = image.shape
    cx, cy = W / 2.0, H / 2.0


    theta = np.radians(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    T1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])
    RS = np.array([[cos_t * scale_x, -sin_t * scale_x, 0],
                   [sin_t * scale_y,  cos_t * scale_y, 0],
                   [0,                0,               1]])
    T2 = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]])

    M = T2 @ RS @ T1
    M_inv = np.linalg.inv(M)


    ys, xs = np.indices((H, W))

    coords = np.stack([xs.ravel() + 0.5, ys.ravel() + 0.5, np.ones(H*W)])
    

    src_coords = M_inv @ coords
    src_x = (src_coords[0] - 0.5).reshape(H, W)
    src_y = (src_coords[1] - 0.5).reshape(H, W)

    x0 = np.floor(src_x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(src_y).astype(int)
    y1 = y0 + 1

    mask = (x0 >= 0) & (x1 < W) & (y0 >= 0) & (y1 < H)

    output = np.zeros_like(image)
    

    a = src_x - x0
    b = src_y - y0

    for c in range(C):
        img_c = image[:, :, c]
        output[mask, c] = (
            (1 - a[mask]) * (1 - b[mask]) * img_c[y0[mask], x0[mask]] +
            a[mask] * (1 - b[mask]) * img_c[y0[mask], x1[mask]] +
            (1 - a[mask]) * b[mask] * img_c[y1[mask], x0[mask]] +
            a[mask] * b[mask] * img_c[y1[mask], x1[mask]]
        )

    return output