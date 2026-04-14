import numpy as np


def warp_bilinear(image: np.ndarray, angle_deg: float, scale_x: float, scale_y: float) -> np.ndarray:
    H, W, C = image.shape
    cx, cy = W / 2.0, H / 2.0

    theta = np.radians(angle_deg)
    M_inv = np.linalg.inv(np.array([[1,0,cx],[0,1,cy],[0,0,1]]) @ 
                          np.array([[np.cos(theta)*scale_x, -np.sin(theta)*scale_x, 0],
                                    [np.sin(theta)*scale_y,  np.cos(theta)*scale_y, 0],
                                    [0,0,1]]) @ 
                          np.array([[1,0,-cx],[0,1,-cy],[0,0,1]]))

    ys, xs = np.indices((H, W))
    coords = np.stack([xs.ravel(), ys.ravel(), np.ones(H*W)])
    src_coords = M_inv @ coords
    
    src_x = src_coords[0].reshape(H, W)
    src_y = src_coords[1].reshape(H, W)

    # מציאת 4 הפיקסלים השכנים
    x0 = np.floor(src_x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(src_y).astype(int)
    y1 = y0 + 1

    # מסיכה לבדיקת גבולות
    mask = (x0 >= 0) & (x1 < W) & (y0 >= 0) & (y1 < H)

    # חישוב משקלים
    wa = (x1 - src_x) * (y1 - src_y)
    wb = (src_x - x0) * (y1 - src_y)
    wc = (x1 - src_x) * (src_y - y0)
    wd = (src_x - x0) * (src_y - y0)

    output = np.zeros_like(image)
    
    # חישוב וקטורי מלא לכל הערוצים יחד
    output[mask] = (wa[mask, None] * image[y0[mask], x0[mask]] +
                    wb[mask, None] * image[y0[mask], x1[mask]] +
                    wc[mask, None] * image[y1[mask], x0[mask]] +
                    wd[mask, None] * image[y1[mask], x1[mask]])

    return output