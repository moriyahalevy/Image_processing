import numpy as np

def deg_to_rad(degrees):
    return degrees * (np.pi / 180)

angles_deg = [1, 5, 10, 30, 45, 180, 90, 0]
print("degrees, radians, sin, cos") # כותרת ה-CSV

for d in angles_deg:
    r = deg_to_rad(d)
    s = np.sin(r)
    c = np.cos(r)
    print(f"{d}, {r:.4f}, {s:.4f}, {c:.4f}")