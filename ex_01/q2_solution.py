import numpy as np
import matplotlib.pyplot as plt


def deg_to_rad(deg):
    return deg * (np.pi / 180)

theta = deg_to_rad(30)
r_30 = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta),  np.cos(theta)]
])
print("r_30 Matrix:\n", r_30)


sx_2 = np.array([
    [2, 0],
    [0, 1]
])
print("\nsx_2 Matrix:\n", sx_2)


rs = r_30 @ sx_2
sr = sx_2 @ r_30
print("\nrs (Rotation @ Scale):\n", rs)
print("\nsr (Scale @ Rotation):\n", sr)


rect = np.array([
    [-1, -0.5],
    [1, -0.5],
    [1, 0.5],
    [-1, 0.5],
    [-1, -0.5]
]).T

def apply_transform(matrix, points):
    return matrix @ points

rect_r30 = apply_transform(r_30, rect)
rect_sx2 = apply_transform(sx_2, rect)
rect_rs  = apply_transform(rs, rect)
rect_sr  = apply_transform(sr, rect)


plt.figure(figsize=(10, 8))

plt.plot(rect[0, :], rect[1, :], label='Original', linewidth=2)
plt.plot(rect_r30[0, :], rect_r30[1, :], label='Rotated 30°')
plt.plot(rect_sx2[0, :], rect_sx2[1, :], label='Scale X by 2')
plt.plot(rect_rs[0, :], rect_rs[1, :], label='RS (Scale then Rotate)', linestyle='--')
plt.plot(rect_sr[0, :], rect_sr[1, :], label='SR (Rotate then Scale)', linestyle=':')

plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.axis('equal')
plt.title('Transformations: Rotation vs Scaling')
plt.show()