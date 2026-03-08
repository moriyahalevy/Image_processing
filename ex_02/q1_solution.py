import numpy as np
import matplotlib.pyplot as plt

def translation_matrix(a, b):
    return np.array([
        [1, 0, a],
        [0, 1, b],
        [0, 0, 1]
    ])

def rotation_matrix(theta_degrees):
    theta = np.radians(theta_degrees)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])

def scale_matrix(sx, sy=None):
    if sy is None:
        sy = sx
    return np.array([
        [sx, 0,  0],
        [0,  sy, 0],
        [0,  0,  1]
    ])

T = translation_matrix(100, 200)
R = rotation_matrix(30)
T_inv = translation_matrix(-100, -200)

combined_matrix = T @ R @ T_inv
print(combined_matrix)