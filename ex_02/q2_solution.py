import numpy as np
import matplotlib.pyplot as plt
from q1_solution import  rotation_matrix, scale_matrix

rect = np.array([
    [-1, -0.5, 1],
    [ 1, -0.5, 1],
    [ 1,  0.5, 1],
    [-1,  0.5, 1],
    [-1, -0.5, 1]
]).T


rect_b = rotation_matrix(30) @ rect


rect_c = scale_matrix(2, 1) @ rotation_matrix(45) @ rect


rect_d = rotation_matrix(45) @ scale_matrix(2, 1) @ rect

plt.figure(figsize=(10, 8))

plt.plot(rect[0, :], rect[1, :], 'b-', label='Original ')
plt.plot(rect_b[0, :], rect_b[1, :], 'r--', label='Rotate 30 ')
plt.plot(rect_c[0, :], rect_c[1, :], 'g-.', label='Rot 45 -> Scale ')
plt.plot(rect_d[0, :], rect_d[1, :], 'm:', label='Scale -> Rot 45 ')
# עיצוב הגרף
plt.axhline(0, color='black', lw=1)
plt.axvline(0, color='black', lw=1)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.axis('equal') 
plt.title('Question 2')


plt.show()