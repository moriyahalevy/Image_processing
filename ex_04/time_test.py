import time
import numpy as np
from q1_solution import warp_nearest_neighbor
from ex3wrap_solution import warp_image
def run_comparison():
    test_sizes = [(500, 500), (1000, 1000), (1500, 1500)]
    
    print(f"{'גודל תמונה':<15} | {'זמן לולאות (שניות)':<20} | {'זמן NumPy (שניות)':<20}")
    print("-" * 60)

    for h, w in test_sizes:
        # יצירת תמונה אקראית
        img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        
        # 1. מדידת זמן לגרסת הלולאות  
        start_loops = time.time()
        # וודאי שזה השם של הפונקציה הישנה שלך:
        _ = warp_image(img, 30, 1, 1) 
        end_loops = time.time()
        time_loops = end_loops - start_loops

        # 2. מדידת זמן לגרסת NumPy 
        start_numpy = time.time()
        _ = warp_nearest_neighbor(img, 30, 1, 1)
        end_numpy = time.time()
        time_numpy = end_numpy - start_numpy

        print(f"{h}x{w:<10} | {time_loops:<20.4f} | {time_numpy:<20.4f}")

run_comparison()