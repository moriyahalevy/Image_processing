def nearest_neighbor(alpha, beta, I00, I01, I10, I11):
    if alpha < 0.5:
        if beta < 0.5:
            return I00  
        else:
            return I10  
    else:
        if beta < 0.5:
            return I01 
        else:
            return I11  

def bilinear_interpolation(alpha, beta, I00, I01, I10, I11):
    top_weighted = (1 - alpha) * I00 + alpha * I01
    bottom_weighted = (1 - alpha) * I10 + alpha * I11
    

    final_value = (1 - beta) * top_weighted + beta * bottom_weighted
    
    return final_value


val_nn = nearest_neighbor(0.2, 0.8, 0, 255, 255, 0)
val_bi = bilinear_interpolation(0.2, 0.8, 0, 255, 255, 0)

print(f"Nearest Neighbor Result: {val_nn}")
print(f"Bilinear Result: {val_bi}")