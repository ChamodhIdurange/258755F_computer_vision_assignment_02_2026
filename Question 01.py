import numpy as np

# Load the dataset
D = np.genfromtxt("lines.csv", delimiter=",", skip_header=1)
X_cols = D[:, :3]
Y_cols = D[:, 3:]

def total_least_squares(x, y):
    """Computes Total Least Squares fitting returning line parameters ax + by + c = 0"""
    x_mean, y_mean = np.mean(x), np.mean(y)
    M = np.vstack((x - x_mean, y - y_mean)).T
    
    _, _, Vt = np.linalg.svd(M)
    
    a, b = Vt[-1, :]
    c = -(a * x_mean + b * y_mean)
    
    # Standardize direction so parameters match visually
    if b < 0:
        a, b, c = -a, -b, -c
        
    return a, b, c

# ==========================================
# Part (a): TLS on the first line data
# ==========================================
x1 = X_cols[:, 0]
y1 = Y_cols[:, 0]

a1, b1, c1 = total_least_squares(x1, y1)
print(f"--- Part (a): Line 1 Parameters ---")
print(f"Equation: {a1:.4f}x + {b1:.4f}y + {c1:.4f} = 0\n")

# ==========================================
# Part (b): RANSAC on all flattened data
# ==========================================
np.random.seed(42) # Seed for reproducibility
points = np.vstack((X_cols.flatten(), Y_cols.flatten())).T

def fit_line_ransac(points, iterations=5000, threshold=0.5):
    best_inliers = []
    
    for _ in range(iterations):
        idx = np.random.choice(len(points), 2, replace=False)
        p1, p2 = points[idx]
        
        a = p2[1] - p1[1]
        b = p1[0] - p2[0]
        c_val = p2[0]*p1[1] - p1[0]*p2[1]
        
        norm = np.hypot(a, b)
        if norm == 0: continue
        a, b, c_val = a/norm, b/norm, c_val/norm
        
        distances = np.abs(a * points[:, 0] + b * points[:, 1] + c_val)
        inliers = np.where(distances < threshold)[0]
        
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            
    # Refine the best model using TLS on the consensus set
    inlier_points = points[best_inliers]
    final_a, final_b, final_c = total_least_squares(inlier_points[:, 0], inlier_points[:, 1])
    
    return (final_a, final_b, final_c), best_inliers

print(f"--- Part (b): RANSAC Three Lines ---")
remaining_points = points.copy()

for i in range(3):
    model, inlier_idx = fit_line_ransac(remaining_points, iterations=5000, threshold=0.5)
    print(f"Line {i+1} Parameters: {model[0]:.4f}x + {model[1]:.4f}y + {model[2]:.4f} = 0 (Inliers: {len(inlier_idx)})")
    
    # Mask the consensus: remove inliers for the next iteration
    remaining_points = np.delete(remaining_points, inlier_idx, axis=0)