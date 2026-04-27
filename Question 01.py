import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor

# 1. LOAD DATA
try:
    D = np.genfromtxt("lines.csv", delimiter=",", skip_header=1)
    X_cols = D[:, :3]
    Y_cols = D[:, 3:]
except FileNotFoundError:
    print("Error: lines.csv not found in the current directory.")
    exit()


# (a) TOTAL LEAST SQUARES (TLS) - First Line Only
def total_least_squares(x, y):
    x_mean, y_mean = np.mean(x), np.mean(y)
    pts = np.vstack([x - x_mean, y - y_mean]).T
    
    _, _, vh = np.linalg.svd(pts)
    a, b = vh[1, :]
    
    # Solve for d using the centroid: ax + by + d = 0
    d = -(a * x_mean + b * y_mean)
    return a, b, d

x1, y1 = X_cols[:, 0], Y_cols[:, 0]
a, b, d = total_least_squares(x1, y1)

print("--- Question 1(a) ---")
print(f"TLS Parameters: {a:.4f}x + {b:.4f}y + {d:.4f} = 0")
print(f"Slope (m): {(-a/b):.4f}, Intercept (c): {(-d/b):.4f}\n")

# (b) SEQUENTIAL RANSAC - Three Lines
X_all = X_cols.flatten().reshape(-1, 1)
Y_all = Y_cols.flatten()

remaining_X = X_all.copy()
remaining_Y = Y_all.copy()
line_colors = ['red', 'green', 'blue']

plt.figure(figsize=(10, 6))
plt.scatter(X_all, Y_all, color='lightgray', label='Original Points', s=10)

print("--- Question 1(b) ---")
for i in range(3):
    # Fit RANSAC model
    ransac = RANSACRegressor(residual_threshold=0.5) 
    ransac.fit(remaining_X, remaining_Y)
    
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)
    
    slope = ransac.estimator_.coef_[0]
    intercept = ransac.estimator_.intercept_
    print(f"Line {i+1} found: y = {slope:.4f}x + {intercept:.4f}")
    
    line_x = np.linspace(X_all.min(), X_all.max(), 100).reshape(-1, 1)
    line_y = ransac.predict(line_x)
    
    # Visualization
    plt.plot(line_x, line_y, color=line_colors[i], label=f'Line {i+1}', linewidth=2)
    plt.scatter(remaining_X[inlier_mask], remaining_Y[inlier_mask], color=line_colors[i], s=15)
    
    remaining_X = remaining_X[outlier_mask]
    remaining_Y = remaining_Y[outlier_mask]

plt.title("Sequential RANSAC Line Fitting")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()