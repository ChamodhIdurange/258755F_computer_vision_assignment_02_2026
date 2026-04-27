import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# SETUP & MANUAL CLICKING (From Listing 1)
# ==========================================
N = 6  # Using 6 points as requested in the assignment text
n = 0
p1 = np.empty((N, 2))
p2 = np.empty((N, 2))

def draw_circle(event, x, y, flags, param):
    global n
    p = param[0]
    # Check if left button clicked and we haven't exceeded N points
    if event == cv.EVENT_LBUTTONDOWN and n < N:
        cv.circle(param[1], (x, y), 5, (255, 0, 0), -1)
        p[n] = (x, y)
        n += 1

# Load images (Update paths if they are in an 'a2_images' folder)
im1 = cv.imread('c1.jpg')
im2 = cv.imread('c2.jpg')

if im1 is None or im2 is None:
    print("Error: Could not load images. Please ensure 'c1.jpg' and 'c2.jpg' exist.")
    exit()

im1copy = im1.copy()
im2copy = im2.copy()

# Collect points for Image 1
cv.namedWindow('Image 1', cv.WINDOW_AUTOSIZE)
cv.setMouseCallback('Image 1', draw_circle, [p1, im1copy])
print(f"Please click {N} distinct points on Image 1 (Press ESC to skip/exit early)...")

while True:
    cv.imshow("Image 1", im1copy)
    if n == N:
        break
    if cv.waitKey(20) & 0xFF == 27: # ESC key
        break
cv.destroyWindow('Image 1')

# Collect points for Image 2
n = 0 # Reset counter for the second image
cv.namedWindow('Image 2', cv.WINDOW_AUTOSIZE)
cv.setMouseCallback('Image 2', draw_circle, [p2, im2copy])
print(f"Please click the SAME {N} corresponding points on Image 2...")

while True:
    cv.imshow("Image 2", im2copy)
    if n == N:
        break
    if cv.waitKey(20) & 0xFF == 27:
        break
cv.destroyWindow('Image 2')

print("\nPoints collected! Processing Homography and SIFT...\n")

# ==========================================
# PART (A) & (B): MANUAL HOMOGRAPHY
# ==========================================
src_pts_manual = np.float32(p1)
dst_pts_manual = np.float32(p2)

# Compute homography and warp
H_manual, _ = cv.findHomography(src_pts_manual, dst_pts_manual)
h, w, _ = im2.shape
warped_im1_manual = cv.warpPerspective(im1, H_manual, (w, h))

# Subtract and threshold to find differences
diff_manual = cv.absdiff(im2, warped_im1_manual)
diff_manual_gray = cv.cvtColor(diff_manual, cv.COLOR_BGR2GRAY)
_, diff_thresh_manual = cv.threshold(diff_manual_gray, 30, 255, cv.THRESH_BINARY)


# ==========================================
# PART (C) & (D): AUTOMATED SIFT MATCHING
# ==========================================
sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(im1, None)
kp2, des2 = sift.detectAndCompute(im2, None)

# Match descriptors
bf = cv.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Apply ratio test to filter out poor matches
good_matches = []
for m, n_match in matches:
    if m.distance < 0.75 * n_match.distance:
        good_matches.append(m)

# Draw the SIFT matches for visual confirmation
matched_img = cv.drawMatches(im1, kp1, im2, kp2, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Extract coordinates of good matches
src_pts_auto = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts_auto = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Compute Homography using RANSAC and warp
H_auto, _ = cv.findHomography(src_pts_auto, dst_pts_auto, cv.RANSAC, 5.0)
warped_im1_auto = cv.warpPerspective(im1, H_auto, (w, h))

# Subtract and threshold
diff_auto = cv.absdiff(im2, warped_im1_auto)
diff_auto_gray = cv.cvtColor(diff_auto, cv.COLOR_BGR2GRAY)
_, diff_thresh_auto = cv.threshold(diff_auto_gray, 30, 255, cv.THRESH_BINARY)


# ==========================================
# DISPLAYING THE RESULTS FOR THE REPORT
# ==========================================
# Using matplotlib to generate a nice grid for your PDF report
plt.figure(figsize=(15, 10))

# Manual Results
plt.subplot(2, 3, 1), plt.imshow(cv.cvtColor(warped_im1_manual, cv.COLOR_BGR2RGB)), plt.title("Manual Warped Image")
plt.subplot(2, 3, 2), plt.imshow(diff_thresh_manual, cmap='gray'), plt.title("Manual Difference (Thresholded)")

# SIFT Results
plt.subplot(2, 3, 4), plt.imshow(cv.cvtColor(warped_im1_auto, cv.COLOR_BGR2RGB)), plt.title("SIFT Warped Image")
plt.subplot(2, 3, 5), plt.imshow(diff_thresh_auto, cmap='gray'), plt.title("SIFT Difference (Thresholded)")
plt.subplot(2, 3, 6), plt.imshow(cv.cvtColor(matched_img, cv.COLOR_BGR2RGB)), plt.title("SIFT Feature Matches")

plt.tight_layout()
plt.show()