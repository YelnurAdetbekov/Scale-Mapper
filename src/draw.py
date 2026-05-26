import cv2
import numpy as np
image = np.zeros((500, 500, 3), dtype=np.uint8)

# 2. Define the rectangle parameters (RotatedRect format)
# Format: ((center_x, center_y), (width, height), rotation_angle_in_degrees)
center = (250, 250)
size = (200, 100)
angle = 45  # Rotate 45 degrees clockwise
rotated_rect = (center, size, angle)

# 3. Calculate the 4 absolute corner points of the rotated rectangle
box_points = cv2.boxPoints(rotated_rect)

# 4. Convert float coordinates to integers (required for drawing)
box_points = np.int64(box_points)

# 5. Draw the rotated rectangle on the image
# -1 indicates drawing the entire contour, (0, 255, 0) is green, 2 is thickness
cv2.drawContours(image, [box_points], 0, (0, 255, 0), 2)

# Display the result
cv2.imshow("Rotated Rectangle", image)
cv2.waitKey(0)
cv2.destroyAllWindows()