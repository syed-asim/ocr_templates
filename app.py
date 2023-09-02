import cv2
import numpy as np
from preprocess import execute

source_img = './user_images/mufazil.jpg'
template_img = './roi_images/log.png'


# Load the user-uploaded image
# cv2.imread(source_img)
image = execute(source_img, "./user_images/id2_modified.jpg")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Load the template image (ROI template you want to find)
# Load the template in grayscale
# cv2.imread(template_img)
template = execute(template_img, "./roi_images/roi_id_no_modified.jpg")
gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
# Perform template matching
result = cv2.matchTemplate(gray_image, gray_template, cv2.TM_CCOEFF_NORMED)

# Define a threshold to consider a match
threshold = 0.6  # Adjust this threshold as needed

# Find the locations where the template matches above the threshold
locations = np.where(result >= threshold)

# Extract ROI coordinates
roi_coordinates = []

for pt in zip(*locations[::-1]):
    x, y = pt
    # h, w, _ = template.shape[::-1]
    # Get the width and height of the template
    w, h = gray_template.shape[::-1]
    roi_coordinates.append((x, y, x + w, y + h))

# Now, roi_coordinates contains the coordinates of all matched ROIs
print(roi_coordinates)

# Draw rectangles around matched ROIs for visualization (optional)
for (x1, y1, x2, y2) in roi_coordinates:
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Display the image with ROI rectangles (optional)
screen_width, screen_height = 1920, 1080  # Replace with your screen resolution
# Resize the image if it's larger than the screen
if image.shape[1] > screen_width or image.shape[0] > screen_height:
    image = cv2.resize(image, (screen_width, screen_height))
cv2.imshow('Matching Result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
