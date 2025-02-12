import cv2
import pytesseract
import numpy as np

# Load the image
image_path = "IMG_2070.jpg"  # Change this to your actual image path
image = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blurring to remove noise
blurred = cv2.GaussianBlur(gray, (5,5), 0)

# Apply Otsu's Thresholding to binarize the image
_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Perform OCR with optimized settings
custom_config = "--psm 6"  # Page segmentation mode optimized for text blocks
extracted_text = pytesseract.image_to_string(thresh, config=custom_config)

# Print the extracted text
print("📝 Extracted Text:\n")
print(extracted_text)