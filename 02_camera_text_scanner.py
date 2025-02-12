import cv2
import pytesseract
import numpy as np
import os
import time

# Custom save directory
SAVE_DIR = "scanned_texts"
IMAGE_FILENAME = "captured_image.jpg"
ROI_FILENAME = "cropped_roi.jpg"

# Full paths
IMAGE_PATH = os.path.join(SAVE_DIR, IMAGE_FILENAME)
ROI_PATH = os.path.join(SAVE_DIR, ROI_FILENAME)

# Ensure the save directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

def capture_image():
    """ Captures an image from the camera and immediately allows RoI selection. """
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Warning:: Error: Camera not accessible. Retrying...")
        time.sleep(2)
        cap = cv2.VideoCapture(0)  # Retry opening the camera
        if not cap.isOpened():
            print("Warning:: Error: Camera failed to open. Exiting.")
            cleanup_and_exit()

    retry_count = 0
    max_retries = 5  # Prevent infinite loops if camera gets stuck

    while retry_count < max_retries:
        ret, frame = cap.read()
        
        if not ret:
            print(f"⚠️ Warning: Failed to capture frame ({retry_count+1}/{max_retries}). Retrying...")
            retry_count += 1
            time.sleep(1)  # Wait before retrying
            continue  # Retry camera capture

        overlay_text(frame, "Instructions:", (20, 50), (255, 0, 0), font_scale=1.2, thickness=3)
        overlay_text(frame, "Step 1: Press 'C' to Capture", (20, 90), (255, 0, 0))
        overlay_text(frame, "Step 2: Select the Region of Interest", (20, 130), (255, 0, 0))
        overlay_text(frame, "Step 3: Confirm OCR Text", (20, 170), (255, 0, 0))
        overlay_text(frame, "Press 'Q' to Quit", (20, 210), (0, 0, 255))
        overlay_text(frame, "_________________________", (20, 230), (255, 0, 0))
        cv2.imshow("Capture Mode", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            cv2.imwrite(IMAGE_PATH, frame)
            cap.release()
            cv2.destroyAllWindows()
            return select_roi(IMAGE_PATH)
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            cleanup_and_exit()

    print("Warning:: Error: Camera failed after multiple attempts. Exiting.")
    cap.release()
    cv2.destroyAllWindows()
    cleanup_and_exit()

def select_roi(image_path):
    """ Opens the captured image and allows the user to select a Region of Interest (RoI). """
    image = cv2.imread(image_path)

    if image is None:
        print("Warning:: Error: Image could not be loaded.")
        return None

    while True:
        cv2.imshow("Select RoI", image)
        roi = cv2.selectROI("Select RoI", image, fromCenter=False, showCrosshair=True)

        if roi[2] == 0 or roi[3] == 0:
            print("Warning:: No RoI selected. Try again.")
            continue  # Let user retry

        # Crop the selected region
        cropped_roi_path = ROI_PATH
        cropped_roi = image[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
        cv2.imwrite(cropped_roi_path, cropped_roi)
        cv2.destroyAllWindows()
        return extract_text_and_display(cropped_roi_path)

def preprocess_image(image_path):
    """ Preprocesses the image to enhance text recognition. """
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def extract_text_from_image(image_path):
    """ Extracts text from the cropped RoI using OCR. """
    processed_image = preprocess_image(image_path)
    return pytesseract.image_to_string(processed_image, config="--psm 6")

def overlay_text(image, text, position, color=(0, 255, 0), font_scale=0.8, thickness=2):
    """ Draws overlayed text on the image. """
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

def generate_unique_filename():
    """ Generates a unique filename for each saved scan. """
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return os.path.join(SAVE_DIR, f"scanned_text_{timestamp}.txt")

def save_text(text):
    """ Saves the extracted text to a unique file. """
    filename = generate_unique_filename()
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"info:: Extracted text saved at: {filename}")

def extract_text_and_display(image_path):
    """ Extracts text and displays it instead of the cropped image. """
    extracted_text = extract_text_from_image(image_path)
    
    # Create a blank white image for displaying text
    text_display = np.ones((500, 800, 3), dtype=np.uint8) * 255  # White background
    
    # Break the text into multiple lines for display
    lines = extracted_text.split("\n")
    y = 50
    for line in lines:
        if line.strip():
            overlay_text(text_display, line, (20, y), (0, 0, 0), font_scale=0.7)
            y += 30  # Adjust spacing
    
    overlay_text(text_display, "info:: Done! Press 'S' to Save, 'R' to Retake, 'Q' to Quit", (20, 480), (255, 0, 0))

    while True:
        cv2.imshow("OCR Result", text_display)
        key = cv2.waitKey(0) & 0xFF

        if key == ord('s'):  # Save extracted text
            save_text(extracted_text)
            overlay_text(text_display, "info:: Saved Successfully!", (20, 420), (0, 255, 0))
            cv2.imshow("OCR Result", text_display)
            cv2.waitKey(1000)  # Show message briefly
            cleanup_and_exit()
            break
        elif key == ord('r'):  # Retake image
            main()
            return
        elif key == ord('q'):  # Quit
            cleanup_and_exit()
            break

    cv2.destroyAllWindows()

def cleanup_and_exit():
    """ Deletes temporary images and exits the program. """
    if os.path.exists(IMAGE_PATH):
        os.remove(IMAGE_PATH)
    if os.path.exists(ROI_PATH):
        os.remove(ROI_PATH)
    print("🧹 Temporary files cleaned. Exiting program.")
    cv2.destroyAllWindows()
    exit()

def main():
    """ Main function that captures an image, allows RoI selection, and performs OCR. """
    capture_image()

if __name__ == "__main__":
    main()