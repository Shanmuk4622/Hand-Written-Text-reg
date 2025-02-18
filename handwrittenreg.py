import cv2
import easyocr

# Load the image
image = cv2.imread("img_2.png")


# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding for better text recognition
_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

# Initialize the EasyOCR model (downloads pre-trained model automatically)
reader = easyocr.Reader(["en"])

# Recognize text from the image
result = reader.readtext(thresh)

# Print recognized text
print("\nDetected Text:")
for detection in result:
    bbox, text, confidence = detection
    print(f"- {text} (Confidence: {confidence:.2f})")

# Show the processed image
cv2.imshow("Processed Image", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()
