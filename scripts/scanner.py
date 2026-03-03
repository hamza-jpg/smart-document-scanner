import cv2
import numpy as np

class DocumentScanner:
    # Constructor
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not load image at {image_path}. "
                             f"Please check the file path again")
        # Resizing the image for better use of algorithms such as cany detection
        self.original = self.image.copy()
        if self.image.shape[0] > 500:
            self.ratio = self.image.shape[0] / 500.0
            self.resized = self._resize_image(self.image, height = 500)
        else:
            self.ratio = 1.0
            self.resized = self.image.copy()

    # Private resize functionality which keeps proportionality of the image
    def _resize_image(self, image, height):
        h, w = image.shape[:2]
        width = int ((height / h) * w)
        return cv2.resize(image, (width, height))

# Test Block
if __name__ == "__main__":
    scanner = DocumentScanner("test.jpg")
    cv2.imshow("Resized Original", scanner.resized)
    cv2.imshow("Original", scanner.image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()