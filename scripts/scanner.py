import cv2
import numpy as np
import imutils

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

    # Edge detection functionality
    def find_document_contour(self):
        greyscale = cv2.cvtColor(self.resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(greyscale, (5,5), 0)

        v = np.median(blurred)
        lower = int(max(0, (1.0 - 0.33) * v))
        upper = int(min(255, (1.0 + 0.33) * v))
        edged = cv2.Canny(blurred, lower, upper)

        cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse = True) [:5]
        self.screen_cnt = None

        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            if len(approx) == 4:
                self.screen_cnt = approx
                break

        if self.screen_cnt is not None:
            cv2.drawContours(self.resized, [self.screen_cnt], -1, (0,255,0,), 2)



# Test Block
if __name__ == "__main__":
    scanner = DocumentScanner("test.jpg")
    scanner.find_document_contour()
    cv2.imshow("Document Outline", scanner.resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()