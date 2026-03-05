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
        # Resizing the image for better use of algorithms such as canny edge detection
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


    def _fast_edge_detection(self):
        greyscale = cv2.cvtColor(self.resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(greyscale, (5,5), 0)

        # Calculates image thresholds using median value
        v = np.median(blurred)
        lower = int(max(0, (1.0 - 0.33) * v))
        upper = int(min(255, (1.0 + 0.33) * v))

        edged = cv2.Canny(blurred, lower, upper)
        cv2.imshow("Debug: Edges", edged)

        cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

        # image_area = self.resized.shape[0] * self.resized.shape[1] (Test)

        # Looking for largest four sided polygon
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            # contour_area = cv2.contourArea(c) (Test)
            print(f"Shape found with {len(approx)} corners")

            if len(approx) == 4:
                print("Found a valid, large document!")
                return approx

        return None

# Test Block
if __name__ == "__main__":
    scanner = DocumentScanner("../pictures/sudoku.png")
    doc_contour = scanner._fast_edge_detection()
    if doc_contour is not None:
        cv2.drawContours(scanner.resized, [doc_contour], -1,
                         (0, 255, 0), 2)
    else:
        print("No document found to draw.")
    cv2.imshow("Document Outline", scanner.resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()