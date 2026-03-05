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

        self.screen_cnt = None

    # Private resize functionality which keeps proportionality of the image
    def _resize_image(self, image, height):
        h, w = image.shape[:2]
        width = int ((height / h) * w)
        return cv2.resize(image, (width, height))

    # Routes the image to the best algorithm
    def find_document_contour(self):
        is_complex = self._needs_heavy_algorithm()

        if is_complex:
            print("Image flagged as complex (low contrast or high noise)")
            self.screen_cnt = self._slow_grabcut_detection()
        else:
            print("Image looks clean. Routing to Fast Edge Detection")
            self.screen_cnt = self._fast_edge_detection()

            # When it fails try grabcut just in case
            if self.screen_cnt is None:
                print("Fast method unexpectedly failed. Falling back to grabcut...")

        return self.screen_cnt

    # Analyzes image contrast and noise to predict the best algorithm.
    def _needs_heavy_algorithm(self):
        grayscale = cv2.cvtColor(self.resized, cv2.COLOR_BGR2GRAY)

        # Measuring contrast (standard deviation)
        std_dev = np.std(grayscale)

        # Checking clutter
        edges = cv2.Canny(grayscale, 50, 150)
        edges_pixels = np.count_nonzero(edges)
        total_pixels = edges.shape[0] * edges.shape[1]
        edge_density = (edges_pixels / total_pixels) * 100

        print(f"[Pre-checking] Contrast: {std_dev:.2f} | Edge Clutter: {edge_density:.2f}%")

        if std_dev < 35.0 or edge_density > 15.0:
            return True

        return False

    # As the name says it is the fast algo based on grayscale and canny
    def _fast_edge_detection(self):
        greyscale = cv2.cvtColor(self.resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(greyscale, (5,5), 0)

        # Calculates image thresholds using median value
        v = np.median(blurred)
        lower = int(max(0, (1.0 - 0.33) * v))
        upper = int(min(255, (1.0 + 0.33) * v))

        edged = cv2.Canny(blurred, lower, upper)
        # cv2.imshow("Debug: Edges", edged) (Test)

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

    # Background segmentation for tricky images
    def _slow_grabcut_detection(self):
        mask = np.zeros(self.resized.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1,65), np.float64)

        h, w = self.resized.shape[:2]
        rect = (1, 1, w - 2, h - 2)

        cv2.grabCut(self.resized, mask, rect, bgdModel, fgdModel,
                    5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0 ,255).astype('uint8')

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
        mask_closed = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
        mask_blurred = cv2.GaussianBlur(mask_closed, (11,11), 0)
        _, mask_clean = cv2.threshold(mask_blurred, 127,255,
                                   cv2.THRESH_BINARY)

        cnts = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        if len(cnts) > 0:
            c = max(cnts, key = cv2.contourArea)
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            if len(approx) == 4:
                return approx
            else:
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect)
                return np.int32(box)

        return None

    # Helper method for perspective transformation gives corner coordinates
    def _order_points(self, pts):
        rect = np.zeros((4,2), dtype = "float32")

        # Top left and bottom right corner
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # Top right and bottom left corner
        diff = np.diff(pts, axis = 1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect

    # Warping the document to top down perspective
    def get_scan(self, contour):
        pts = contour.reshape(4,2) * self.ratio
        rect = self._order_points(pts)
        (tl, tr, br, bl) = rect

        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0,0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype = "float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(self.original, M, (maxWidth, maxHeight))

        return warped

# Test Block
if __name__ == "__main__":
    scanner = DocumentScanner("../pictures/sudoku.png")
    doc_contour = scanner.find_document_contour()

    if doc_contour is not None:
        cv2.drawContours(scanner.resized, [doc_contour], -1,
                         (0, 255, 0), 2)
        cv2.imshow("STEP 1: Detected Document", scanner.resized)

        final_scan = scanner.get_scan(doc_contour)
        cv2.imshow("STEP 2: Flattened Scan", final_scan)
    else:
        print("Both algorithms failed")

    cv2.waitKey(0)
    cv2.destroyAllWindows()