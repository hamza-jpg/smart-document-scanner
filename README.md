#  Smart Document Scanner (OpenCV)

> A robust, Python-based computer vision pipeline that automatically detects, flattens, and enhances documents from raw photographs.

Unlike standard edge-detection scripts that fail on complex backgrounds or low-contrast images, this scanner implements a **dynamic routing architecture** (a "Traffic Cop" system) that analyzes image heuristics in real-time to select the optimal detection algorithm.

---

##  Key Architectural Features

* **Pre-Flight Image Assessment:** Analyzes global contrast (Standard Deviation) and background clutter (Edge Density) in milliseconds to predict image complexity before processing.
* **Dynamic Algorithm Routing (Fast Path):** Utilizes lightning-fast Canny edge detection and contour mapping for high-contrast, clean environments.
* **Dynamic Algorithm Routing (Heavyweight Fallback):** Automatically routes washed-out or highly cluttered images (like white-on-white) to a GrabCut foreground segmentation model.
* **Extreme Corner Extraction:** Bypasses strict rigid geometry by mathematically isolating the 4 outermost points of any detected polygon, perfectly handling jagged edges like spiral notebook bindings.
* **3D Perspective Transform:** Calculates a transformation matrix to warp angled, trapezoidal document boundaries into a perfectly flat, top-down 2D perspective.
* **Adaptive Thresholding:** Applies a localized Gaussian threshold filter to eliminate uneven room lighting and shadow gradients, resulting in a crisp, iOS-style black-and-white scan.

---

##  Tech Stack

| Technology | Purpose |
| :--- | :--- |
| **Python 3.x** | Core programming language. |
| **OpenCV (`opencv-python`)** | Core matrix transformations, edge detection, and thresholding. |
| **NumPy** | Mathematical arrays and standard deviation calculations. |
| **Imutils** | Contour handling and image manipulation. |

---

##  How It Works (The Pipeline)

1. **Input:** Raw RGB photograph is loaded and resized for processing speed.
2. **Heuristic Check:** Calculates edge density (`<15%`) and contrast (`>35.0`).
3. **Detection:** Extracts the document boundary using either `cv2.Canny` + `cv2.approxPolyDP` or `cv2.grabCut`.
4. **Warp:** Calculates the geometric distance between the 4 extreme corners and applies `cv2.warpPerspective`.
5. **Filter:** Converts to grayscale and applies `cv2.adaptiveThreshold` to mimic scanned ink on paper.

---

##  Future Roadmap

While this classical CV pipeline handles ~90% of standard use cases, it reaches its mathematical limits with heavily wrinkled paper or extreme occlusions (e.g., a hand covering the document). The next iteration of this project would involve replacing the classical routing logic with a lightweight Deep Learning semantic segmentation model (like MobileNetV2) to guess occluded corners.

## 📖 User Manual: How to Scan Your Own Documents

Want to test this scanner on your own photos? It is incredibly easy to set up and run.

**1. Add Your Photo**
Place your `.jpg` or `.png` document photo directly into the same folder as the `scanner.py` script. 

**2. Point the Code to Your Image**
Open `scanner.py` in your editor and scroll to the very bottom of the file. Look for the `if __name__ == "__main__":` test block. 

Change the filename inside the `DocumentScanner` initialization to match the picture you just uploaded:

```python
# Change "test.jpg" to your actual file name!
if __name__ == "__main__":
    scanner = DocumentScanner("your_custom_photo.jpg") 
```

**3. Run the Scanner**
Execute the script in your terminal:
```bash
python scanner.py
```
The pipeline will run its Pre-Flight Checks, route the image, flatten the perspective, and display the final result on your screen. It will also automatically save a high-resolution copy named `scanned_document.jpg` directly to your hard drive!