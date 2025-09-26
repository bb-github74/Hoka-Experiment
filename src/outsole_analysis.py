
---

## 📄 outsole_analysis.py

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def analyze_outsole_histogram(image_path):
    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image could not be loaded. Check the file path.")
    
    # Compute histogram
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist = hist.flatten()

    # Normalize histogram
    hist_norm = hist / hist.sum()

    # Basic measures
    mean_intensity = np.mean(img)
    variance = np.var(img)
    skewness = (np.mean((img - mean_intensity)**3)) / (np.std(img)**3 + 1e-9)
    
    # Peak count in histogram
    peaks = np.sum((hist[1:-1] > hist[:-2]) & (hist[1:-1] > hist[2:]))

    # Plot histogram
    plt.figure(figsize=(10,5))
    plt.title("Outsole Grayscale Histogram")
    plt.xlabel("Pixel Intensity (0=dark, 255=bright)")
    plt.ylabel("Frequency")
    plt.plot(hist, color='black')
    plt.show()

    return {
        "Mean Intensity": mean_intensity,
        "Variance": variance,
        "Skewness": skewness,
        "Histogram Peaks": int(peaks)
    }

# Example usage (replace with your outsole image path)
# results = analyze_outsole_histogram("outsole_sample.jpg")
# print(results)
