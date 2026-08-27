# Lunar Terrain Semantic Segmentation

A multi-class U-Net++ architecture designed for highly accurate lunar surface mapping. This pipeline prioritizes both precision and interpretability, extracting internal convolutional representations rather than treating the model as a pure black box.

**Context:**  Deep Learning Project Review-1 Submission

## Quantitative Performance
Evaluated across a rigorous 4-channel multi-class pipeline (e.g., sky, background regolith, boulders, craters) to ensure robust performance on minority classes.

| Metric | Test Set Score |
| :--- | :--- |
| **Pixel Accuracy** | 77.90% |
| **Mean Dice Coefficient** | 0.7808 |
| **Mean IoU (Jaccard)** | 0.6594 |
| **Test Loss (Dice)** | 0.2210 |

*(Note: Evaluated on an unseen test subset using exact one-hot encoded ground truth mapping).*

## Visual Telemetry & Interpretability

### 1. Training Convergence
The architecture demonstrates highly stable learning dynamics with near-zero validation divergence across 30 epochs.
<img width="5400" height="1500" alt="training_curves_output" src="https://github.com/user-attachments/assets/e41fe684-bc44-42a2-8ed3-5a8c461392de" />


### 2. Glass-Box Feature Extraction (Conv2D)
Internal layer activations isolate edge detection, target salience (high-reflectance boulders), and micro-shadow mapping.
<img width="3000" height="3000" alt="feature_maps_output" src="https://github.com/user-attachments/assets/52048b23-c1b9-480f-a252-fae4a8883c15" />


### 3. Segmentation Mask Prediction
<img width="3000" height="1500" alt="prediction_output" src="https://github.com/user-attachments/assets/90d53867-a1fd-4cf6-aa85-20b2d272e204" />


## Local Execution

**1. Environment Setup**
Ensure you have Python 3.12+ installed, then replicate the exact environment:
```bash
uv pip install -r requirements.txt
```
**2. Run the Pipeline**
Ensure lunar_dataset/ and the .keras weights file are in the project root.
```bash
python predict_and_analyze.py
```
