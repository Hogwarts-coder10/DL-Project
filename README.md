# Lunar Terrain Semantic Segmentation

A multi-architecture pipeline designed for highly accurate lunar surface mapping. This framework prioritizes both precision and interpretability, extracting internal convolutional representations rather than treating the model as a pure black box.


## 🛠 Tech Stack
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)

---

## 🧠 Model Architectures
This framework supports dynamic swapping between three distinct segmentation models via a custom CLI factory:

* **U-Net++ (Nested):** Integrates nested dense skip pathways to reduce the semantic gap between encoder and decoder feature maps. *(Current Benchmarking Model)*
* **Attention U-Net (~34.8M Params):** Employs additive attention gates to actively suppress irrelevant background activations and highlight highly specific target regions (e.g., small lunar rocks).
* **Standard U-Net:** The baseline encoder-decoder architecture.

---

## 📊 Quantitative Performance
Evaluated across a rigorous 4-channel multi-class pipeline (sky, background regolith, boulders, craters) to ensure robust performance on minority classes.

| Metric | Test Set Score |
| :--- | :--- |
| **Pixel Accuracy** | 96.62% |
| **Mean Dice Coefficient** | 0.9653 |
| **Mean IoU (Jaccard)** | 0.9359 |
| **Test Loss (Dice)** | 0.0339 |

*(Note: Evaluated on an unseen test subset using exact one-hot encoded ground truth mapping).*

---

## 👁️ Visual Telemetry & Interpretability

### 1. Training Convergence
The architecture demonstrates highly stable learning dynamics with near-zero validation divergence across 30 epochs.
<img width="5400" height="1500" alt="training_curves_unet_plus_plus" src="https://github.com/user-attachments/assets/a2dc8650-7e4d-49a6-85aa-28c3c423521e" />


### 2. Glass-Box Feature Extraction (Conv2D)
Internal layer activations isolate edge detection, target salience (high-reflectance boulders), and micro-shadow mapping.
<img width="3000" height="3000" alt="feature_maps_unet_plus_plus" src="https://github.com/user-attachments/assets/4d885f61-e26b-4f29-a8b6-3eecdde1f9b3" />


### 3. Segmentation Mask Prediction
<img width="4500" height="6000" alt="comparison_grid_unet_plus_plus" src="https://github.com/user-attachments/assets/997c5570-0c5b-4713-b9b4-b538679a82b9" />


---

## 🚀 Local Execution & Pipeline

**1. Environment Setup**
Ensure you have Python 3.12+ installed, then replicate the exact environment:
```bash
uv pip install -r requirements.txt
```
**2. Training (Colab / GPU Ready)**
Launch the training loop for any architecture. Models are automatically checkpointed based on validation Dice scores.
```bash
python train.py --model attention_unet --epochs 50 --batch_size 16
```
**3. Evaluation & Inference**
Ensure lunar_dataset/ and the .keras weights file are in the project root. Run the analysis script to generate telemetry plots, feature maps, and true-color prediction grids:
```bash
python predict_and_analyze.py --model unetplusplus
```
