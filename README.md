# Beyond Clean Accuracy: A Deep Dive into the Robustness and Transferability of Vision Transformers for Urdu Handwritten Character Recognition

**Author:** Zaryab Rahman


---

<div align="center">
 
  <img src="notebooks/complete-summary-of-paper-expirements.png" alt="Study Framework" width="400" height="500"/>
  <p><em>Fig 1: The experimental framework comparing CNNs and ViTs across clean data, corrupted data, and cross-lingual transfer learning.</em></p>
</div>

---

## 🚀 Project Overview

Automated recognition of **Urdu Handwritten Characters** is a complex task due to the cursive Nastaliq script. Current research focuses almost exclusively on "clean" data accuracy, ignoring how models perform in the real world (scans with noise, blur, or compression).

**This study challenges that paradigm.** We conducted the first systematic "stress test" of 7 modern architectures (CNNs vs. Vision Transformers) to answer: **Which models are actually reliable?**

### 💡 Key Breakthroughs
1.  **The Fragility of Efficiency:** Lightweight CNNs (MobileNetV2, EfficientNet), while fast and accurate on clean data, **collapse** under noise.
2.  **The Power of Transformers:** Vision Transformers (ViTs) demonstrate superior resilience to structural corruption.
3.  **The Arabic Advantage:** Pre-training on **Arabic script** (instead of ImageNet) boosts the Swin Transformer to State-of-the-Art performance:
    *   **Accuracy:** 92% ➔ **99%**
    *   **Training Time:** Reduced by **40%**

---

## 📊 Experiment 1: The Baseline (Clean Data)

We first established a baseline on the uncorrupted UHAT dataset. CNNs initially dominate due to their inductive bias for local features (strokes and dots).

| Model | Type | Params (M) | GFLOPs | Accuracy | F1-Score | Inference (ms/img) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet-18** | CNN | 11.20 | 3.65 | **0.9882** | 0.9634 | 3.06 |
| **ResNet-50** | CNN | 23.59 | 8.27 | 0.9869 | **0.9924** | 3.54 |
| **MobileNetV2-100**| CNN | **2.28** | **0.60** | 0.9810 | 0.9569 | **2.86** |
| **EfficientNet-B0**| CNN | 4.06 | 0.77 | 0.9876 | 0.9629 | 2.90 |
| ViT-Tiny | ViT | 5.53 | 2.15 | 0.9606 | 0.9379 | 2.86 |
| Swin-Tiny | ViT | 27.55 | 8.74 | 0.9216 | 0.9000 | 4.65 |
| DeiT-Tiny | ViT | 5.54 | 2.16 | 0.9743 | 0.9503 | 3.05 |

<div align="center">
  <img src="Data/eff-vs-size.png" alt="Baseline Accuracy" width="600"/>
</div>

<div align="center">
  <img src="Data/Acc-vs-f1-score.png" alt="Baseline Accuracy" width="600"/>
</div>

---

## 🛡️ Experiment 2: Robustness Analysis (The Stress Test)

This is the core contribution of our work. We subjected all models to 6 types of corruption (Gaussian Noise, Motion Blur, etc.) at 5 severity levels.

**The Result:** The hierarchy shatters. The efficient CNNs are "Fragile Specialists," failing completely under high-frequency noise. ViTs, particularly **Swin** and **DeiT**, prove to be much more robust contenders.

### Normalized Robustness AUC (Higher is Better)

| Model | Gaussian Noise | JPEG | Motion Blur | Occlusion | Rotation | Salt & Pepper |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| ResNet18 | **0.978** | **0.988** | 0.283 | 0.546 | 0.673 | 0.641 |
| ResNet50 | 0.918 | 0.987 | 0.272 | 0.575 | **0.682** | 0.100 |
| MobileNetV2 | ❌ 0.070 | 0.974 | 0.256 | 0.606 | 0.643 | 0.359 |
| EfficientNet-B0 | ❌ 0.036 | 0.985 | 0.197 | **0.604** | 0.639 | 0.040 |
| ViT-Tiny | 0.956 | 0.961 | 0.417 | 0.554 | 0.484 | 0.810 |
| **Swin-Tiny** | 0.920 | 0.922 | **0.511** | 0.480 | 0.487 | 0.784 |
| **DeiT-Tiny** | 0.972 | 0.974 | 0.303 | 0.587 | 0.488 | **0.856** |

<div align="center">
  <!-- 🔴 PLACEHOLDER: Insert 'Figure 11' (The Heatmap) here. This is visually very important. -->
  <img src="Data/robustness_auc_heatmap.png" alt="Robustness Heatmap" width="700"/>
  <p><em>Fig 2: Heatmap of Normalized Robustness AUC. Note the dark purple cells for lightweight CNNs indicating collapse under noise.</em></p>
</div>

---

## 🌍 Experiment 3: Cross-Lingual Transfer Learning

Standard ImageNet pre-training is suboptimal for handwriting. We hypothesized that pre-training on **Arabic** (visually similar to Urdu) would be superior.

**The results were transformative.** The Swin Transformer, which struggled with ImageNet weights, became the best performing model.

| Model | Pre-training Source | Accuracy (%) | F1-Score | Train Time (s) |
| :--- | :--- | :---: | :---: | :---: |
| DeiT-Tiny | ImageNet | 97.00 | 0.95 | 2680 |
| **DeiT-Tiny** | **Arabic (Ours)** | **98.00** | **0.96** | **2680** |
| Swin-Tiny | ImageNet | 92.00 | 0.90 | 5587 |
| **Swin-Tiny** | **Arabic (Ours)** | **99.00** 🚀 | **0.96** | **3327** 📉 |

<div align="center">
  <img src="Data/summary_comparison_arabic_vs_imagenet.png" alt="Transfer Learning Impact" width="800"/>
  <p><em>Fig 3: Arabic pre-training (Orange) consistently outperforms ImageNet pre-training (Blue) in both Accuracy and Speed.</em></p>
</div>

---

## 🧠 Explainable AI (XAI) Visualizations

We utilized **Grad-CAM** to verify *why* Transformers are more robust.

### Case Study: Motion Blur
The **Swin Transformer** (Top) successfully reconstructs the character features despite the blur. **ResNet-18** (Bottom) fails, looking at the wrong features.

<div align="center">
  <!-- 🔴 PLACEHOLDER: Insert 'Figure 20' (Swin vs ResNet on Motion Blur) here -->
  <img src="path/to/your/xai_motion_blur.png" alt="XAI Motion Blur" width="500"/>
</div>

### Case Study: Salt & Pepper Noise
**DeiT** (Top) ignores the noise pixels and focuses on the stroke. **EfficientNet** (Bottom) gets confused by the noise and fails.

<div align="center">
  <!-- 🔴 PLACEHOLDER: Insert 'Figure 19' (DeiT vs EfficientNet on Salt & Pepper) here -->
  <img src="path/to/your/xai_salt_pepper.png" alt="XAI Salt and Pepper" width="500"/>
</div>

---

## 📝 Conclusion

1.  **Context Matters:** Do not choose architectures based on clean accuracy alone. If the deployment environment is noisy, lightweight CNNs are risky.
2.  **Reliability:** ResNet remains a safe baseline, but ViTs offer specialized robustness against structural corruption.
3.  **Optimal Strategy:** For Urdu HCR, the **Swin Transformer pre-trained on Arabic data** is the definitive winner, offering the best trade-off between accuracy, robustness, and training efficiency.

---
