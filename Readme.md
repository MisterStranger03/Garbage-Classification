# ♻️ Waste Classification ML Project

## Overview
A machine learning project to classify waste into 6 categories using EfficientNetV2 and a custom Streamlit web app for interactive prediction.

---

## 🗂️ Classes
- **Plastic**
- **Metal**
- **Glass**
- **Paper**
- **Cardboard**
- **Trash**

---

## ✅ Weekly Progress

### 📅 Week 1
- Added initial datasets (train/val/test) organized into six classes
- Cleaned out corrupted/non‑image files to ensure smooth data loading  
- Learned image‑dataset pipelines (`image_dataset_from_directory`) and folder‑based labeling  
- Built data‑augmentation stack with random flip, rotation, zoom plus EfficientNetV2 preprocessor  
- Experimented with backbone selection: loaded `EfficientNetV2B0(weights="imagenet", include_top=False)`  
- Constructed & compiled initial model head (GlobalAveragePooling → BatchNorm → Dense → softmax)  

---

### 📅 Week 2
- Added additional images to enrich class balance and improve generalization  
- Trained only the new classification head for 5 epochs with EarlyStopping on validation loss  
- Unfroze top 20 layers of EfficientNetV2B0 and fine‑tuned on the waste dataset at a low LR  
- Monitored training via accuracy/loss curves, confusion matrix and classification reports  
- Achieved substantially improved validation accuracy through two‑phase transfer learning  

---

### 📅 Week 3
- Developed an interactive **Streamlit app** for Waste Classification  
- Features included:
  - Image upload, camera capture, and live webcam prediction
  - Real-time inference using `EfficientNetV2B2.keras` model
  - Class prediction with probability progress bar
  - Light/Dark mode toggle
  - Option to download prediction results as CSV
  - Styled UI with icons and animations
  - Footer credit: _"Made with ❤️ by Raman"_
- Optimized app layout:
  - Compact preview image display
  - Reduced scroll overhead
  - Custom button logic for “Start”/“Stop” live prediction
- Deployed via Streamlit Cloud and explored stlite (Pyodide) for GitHub Pages hosting

---

## 🛠️ Tech Stack
- **TensorFlow / Keras** (EfficientNetV2)
- **Streamlit**
- **Pandas, NumPy, Matplotlib**
- **OpenCV** (for live webcam)
- **stlite** (experimental browser-side hosting)

---

## 📦 Model
- Final model: `EfficientNetV2B2.keras`
- Trained on 6-class waste dataset with transfer learning
- Best validation accuracy achieved: **_X.XX%_** (fill in)

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
