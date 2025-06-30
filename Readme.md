# Waste Classification ML Project

## Week 1
- Added initial datasets (train/val/test) organized into six classes: plastic, metal, glass, paper, cardboard, trash  
- Cleaned out corrupted/non‑image files to ensure smooth data loading  
- Learned image‑dataset pipelines (`image_dataset_from_directory`) and folder‑based labeling  
- Built data‑augmentation stack with random flip, rotation, zoom plus EfficientNetV2 preprocessor  
- Experimented with backbone selection: loaded `EfficientNetV2B0(weights="imagenet", include_top=False)`  
- Constructed & compiled initial model head (GlobalAveragePooling → BatchNorm → Dense → softmax)  

## Week 2
- Added additional images to enrich class balance and improve generalization  
- Trained only the new classification head for 5 epochs with EarlyStopping on validation loss  
- Unfroze top 20 layers of EfficientNetV2B0 and fine‑tuned on the waste dataset at a low LR  
- Monitored training via accuracy/loss curves, confusion matrix and classification reports  
- Achieved substantially improved validation accuracy through two‑phase transfer learning  
