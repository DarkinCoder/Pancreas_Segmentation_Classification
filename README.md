# Pancreas Segmentation + Subtype Classification (nnU-Netv2 Two-Head)

This project implements pancreas segmentation and case-level subtype classification using nnU-Netv2 with a custom two-head architecture (shared encoder, separate segmentation and classification heads).
Our goal was to build a single model capable of accurately identifying pancreas anatomy, segmenting lesions, and classifying cases into one of three subtypes.

We used the Dataset999_Pancreas (custom split: 252 training, 36 validation cases) in 3D full-resolution mode.
The segmentation head was trained following standard nnU-Netv2 procedures, and the classification head was fine-tuned on the bottleneck features of the shared encoder.

The classification task experienced mode collapse, with the model predicting a single class for all validation cases, resulting in a low Macro-F1 despite good segmentation performance. Detailed error analysis is provided in the train_class_predictions.csv file.

We also implemented a runtime optimization for inference by:

Disabling test-time augmentation (--disable_tta)

Adjusting the patch step size to reduce redundant computations

This led to a ~10% speed improvement in prediction runtime while maintaining segmentation accuracy.

Key Results (Validation set)
Mean Dice (Whole Pancreas)	0.9561	Combines normal pancreas (label=1) + lesion (label=2)
Mean Dice (Lesion)	0.8406	Dice score for lesion class only
Macro-F1 (Classification)	0.1667	Model collapsed to a single predicted class
