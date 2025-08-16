# Pancreas Segmentation + Subtype Classification (nnU-Netv2 Two-Head)

This repo contains code to reproduce our pancreas segmentation and case-level subtype classification using **nnU-Netv2** with a two-head architecture (segmentation + classification).  
It includes: data prep notes, training/inference commands, evaluation, and an inference runtime optimization (~10% faster).

**Key results (VAL):**
- Mean Dice (whole pancreas): **0.9561**
- Mean Dice (lesion): **0.8406**
- Macro-F1 (classification): **0.1667** (model collapsed to one class at inference; analysis inside)
