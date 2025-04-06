# MorpHista

**MorpHista** is an application for morphology-based histological image segmentation and tissue analysis. It automates the detection and quantification of interstitial cells of Cajal (ICC) in histopathological images using state-of-the-art deep neural network techniques and classical image processing methods.

---

## Citation

If you use this work in your research or projects, please cite the original article as:

```bibtex
@article{Fisere2025,
  author = {Fišere, I. and Edelmers, E. and Svirskis, Š. and Groma, V.},
  title = {Utilisation of Deep Neural Networks for Estimation of Cajal Cells in the Anal Canal Wall of Patients with Advanced Haemorrhoidal Disease Treated by LigaSure Surgery},
  journal = {Cells},
  year = {2025},
  volume = {14},
  number = {7},
  article-number = {550},
  doi = {10.3390/cells14070550},
  url = {https://doi.org/10.3390/cells14070550}
}
```

You can access the full article [here](https://doi.org/10.3390/cells14070550).

---

## Overview

MorpHist was developed as part of a research project focused on improving the accuracy and reproducibility of histological analysis. By integrating a AI model for cell detection with advanced tissue segmentation methods, MorpHist facilitates:
- **Automated Detection:** Identification of ICC cells in whole slide images.
- **Tissue Segmentation:** Generation of tissue masks.
- **Batch Processing:** Capability to process single images or entire folders, enabling high-throughput analysis.
- **Detailed Reporting:** Annotated output images and comprehensive statistical summaries for each processed image.

The methodologies implemented in MorpHist are detailed in the accompanying research paper.

---

## Features

- **Deep Learning Detection:** Utilizes a AI model to detect ICC cells.
- **Tissue Mask Generation:** Uses adaptive thresholding and filtering to create tissue masks.
- **Whole Slide Image Processing:** Handles large histological images.
- **User-Friendly GUI:** The interface supports both single-image and batch processing modes.
- **Configurable Parameters:** Easily adjust detection thresholds, kernel sizes, pixel dimensions, annotation styles, and more through the GUI.
- **Persistent Settings:** User preferences are saved between sessions to streamline repetitive tasks.

---

## Contact
For questions, suggestions, or bug reports, please contact:
- Edgars Edelmers – edgars.edelmers@disroot.org

Or open an issue on the GitHub repository.
