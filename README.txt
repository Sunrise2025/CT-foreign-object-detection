This repository contains the implementation of the proposed CT foreign object detection framework, including two main modules:

1) data_generation: This module implements the single-sample data generation method without preset foreign object quantity. It includes CT projection preprocessing, 3D reconstruction, material segmentation, structural standardization, and forward projection-based data synthesis.

2) training_strategy: This module implements the ES-BS (Early Stopping–Best Selection) training control strategy. It provides the training pipeline, early stopping mechanism, and model selection method based on sliding window statistics.

These two modules jointly support the complete workflow of data construction and model training in this study.
