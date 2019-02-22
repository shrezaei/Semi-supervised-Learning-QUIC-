# Semi-supervised-Learning-QUIC-
Implementation of "[How to Achieve High Classification Accuracy with Just a Few Labels: A Semi-supervised Approach Using Sampled Packets](https://arxiv.org/pdf/1812.09761.pdf)". For more information read the paper.
## Prerequisites
Python 3 and Pytorch
## Dataset
Dataset is available in [QUIC Dataset](https://drive.google.com/drive/folders/1Pvev0hJ82usPh6dWDlz7Lv8L6h3JpWhE)
## Data Input format
It is assumed that each flow is converted into a text file containing 4 columns: Timestamp, relative time (from the first packet in the flow), packet lenght, direction. 
Statistical features are calculated in dataProcessInMemoryQUIC.py file.
## To reproduve our results
1. Run dataProcessInMemoryQUIC.py to do the pre-processing and caculating statistical features.
2. Run pre-training.py to train the model to predict statistical features.
3. Run re-training to transfer the weight from previous step and re-train the model to predict class labels.
