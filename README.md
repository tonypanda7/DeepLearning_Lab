# Deep Learning Lab

Lab work for **CS3807 – Deep Learning Laboratory**, B.Tech AI & DS, Semester V, Shiv Nadar University Chennai.

Each folder holds a Jupyter notebook with the implementation and a PDF/TeX write-up of the experiment.

## Experiments

| Lab | Notebook | What it covers |
|-----|----------|----------------|
| 1 | [`lab1/DL_lab1.ipynb`](lab1/DL_lab1.ipynb) | Single-layer perceptron written from scratch in NumPy for binary classification on the UCI Banknote Authentication dataset, with weight/bias evolution and decision-boundary plots, benchmarked against scikit-learn's `Perceptron`. |
| 2 | [`lab2/DL_lab2.ipynb`](lab2/DL_lab2.ipynb) | Feedforward neural network on Fashion-MNIST in Keras — a baseline dense model, then hyperparameter tuning (depth, width, activation, learning rate) via `RandomizedSearchCV` with scikeras. |
| 3 | [`lab3/dl_lab3.ipynb`](lab3/dl_lab3.ipynb) | A CNN built from scratch for CIFAR-10 classification, plus feature-map visualisation of the convolutional layers and a full classification report / confusion matrix. |
| 4 | [`lab4/dl4-lab.ipynb`](lab4/dl4-lab.ipynb) | Classic architectures (LeNet-5, a mini AlexNet) versus transfer learning with VGG16, ResNet50, InceptionV3 and MobileNetV2 on CIFAR-10, comparing accuracy and training time across optimizers and batch sizes. |
| 5 | [`lab5/dl-lab5.ipynb`](lab5/dl-lab5.ipynb) | Fine-tuning MobileNetV2 on the 37-class Oxford-IIIT Pets dataset, studying weight initialisation, L2 regularisation, dropout, batch norm and optimizers, validated with K-fold cross-validation. |
| 6 | [`Lab6/lab6dl.ipynb`](Lab6/lab6dl.ipynb) | Recurrent sequence learning and video understanding: comparative study of Vanilla RNN, LSTM, and GRU on the UCI Human Activity Recognition (HAR) dataset with BPTT forward-pass verification and sequence-length analysis, extended to CNN–LSTM/GRU video classification with a frozen MobileNetV2 backbone and an encoder–decoder LSTM Seq2Seq model. |
| 7 | [`lab7/lab7dl.ipynb`](lab7/lab7dl.ipynb) | End-to-end study of autoencoders on MNIST: Fully Connected Autoencoder (FC-AE) with latent-dimension sweeps, Convolutional Autoencoder (CAE), Denoising CAE under Gaussian noise, and Variational Autoencoder (VAE) with a 2D latent space evaluated via MSE, MAE, SSIM, and latent manifold generation. |
| 8 & 9 | [`Lab8/Dl_lab_89.ipynb`](Lab8/Dl_lab_89.ipynb) | Attention mechanisms in a pretrained ResNet50 backbone on the Oxford Flowers (`tf_flowers`) dataset: comparative analysis of channel attention, spatial attention, and early vs. late attention insertion positions. |

## Stack

TensorFlow / Keras · scikit-learn · NumPy · pandas · Matplotlib · Seaborn · TensorFlow Datasets

## Running

```bash
git clone https://github.com/tonypanda7/DeepLearning_Lab.git
cd DeepLearning_Lab
jupyter notebook
```

Labs 2–5, 7, and 8 download their datasets automatically (Lab 6 auto-downloads the UCI HAR dataset and falls back to synthetic video data if UCF101 is not provided). Lab 1 expects `data_banknote_authentication.txt` from the UCI repository in the `lab1/` directory.
