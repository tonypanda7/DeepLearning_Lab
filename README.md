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

## Stack

TensorFlow / Keras · scikit-learn · NumPy · pandas · Matplotlib · Seaborn · TensorFlow Datasets

## Running

```bash
git clone https://github.com/tonypanda7/DeepLearning_Lab.git
cd DeepLearning_Lab
jupyter notebook
```

Labs 2–5 download their datasets automatically. Lab 1 expects `data_banknote_authentication.txt` from the UCI repository in the `lab1/` directory.
