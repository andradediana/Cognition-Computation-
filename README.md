# Deep Belief Networks on Fashion-MNIST: Representations, Noise and Adversarial Robustness

Final project for **Cognition & Computation** (A.Y. 2025/26), Department of Mathematics "Tullio Levi-Civita", University of Padua.

**Student:** Diana Cristina Andrade Damian (2141895)

---

## Overview

This project explores the behaviour of a **Deep Belief Network (DBN)**, a stack of three Restricted Boltzmann Machines (RBMs) following the architecture of Hinton & Salakhutdinov (2006), on the **Fashion-MNIST** dataset. It goes beyond raw accuracy to study what each layer of the hierarchy learns and how it compares with a standard feed-forward network on two hard cases: **noisy inputs** and **adversarial attacks**.

The notebook is organised in three parts:

1. **Deep Belief Network**: unsupervised training, visualization of receptive fields, clustering of internal representations, and supervised linear read-out on each layer.
2. **Comparison with a feed-forward neural network (FFNN)**: same layer sizes, trained end-to-end, then tested under Gaussian noise.
3. **Adversarial attacks**: FGSM attacks on both models, and a defence for the DBN based on top-down reconstruction.

---

## Repository contents

| File | Description |
|---|---|
| `CC_Final_Project__Diana_Andrade_.ipynb` | Full notebook with code, outputs, plots and written analysis |

The notebook downloads its data and the DBN implementation at runtime, so nothing else is needed in the repo.

## Dataset

[Fashion-MNIST (Zalando Research, Kaggle)](https://www.kaggle.com/datasets/zalando-research/fashionmnist): 60,000 training and 10,000 test grayscale images of 28×28 pixels, in 10 classes.

| Label | Class | Label | Class |
|---|---|---|---|
| 0 | T-shirt/top | 5 | Sandal |
| 1 | Trouser | 6 | Shirt |
| 2 | Pullover | 7 | Sneaker |
| 3 | Dress | 8 | Bag |
| 4 | Coat | 9 | Ankle boot |

**Preprocessing:** the CSV files (`fashion-mnist_train.csv`, `fashion-mnist_test.csv`) have 785 columns (label + 784 pixels). Pixels are divided by 255 to lie in [0, 1]. The training set is split into **50,400 training** and **9,600 validation** images (`test_size=0.16`, `random_state=42`); the 10,000 test images are held out.

---

## Methods

### DBN and unsupervised training
- Architecture: 784 → 700 → 500 → 300 hidden units (3 stacked RBMs)
- Learning: Contrastive Divergence (k=1), 80 epochs per layer, batch size 125
- Hyperparameters: learning rate 0.1, momentum 0.5 → 0.95, weight decay 0.0001
- DBN implementation: [flavio2018/Deep-Belief-Network-pytorch](https://github.com/flavio2018/Deep-Belief-Network-pytorch) (`DBN.py`, `RBM.py`, downloaded with `wget`)

### Analysis of the hierarchy
- **Receptive fields**: learned weights of each layer, thresholded and min-max scaled. Deeper layers are projected into pixel space by matrix multiplication.
- **Hierarchical clustering**: per-class centroids of the third-layer representation, clustered with complete linkage and shown as a dendrogram.

### Supervised linear read-out
A separate linear classifier is trained on the hidden representation of each layer (Adam, lr 0.001, up to 3000 epochs, early stopping with patience 50), to measure how much class information each layer holds.

### Feed-forward baseline
A 784-700-500-300-10 ReLU network with the same layer sizes, trained end-to-end for 3080 epochs (3000 supervised + 80 to match the unsupervised phase).

### Robustness tests
- **Noise**: Gaussian noise added to test images at increasing levels (0 to 2), evaluated on the DBN read-outs and the FFNN.
- **Adversarial attack**: FGSM with ε from 0 to 0.3 on a combined DBN + third-layer read-out model and on the FFNN.
- **Defence**: 2 and 10 top-down reconstruction steps (hidden → visible → hidden) applied to the attacked input before classification.

---

## Results

### Clean test accuracy

| Model | Test accuracy |
|---|---|
| DBN, layer 1 read-out (700 units) | 88.97% |
| DBN, layer 2 read-out (500 units) | 88.24% |
| DBN, layer 3 read-out (300 units) | 87.30% |
| Feed-forward network | 88.61% |

A linear classifier on unsupervised DBN features is roughly on par with a non-linear network trained end-to-end.

### Noise robustness (Gaussian noise level 0.3)

| Model | Accuracy |
|---|---|
| DBN layer 1 read-out | 82.1% |
| DBN layer 2 read-out | 85.8% |
| DBN layer 3 read-out | 85.5% |
| Feed-forward network | 45.0% |

Across the full noise sweep, all three DBN layers stay above the FFNN, with the deepest layers the most robust.

### Adversarial robustness (FGSM, ε = 0.2)

| Model | Accuracy |
|---|---|
| FFNN | 8.83% |
| DBN | 14.60% |
| DBN + 2 top-down reconstruction steps | 17.36% |
| DBN + 10 top-down reconstruction steps | 19.39% |

Accuracy of the DBN with 10 reconstruction steps as attack strength grows:

| ε | 0 | 0.05 | 0.10 | 0.15 | 0.20 | 0.25 | 0.30 |
|---|---|---|---|---|---|---|---|
| Accuracy | 79.03% | 68.87% | 43.22% | 27.66% | 19.39% | 14.15% | 11.08% |

### Main findings

- **Hierarchy:** deeper layers learn more abstract features and need fewer units. Reducing the size of the last layer reduced the number of "dead" units (receptive fields that learned nothing).
- **Clustering:** the dendrogram shows three groups that match human intuition: upper-body clothing (pullover, coat, shirt, t-shirt/top, plus bag), footwear (sandal, sneaker, ankle boot), and trouser with dress.
- **Noise:** the DBN is much more robust to Gaussian noise than the FFNN, and the third layer is the most stable.
- **Adversarial attacks:** both models are highly vulnerable to FGSM. Top-down reconstruction helps, and more steps help more, but the results are still far from satisfactory. Complementary defences such as adversarial training or data augmentation would be needed.

---

## How to run

The notebook was developed on **Google Colab with a GPU** (it falls back to CPU automatically, but training is much slower).

1. Open the notebook in Colab (or Jupyter).
2. Run all cells in order. The first cells download `DBN.py` and `RBM.py` with `wget`, and the dataset with `kagglehub`, which may ask for Kaggle credentials.
3. Expect a long runtime: 80 epochs × 3 RBMs, then up to 3000 epochs for each read-out and for the FFNN. The attack sweeps loop over 10,000 images one at a time.

### Requirements

- Python 3
- `torch`, `torchvision`
- `numpy`, `pandas`, `scipy`, `scikit-learn`
- `matplotlib`, `tqdm`
- `kagglehub`

```bash
pip install torch torchvision numpy pandas scipy scikit-learn matplotlib tqdm kagglehub
```

Outside Colab, replace the `!wget` calls with a manual download of the two files (or clone the DBN repository).

## Notes and limitations

- Results come from a single training run; no seeds are fixed for the DBN, the noise or the FFNN, so numbers will vary slightly between runs.
- The FFNN is trained without early stopping, while the DBN read-outs use it, so the FFNN's clean accuracy comes from the final epoch.
- The attack is a single-step FGSM. Stronger attacks (e.g. PGD) would likely reduce robustness further.

## References

- Hinton, G., & Salakhutdinov, R. (2006). [Reducing the Dimensionality of Data with Neural Networks](https://www.science.org/doi/10.1126/science.1127647). *Science*.
- Hinton, G. (2010). [A Practical Guide to Training Restricted Boltzmann Machines](https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf).
- Testolin, A., et al. (2013). [Deep unsupervised learning on a desktop PC: a primer for cognitive scientists](https://www.frontiersin.org/articles/10.3389/fpsyg.2013.00251/full). *Frontiers in Psychology*.
- Xiao, H., Rasul, K., & Vollgraf, R. (2017). Fashion-MNIST: a novel image dataset for benchmarking machine learning algorithms. [Kaggle dataset](https://www.kaggle.com/datasets/zalando-research/fashionmnist).
