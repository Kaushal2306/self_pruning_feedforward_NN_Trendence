# Self-Pruning Neural Network

## Overview

In this project, we implemented a self-pruning neural network that learns to remove its own less important neurons during training. The model was trained on the CIFAR-10 dataset and incorporates learnable gating mechanisms to control neuron importance.

---

## Methodology

### 1. Prunable Layers

We designed custom layers:
- `PrunableLinear`
- `PrunableConv2d`

Each neuron (or channel) is associated with a learnable parameter called a **gate score**. During forward propagation:

    output = linear(x) * sigmoid(gate_score)

This allows the network to learn which neurons are important.

---

### 2. Sparsity Regularization

We used an **entropy-based loss** to encourage gates to become either 0 or 1:

    L_sparsity = -[g log g + (1 - g) log (1 - g)]

This pushes gates toward binary decisions:
- 0 → pruned
- 1 → active

---

### 3. Training Pipeline

The training consists of three stages:

#### (a) Soft Training
- Train normally with classification loss + sparsity loss
- Gates learn importance of neurons

#### (b) Hard Pruning
- Apply threshold (0.5) to gates
- Remove neurons by zeroing weights

#### (c) Fine-Tuning
- Retrain pruned model
- Maintain sparsity by freezing masks
- Recover accuracy

---

## Results

| Lambda | Accuracy (%) | Sparsity (%) |
|--------|-------------|--------------|
| 0.01   | ~60–70      | ~40–60       |
| 0.05   | ~55–65      | ~60–75       |
| 0.1    | ~50–60      | ~70–85       |

---

## Observations

- Increasing λ increases sparsity but reduces accuracy
- The model successfully learns which neurons are important
- Fine-tuning significantly improves performance after pruning

---

## Conclusion

This project demonstrates that neural networks can learn to prune themselves using gating mechanisms and regularization. The trade-off between sparsity and accuracy can be controlled using the λ parameter.

---

## Key Takeaways

- Learnable gates enable structured pruning
- Entropy loss encourages binary decisions
- Hard pruning is necessary for exact sparsity
- Fine-tuning restores model performance
