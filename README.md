# Sequence-to-Function Mapping: 1D CNN for Synthetic Motif Detection 🧬🤖

An end-to-end PyTorch deep learning pipeline designed to automatically discover regulatory motifs in synthetic DNA sequences. 

This project bridges scalable software engineering with computational biology, demonstrating how 1D Convolutional Neural Networks (CNNs) can mimic how transcription factors physically scan DNA strands for binding sites.

## 📖 Project Overview

In regulatory genomics, the exact sequence of DNA nucleotides (A, C, G, T) dictates cellular function. Proteins known as transcription factors regulate gene expression by binding to specific, short geometric character patterns called **motifs** (e.g., the TATA box).

Mathematically, identifying these motifs within massive genomic datasets is a sequence classification problem. This project implements a production-ready PyTorch pipeline that:
1. **Generates a synthetic dataset** of one-hot encoded DNA sequences.
2. **Injects a known biological pattern** (the universal `TATAAAA` motif) into positive samples, hidden within random background sequence noise.
3. **Trains a custom 1D CNN architecture** to automatically discover this spatial pattern and classify sequences as "functional" (1) or "non-functional" (0).

## 🏗️ Architecture & Engineering Highlights

Unlike standard analytical scripts, this project is structured as a modular, reproducible machine learning system:

* **Vectorized Data Transformation:** Utilizes custom PyTorch `Dataset` classes and NumPy vectorization for rapid, memory-efficient one-hot encoding of sequence data.
* **Biologically-Inspired ML Architecture:** * `Conv1d` (`kernel_size=8`): Acts as the motif detector, sliding across the sequence exactly as a transcription factor would.
  * `MaxPool1d`: Provides spatial invariance, allowing the model to recognize the motif regardless of its exact base-pair location.
* **Hardware-Agnostic Training:** The training loop automatically detects and utilizes Apple Silicon (`mps`), NVIDIA GPUs (`cuda`), or falls back to `cpu`.

## 📂 Repository Structure

```text
├── dataset.py      # PyTorch Dataset class, synthetic generation, and OHE logic
├── model.py        # 1D CNN neural network architecture (PyTorch nn.Module)
├── train.py        # Optimization loop, train/val split, and backpropagation
├── demo.ipynb      # Interactive Jupyter Notebook with data visualizations
└── README.md       # Project documentation
```

## Requirements

The core dependencies for this project are:
- `torch`
- `numpy`
- `matplotlib`
Ensure you have Python 3.8+ installed along with PyTorch and Matplotlib.
```bash
pip install torch torchvision torchaudio matplotlib numpy`
```
## Usage

You can train the model directly from the terminal by running `train.py`:

```bash
python train.py
```

This will instantiate the `SyntheticDNADataset`, configure the `DNAMotifCNN`, and run the training loop for 40 epochs. It will automatically utilize hardware acceleration (CUDA or MPS) if available.

Alternatively, you can run all the cells in the `demo.ipynb` notebook to train the model and visualize the results.

### Dataset Testing

You can run a quick unit test on the dataset generation logic by executing:

```bash
python dataset.py
```

### Model Architecture Testing

You can verify the CNN architecture handles input and output shapes correctly by executing:

```bash
python model.py
```
