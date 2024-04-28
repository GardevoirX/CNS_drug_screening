# Central Nervous System (CNS) Drug Development: Drug Screening and Optimization
This repo holds the code of the competition "Central Nervous System (CNS) drug development: drug screening and optimization" and serves as a semester project of "AI for Chemistry" (CH-457).

## Quickstart
### Requirements
- Python 3.11
- Conda environment (Recommended)
- CUDA 12.1 (Recommended, for PyTorch)

```shell
git clone https://github.com/GardevoirX/CNS_drug_screening.git
cd CNS_drug_screening
pip install --upgrade pip
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
pip install -r requirements.txt
```

### Test
``` shell
cd CNS_drug_screening
pytest
```

## Introduction
The treatment of central nervous system (CNS) diseases is very tricky due to the existence of the blood-brain barrier (BBB). The BBB is a highly selective barrier between the circulatory system and the CNS, which protects the brain from harmful substances in the blood while also keeping the drugs against CNS diseases from the focus of infection. Near 98% of small molecular drugs and almost all macromolecular drugs cannot pass that barrier.

Quantitative structure-activity relationship (QSAR) is a model that relates a series of molecular properties (X, descriptors) to the activities of the molecular (Y, labels). Hansch and Fujita first proposed a linear model between molar concentrations, Hammett constants and the partition coefficients:

$$\log(1/C) = k_1 \pi + k_2 \sigma + k_3$$

where $\pi$ stands for the partition constant, $\sigma$ stands for the Hammett constant, and $k_1$, $k_2$ and $k_3$ are obtained via the least squares. ([Hansch, 1962](https://doi.org/10.1021/ar50020a002))

This model can be further generalized as:

$$Activity = f(property_1, propterty_2, ...) + error$$

the $f$ here can be either a linear model or a very complex neural network.

## Dataset
The dataset is organized as:

| SMILES             | Target |
|--------------------|--------|
| CC(=O)Nc1ccc(cc1)O | 1      |
| CC1OC1P(=O)(O)O    | 0      |
| ...                | ...    |

Here 1 stands for the CNS drugs and 0 stands for non-CNS drugs. There are a total of 701 data in the training set and 368 data in the test set.

## Methods

### Descriptors
Descriptors are mainly calculated with the help of the [descriptor module of RDKit](https://www.rdkit.org/docs/source/rdkit.Chem.Descriptors.html).

### Models
Models can be simple models provided by [scikit-learn](https://scikit-learn.org/stable/) or complex models build by [PyTorch](https://pytorch.org/)
