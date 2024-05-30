
# Central Nervous System (CNS) Drug Development: Drug Screening and Optimization
[![Powered by RDKit](https://img.shields.io/badge/Powered%20by-RDKit-3838ff.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQBAMAAADt3eJSAAAABGdBTUEAALGPC/xhBQAAACBjSFJNAAB6JgAAgIQAAPoAAACA6AAAdTAAAOpgAAA6mAAAF3CculE8AAAAFVBMVEXc3NwUFP8UPP9kZP+MjP+0tP////9ZXZotAAAAAXRSTlMAQObYZgAAAAFiS0dEBmFmuH0AAAAHdElNRQfmAwsPGi+MyC9RAAAAQElEQVQI12NgQABGQUEBMENISUkRLKBsbGwEEhIyBgJFsICLC0iIUdnExcUZwnANQWfApKCK4doRBsKtQFgKAQC5Ww1JEHSEkAAAACV0RVh0ZGF0ZTpjcmVhdGUAMjAyMi0wMy0xMVQxNToyNjo0NyswMDowMDzr2J4AAAAldEVYdGRhdGU6bW9kaWZ5ADIwMjItMDMtMTFUMTU6MjY6NDcrMDA6MDBNtmAiAAAAAElFTkSuQmCC)](https://www.rdkit.org/)

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

### Inference
``` shell
python ./inference.py --data_file your_smiles.csv
```

### Test
``` shell
cd CNS_drug_screening
pytest
```

### Run
``` shell
python ./train.py
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

Here 1 stands for the CNS drugs and 0 stands for non-CNS drugs. There are a total of 701 data in the training set and 368 data in the test set. Below is the composition of the dataset: 453 non-CNS drugs and 247 CNS drugs.

![pie](https://github.com/GardevoirX/CNS_drug_screening/assets/92628709/71029b5a-a983-476c-9099-71ff6a933013)


## Methods

### Descriptors
Descriptors are mainly calculated with the help of the [descriptor module of RDKit](https://www.rdkit.org/docs/source/rdkit.Chem.Descriptors.html). Here we use a total of 14 descriptors, which can be further categorized into 6 types
| Type | Descriptor| # of features|
|:----:|:---------:|:------------:|
| Molecular characteristics | MW, abs. net charge, abs. max./min. partial charge,<br /> # of rotatable bonds, # of heavy atoms| 6 |
| Topological descriptors|[USR](https://www.rdkit.org/docs/source/rdkit.Chem.rdMolDescriptors.html#rdkit.Chem.rdMolDescriptors.GetUSR), [USR-CAT](https://www.rdkit.org/docs/source/rdkit.Chem.rdMolDescriptors.html#rdkit.Chem.rdMolDescriptors.GetUSRCAT), [2D autocorrelation](https://www.rdkit.org/docs/source/rdkit.Chem.rdMolDescriptors.html#rdkit.Chem.rdMolDescriptors.CalcAUTOCORR2D)| 164 |
| Quantum descriptor| [MQM](https://www.rdkit.org/docs/source/rdkit.Chem.rdMolDescriptors.html#rdkit.Chem.rdMolDescriptors.MQNs_) | 42 |
| Electronegative descriptor| [PEOE](https://www.rdkit.org/docs/source/rdkit.Chem.rdMolDescriptors.html#rdkit.Chem.rdMolDescriptors.PEOE_VSA_) | 10 |
| Partition coefficients| [VSA-logP](https://www.rdkit.org/docs/source/rdkit.Chem.rdMolDescriptors.html#rdkit.Chem.rdMolDescriptors.SlogP_VSA_)| 12 |
| Topological fingerprints| [topological torsion](https://www.rdkit.org/docs/source/rdkit.Chem.rdMolDescriptors.html#rdkit.Chem.rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect), [Morgan fingerprints](https://www.rdkit.org/docs/source/rdkit.Chem.rdMolDescriptors.html#rdkit.Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect)| 3072 (bits) |

In the real training process, some features are found to have only one value. These features are later removed leading to a total of 2912 features in the final scope.

### Models
Models can be simple models provided by [scikit-learn](https://scikit-learn.org/stable/) or complex models built by [PyTorch](https://pytorch.org/).

Our final model is a perceptron with five hidden layers. The number of neurons in each layer is 3076, 2048, 1024, 512 and 128, respectively. Layers are all equipped with LayerNorm, ReLU activation function and dropout. The dropout rate varies, and is 0.8, 0.6, 0.4, 0.4, 0.4 for each layer. Below is a schematic figure of our model.

![model](https://github.com/GardevoirX/CNS_drug_screening/assets/92628709/b8d97f84-2bee-4410-a762-cdec6cda68c8)


Our model finally achieved an F2 score of .838 in the online test provided by the [Bohrium platform](https://bohrium.dp.tech/competitions/9169114995?tab=leaderboard).

#### Performance of different models
Below is the performance of different models. Though the Bayesian regression performs the best in the validation set, it is far behind perceptrons in the test set, which might be explained by the stronger generalization ability led by the more complex model.

|Model | F2-score (validation)| F2-score (test)|
|------|----------------------|----------------|
|Logistic|0.747           |   |
|Linear|0.679             |   |
|Ridge |0.649             |   |
|Lasso |0.676             |   |
|ElasticNet|0.746         |   |
|Bayesian|**0.843**           |0.702   |
|SGD   |0.623             |   |
|Kernel|0.731             |   |
|SVC   |0.000             |   |
|KNN   |0.675             |   |
|KMeans|0.441             |   |
|GMM   |0.783             |   |
|3-Layer perceptron| 0.783|0.811   |
|**5-Layer perceptron**| 0.796|**0.838**   |

## References
1. https://bohrium.dp.tech/competitions/9169114995?tab=datasets (You can change the language in the menu hiding behind the up-right icon)
2. https://www.rdkit.org/docs/index.html
