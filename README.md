
# Towards Uncertainty Quantification in Generative Model Learning

## Description

While generative models have become increasingly prevalent across various domains, fundamental concerns regarding their reliability persist. A crucial yet understudied aspect of these models is the uncertainty quantification surrounding their distribution approximation capabilities. Current evaluation methodologies focus predominantly on measuring the closeness between the learned and the target distributions, neglecting the inherent uncertainty in these measurements. In this position paper, we formalize the problem of uncertainty quantification in generative model learning. We discuss potential research directions, including the use of ensemble-based precision-recall curves. Our preliminary experiments on synthetic datasets demonstrate the effectiveness of aggregated precision-recall curves in capturing model approximation uncertainty, enabling systematic comparison among different model architectures based on their uncertainty characteristics.

## Usage

The Python file used to reproduce the plots reported in our paper is `Main.py`.

***Parameters***:

*   `grT`: Truncated Gaussian ring dataset. See [here](https://github.com/GiorgioMorales/UQ-PRD/blob/main/src/PREnsemble/Data/DataLoader.py#L49) for more details. 
*   `complexity`: Model complexity. Options: 1, 2, 4, and 8.

```python
from PREnsemble.Data.DataLoader import DataLoader
from PREnsemble.EvaluateModel import EvaluateModel

if __name__ == '__main__':
    data = DataLoader(name='grT')
    emodel = EvaluateModel(dataset=data.dataset, complexity=4)
    emodel.eval_ensemble(ensemble_size=30)
```

To retrain the diffusion models, use the `src/TrainModel.py` file. 

***Parameters***:

*   `grT`: Truncated Gaussian ring dataset. See [here](https://github.com/GiorgioMorales/UQ-PRD/blob/main/src/PREnsemble/Data/DataLoader.py#L49) for more details. 
*   `complexity`: Model complexity. Options: 1, 2, 4, and 8.
*   `n`: Dataset size. Options: 2500, 5000, 7500, and 10000.
*   `plotR`: If True, plot generated vs. real samples at the end of each model's training.

```python
from PREnsemble.Data.DataLoader import DataLoader
from PREnsemble.EvaluateModel2 import TrainModel

if __name__ == '__main__':
    c = 4
    print("Training ensemble of models of complexity ", c)
    data = DataLoader(name='grT', n=2500)
    model = TrainModel(dataset=data.dataset, complexity=c, plotR=False, suffix=str(len(data.X)))
    model.train_ensemble(ensemble_size=30)
```

# Citation
Use this Bibtex to cite this repository

```
@inproceedings{morales2025towards,
title={Towards Uncertainty Quantification in Generative  Model Learning},
author={Giorgio Morales, Frederic Jurie, Jalal Fadili},
booktitle={EurIPS 2025 Workshop: Epistemic Intelligence in Machine Learning},
year={2025},
url={https://openreview.net/forum?id=qTbJ1SiFrh}
}
```
