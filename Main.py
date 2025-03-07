import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from PREnsemble.Data.DataLoader import DataLoader
from PREnsemble.EvaluateModel2 import EvaluateModel


if __name__ == '__main__':
    data = DataLoader(name='grT')
    emodel = EvaluateModel(dataset=data.dataset, complexity=8)
    emodel.eval_ensemble(ensemble_size=30)
