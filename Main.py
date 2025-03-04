from PREnsemble.Data.DataLoader import DataLoader
from PREnsemble.EvaluateModel import EvaluateModel

if __name__ == '__main__':
    data = DataLoader(name='grT')
    emodel = EvaluateModel(dataset=data.dataset, complexity=2)
    emodel.eval_ensemble(ensemble_size=30)
