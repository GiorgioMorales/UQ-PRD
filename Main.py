from PREnsemble.Data.DataLoader import DataLoader
from PREnsemble.EvaluateModel import EvaluateModel

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

if __name__ == '__main__':
    data = DataLoader(name='grT')
    emodel = EvaluateModel(dataset=data.dataset, complexity=2)
    emodel.eval_ensemble(ensemble_size=30)
