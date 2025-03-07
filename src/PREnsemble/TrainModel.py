import os
from utils import *
from tqdm import trange
from Models.DDPM import *
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from Data.DataLoader import DataLoader, InputData


class MyDataset(Dataset):
    def __init__(self, x):
        self.x = x

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx]


class TrainModel:
    def __init__(self, dataset=InputData, modelType='DDPM', complexity=1, verbose=False, plotR=False):
        self.modelType = modelType
        self.complexity = complexity
        self.dataset = dataset
        self.X = MyDataset(self.dataset.X)
        self.model = self.reset_model()
        self.verbose = verbose
        self.plot = plotR

    def reset_model(self):
        h = 2
        return DenoisingNet(h, 128, complexity=self.complexity)

    def train(self, epochs=1001, filepath=''):
        self.model = self.reset_model()
        optimizer = optim.Adam(self.model.parameters(), lr=2e-3)
        loader = torch.utils.data.DataLoader(self.X, batch_size=500, shuffle=True)

        for t in trange(epochs):
            loss = None
            for index, x in enumerate(loader):
                noise_steps = (torch.randint(1, T - 1, (1,)) * torch.ones(x.shape[0])).long()

                noise = torch.randn_like(x)
                x_noisy = x * alphas_bar_sqrt[noise_steps].reshape(-1, 1) + noise * one_minus_alphas_bar_sqrt[
                    noise_steps].reshape(-1, 1)
                output = self.model(x_noisy, noise_steps)
                loss = (noise - output).square().mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
                optimizer.step()

            if t % 200 == 0 and self.verbose:
                print('epoch:', t, 'loss:', loss)
        torch.save(self.model, filepath)

    def train_ensemble(self, ensemble_size=10):
        # If the folder does not exist, create it
        root = get_project_root()
        folder = os.path.join(root,
                              "Models//saved_models//" + self.dataset.name + '//ModelType-' + str(self.complexity))
        if not os.path.exists(os.path.join(root, "Models//saved_models//" + self.dataset.name)):
            os.mkdir(os.path.join(root, "Models//saved_models//" + self.dataset.name))
        if not os.path.exists(folder):
            os.mkdir(folder)

        # Define path where the model will be saved
        feats_real = self.dataset.Xval.cpu().detach().numpy()
        epochs = 10001
        if self.complexity == 4:
            epochs = 15001
        elif self.complexity == 8:
            epochs = 20001

        for mi in range(ensemble_size):
            if mi == 21:
                filepath = folder + "//Model-" + self.dataset.name
                filepath = [filepath] * ensemble_size
                filepath[mi] = filepath[mi] + "-Instance" + str(mi) + '.pth'
                f = filepath[mi]
                # Train the model
                self.model = self.reset_model()
                print("\tTraining ", mi + 1, "/", ensemble_size, " model")
                self.train(epochs, f)
                # Plot results
                if self.plot:
                    x_seq = denoise(self.model, self.X.x.shape)
                    plt.figure()
                    cur_x = x_seq[-1].detach()
                    plt.scatter(feats_real[:, 0], feats_real[:, 1], s=2)
                    plt.scatter(cur_x[:, 0], cur_x[:, 1], s=2)
                    plt.show()


if __name__ == '__main__':
    complexities = [4]
    for c in complexities:
        print("Training ensemble of models of complexity ", c)
        data = DataLoader(name='grT')
        model = TrainModel(dataset=data.dataset, complexity=c, plotR=True)
        model.train_ensemble(ensemble_size=30)
