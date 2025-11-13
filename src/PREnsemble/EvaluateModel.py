import os
from PREnsemble.utils import *
from PREnsemble.Models.DDPM import *
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from scipy.interpolate import interp1d
from PREnsemble.PRCurves.getPRCurves import getPRCurves
from Data.DataLoader import DataLoader, InputData

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


class MyDataset(Dataset):
    def __init__(self, x):
        self.x = x

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx]


class EvaluateModel:
    def __init__(self, dataset=InputData, modelType='DDPM', complexity=1, suffix=None):
        self.modelType = modelType
        self.complexity = complexity
        self.dataset = dataset
        self.X = MyDataset(self.dataset.X)
        self.suffix = ''
        if suffix is not None:
            self.suffix = '_' + suffix

    def eval_ensemble(self, ensemble_size=10):
        # If the folder does not exist, create it
        root = get_project_root()
        folder = os.path.join(root,
                              "Models//saved_models//" + self.dataset.name + '//ModelType-' + str(self.complexity))
        if not os.path.exists(os.path.join(root, "Models//saved_models//" + self.dataset.name)):
            os.mkdir(os.path.join(root, "Models//saved_models//" + self.dataset.name))
        if not os.path.exists(folder):
            os.mkdir(folder)

        # Define path where the model will be saved
        total_curves = []
        for mi in range(ensemble_size):
            print("Model ", mi)
            filepath = folder + "//Model-" + self.dataset.name
            filepath = [filepath] * ensemble_size
            filepath[mi] = filepath[mi] + "-Instance" + str(mi) + self.suffix + '.pth'
            # Load the model
            model = torch.load(filepath[mi])
            # Compute PR curve
            num_samples = self.X.x.shape[0]
            feats_gen = []
            batch_size, total_size = 1000, 0
            while total_size < num_samples:
                num_gen = num_samples - len(feats_gen)
                for start_idx in range(0, num_gen, batch_size):
                    end_idx = min(start_idx + batch_size, num_gen)
                    batch_shape = (end_idx - start_idx, *self.X.x.shape[1:])  # Maintain shape consistency
                    x_seq = denoise(model, batch_shape)
                    feats_gen_batch = x_seq[-1].detach().numpy()
                    feats_gen_batch = feats_gen_batch[np.unique(np.where(~np.isnan(feats_gen_batch))[0]), :]
                    feats_gen.append(feats_gen_batch)
                total_size = len(np.vstack(feats_gen))
            # Concatenate all batches
            feats_gen = np.vstack(feats_gen)
            feats_real = self.dataset.Xval.cpu().detach().numpy()[: len(feats_gen), :]
            prd_data, slopes = getPRCurves(methods=['knn'],
                                           real_data=feats_real,
                                           fake_data=feats_gen,
                                           num_samples=len(feats_real),
                                           split_train_test_ratio=0.5,
                                           number_angles=200,
                                           nearest_k=int(np.sqrt(len(feats_real))),
                                           output_folder='Results//' + "Curves-Dataset_" + self.dataset.name +
                                                         "-Complexity_" + str(self.complexity) + "-Iteration_" + str(mi),
                                           device='cpu',
                                           c_dist_approx=True,
                                           )
            total_curves.append((prd_data['knn'][:, 0], prd_data['knn'][:, 1]))
            # plt.figure(figsize=(4, 4))
            # plt.scatter(feats_gen[:, 0], feats_gen[:, 1], alpha=0.2, s=10, label='Generated samples')
            # plt.scatter(feats_real[:, 0], feats_real[:, 1], alpha=0.1, s=10, label='Real samples')
            # plt.tight_layout()
            # plt.legend()
            # plt.xlim(-6.5, 6.5)
            # plt.ylim(-6.5, 6.5)
            # plt.show()

        # Define a common recall grid
        num_points = 1001
        common_recall = np.linspace(0, 1, num_points)[1:]
        # Interpolate all precision curves to the common recall grid
        interpolated_precisions = []
        for precision, recall in total_curves:
            recall = np.append(recall, 1)
            precision = np.append(precision, 0)
            interp_func = interp1d(recall, precision, kind='linear', bounds_error=False, fill_value=0)
            interpolated_precisions.append(interp_func(common_recall))

        interpolated_precisions = np.array(interpolated_precisions)
        mean_precision = np.mean(interpolated_precisions, axis=0)

        # Compute the 10th and 90th percentiles
        lower_bound = np.percentile(interpolated_precisions, 10, axis=0)
        upper_bound = np.percentile(interpolated_precisions, 90, axis=0)

        # Plot the aggregated PR curves
        plt.figure()
        plt.plot(common_recall, mean_precision, color='blue', label="Mean PR Curve", linewidth=3)
        plt.fill_between(common_recall, lower_bound, upper_bound, color='blue', alpha=0.2,
                         label="80% Confidence Interval")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Aggregated Precision-Recall Curves")
        plt.legend()
        plt.grid(True)
        plt.show()
        # suffix = "-Dataset_" + self.dataset.name + "-Complexity_" + str(self.complexity)
        # plt.savefig("Results//precision_recall_curve" + suffix + ".jpg", dpi=600)
        # # Save mean and standard deviation curves as npy files
        # np.save("Results//mean_precision" + suffix + ".npy", mean_precision)
        # np.save("Results//std_precision" + suffix + ".npy", std_precision)
        # np.save("Results//recalls" + suffix + ".npy", common_recall)


if __name__ == '__main__':
    data = DataLoader(name='grT', n=7500)
    emodel = EvaluateModel(dataset=data.dataset, complexity=4, suffix=str(len(data.X)))
    emodel.eval_ensemble(ensemble_size=30)
