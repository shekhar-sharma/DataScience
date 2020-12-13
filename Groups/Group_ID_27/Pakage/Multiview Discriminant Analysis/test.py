import mvda
import numpy as np
import torch
from sklearn.datasets import make_blobs


def generate_random_multiview_dataset(seed=138):
    np.random.seed(seed)
    X_veiw1, y = make_blobs(n_samples=10, n_features=3, centers=3)
    X_veiw2 = np.array([x + np.random.rand(len(x.shape)) * 3 for x in (X_veiw1 + np.random.randn(3) * 7)])
    X_veiw3 = np.array([x + np.random.rand(len(x.shape)) * 3 for x in (X_veiw1 + np.random.randn(3) * -7)])
    

    return [torch.tensor(X_veiw1).float(), torch.tensor(X_veiw2).float(), torch.tensor(X_veiw3).float()], y

if __name__ == '__main__':
    def main():

        X,Y = generate_random_multiview_dataset()
        
        MvDA = mvda.MVDA()
        
        vTransforms = MvDA.fit_transform(X,Y)
        
        print(len(vTransforms))
        print()
        print(vTransforms)

    main()