import numpy as np


class ParkinsonUPDRSInputs(object):
    def __init__(self, xp_dim=0, split_ratio=.8, batch_size=50):
        """Parkinson UPDRS dataset

        Parameters
        ----------
        xp_dims: int {0, 1}
            regression target for y
        split_ratio: float [0, 1] (default: .8)
            split ratio between train and test
        batch_size: int (default: 50)
            size of the batch returned by get_next_batch
        """
        self.xp_dim = xp_dim
        self.split_ratio = split_ratio
        self.batch_size = batch_size

        self._load_dataset()
        self.batch_index = 0

    def _load_dataset(self):
        print("Loading parkinson data set")
        f = np.loadtxt('parkinsons_updrs.data.txt', delimiter=",", skiprows=2
                       )[:, 3:]
        x_train = f[:, 2:]
        y_train = f[:, self.xp_dim:1 + self.xp_dim]

        # Normalize the data
        mx = np.mean(x_train, axis=0)
        x_train -= mx
        nx = np.sqrt(np.sum(x_train**2) / len(y_train))
        x_train /= nx

        my = np.mean(y_train, axis=0)
        y_train -= my
        ny = np.sqrt(np.sum(y_train**2) / len(y_train))
        y_train /= ny

        train = np.hstack([x_train, y_train])

        # Split in train/test
        self.N_train = int(len(train) * self.split_ratio)

        test = train[self.N_train:]
        self.train = train[:self.N_train]

        self.x_train = self.train[:, :-1]
        self.y_train = self.train[:, -1:]
        self.x_test = test[:, :-1]
        self.y_test = test[:, -1:]

        self.n_dim = self.x_train.shape[1]

    def get_next_batch(self):
        """Return the current batch and shuffle training data on epochs
        """
        self.batch_index += 1
        if self.batch_index * self.batch_size > len(self.x_train):
            self.batch_index = 1
            np.random.shuffle(self.train)
            self.x_train = self.train[:, :-1]
            self.y_train = self.train[:, -1:]

        # Return the current training batch
        i_start = self.batch_size * (self.batch_index - 1)
        i_end = i_start + self.batch_size

        return self.x_train[i_start:i_end], self.y_train[i_start:i_end]

    def get_train_set(self):
        return self.x_train, self.y_train

    def get_test_set(self):
        return self.x_test, self.y_test
