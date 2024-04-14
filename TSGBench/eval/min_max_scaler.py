# min_max_scaler.py
class MinMaxScaler():
    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def fit(self, data):
        self.mini = np.min(data, 0)
        self.range = np.max(data, 0) - self.mini
        return self

    def transform(self, data):
        return (data - self.mini) / (self.range + 1e-7)

    def inverse_transform(self, data):
        return data * self.range + self.mini
