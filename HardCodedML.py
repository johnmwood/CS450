
class HardCodedModel:

    def predict(self, data):
        return [ 0 for _ in data ]


class HardCodedClassifier:
    def __init__(self):
        self.model = HardCodedModel()

    def fit(self, x_train, y_train):
        return self.model
