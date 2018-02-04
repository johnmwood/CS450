import numpy as np
import pandas as pd
import pprint


class ID3Model:
    def __init__(self, data_train, targets_train):
        # self.data_train = data_train
        # self.targets_train = targets_train

        if not isinstance(data_train, pd.DataFrame):
            data_train = pd.DataFrame(data_train)
        if not isinstance(targets_train, pd.DataFrame):
            targets_train = pd.DataFrame(targets_train, columns=['targets'])

        self.dataset = pd.concat([data_train, targets_train], axis=1)

        feature_names = self.dataset.drop(["targets"], axis=1).columns.values
        self.tree = self.create_tree(self.dataset, feature_names)

    def show_tree(self):
        pass

    def create_tree(self, df, features):
        """
        if no features left to test
            return leaf with most common label
        else if examples have the same label
            return leaf with label
        else
            consider each availabe feature
            choose the one that maximizes information gain
            create a new node for that feature

            For each possible value of the feature
                create branch for this value
                create a sebset of the examples for each branch
                recursively call the function to create a new node at that branch
        """
        data = df.copy()
        values = data[features]
        default = data["targets"].value_counts().index.tolist()[0]
        print(f"default: {default}")

        print(f"Starting dataframe columns: {data.columns}")
        print(f"values: {values}")
        print(f"default: {default}")

        if len(features) == 0:
            return default
        elif len(set(data["targets"])) == 1:
            return data["targets"].iloc[0]
        else:
            # calculate gain of all features
            print("else")
            gain_dict = {}
            for feature in features:
                print(f"Feature: {feature}")
                print(f"This is the data: {data}")
                gain_dict[feature] = self.calc_info_gain(feature, data)
            print("past for-loop")
            best_feature = max(gain_dict, key=gain_dict.get)

            tree = { best_feature: {} }
            print("after dictionary")
            for value in np.unique(data[best_feature]):
                new_data = data[data[best_feature] == value].drop([best_feature], axis=1)
                print(f"Best feature: {best_feature}\'s value: {value}")
                subtree = self.create_tree(new_data, new_data.drop(["targets"], axis=1).columns)
                tree[best_feature][value] = subtree

            return tree


    def calc_info_gain(self, feature, df):
        # create one df for every column in self.dataset, grouped by column and targets
        grouped_data = df.groupby([feature, "targets"])\
                         .size().reset_index().rename(columns={0:"count"})

        # Calc total target count for feature
        feature_count = grouped_data["count"].sum()
        entropy_list = []

        # Calculate the entropy of every value in feature
        unique_values = np.unique(grouped_data[feature])
        # get a list of unique target values
        unique_targets = np.unique(df["targets"])
        for value in unique_values:
            # Subset values
            data = grouped_data[grouped_data[feature] == value]
            # Calc total target count for value
            value_count = data["count"].sum()

            entropy_values = []
            for target in unique_targets:
                prob = data[data["targets"] == target]["count"].sum() / value_count
                if not pd.isnull(prob):
                    entropy_values.append(self.calc_entropy(prob))

            entropy_list.append(np.sum(entropy_values) * (value_count / feature_count))

        row_count = df.shape[0]
        df_value_counts = df["targets"].value_counts()

        feature_entropy = np.sum(entropy_list)
        total_entropy = np.sum([self.calc_entropy(df_value_counts[target] / row_count)
                                                  for target in unique_targets])

        return total_entropy - feature_entropy

    def calc_entropy(self, p):
        """Algorithm calculating entropy from Machine Learning, an Algorithmic Perspective
        credit for function: Stephen Marsland
        """
        if p != 0:
            return -p * np.log2(p)
        else:
            return 0

    def findPath(graph, start, end, pathSoFar):
        """Algorithm calculating entropy from Machine Learning, an Algorithmic Perspective
        credit for function: Stephen Marsland
        """
        pathSoFar = pathSoFar + [start]
        if start == end:
            return pathSoFar
        if start not in graph:
            return None
        for node in graph[start]:
            if node not in pathSoFar:
                newpath = findPath(graph, node, end, pathSoFar)
                return newpath
        return None

    def predict(self):
        pass


class ID3DecisionTree:
    def __init__(self):
        pass

    def fit(self, data_train, targets_train):
        return ID3Model(data_train, targets_train)


############# TEST ###############

data = {"Weather": ["Hot", "Cold", "Nice", "Cold", "Hot", "Hot", "Cold", "Nice", "Nice", "Hot", "Nice"],
         "Test": ["Pass", "Pass", "Fail", "Pass", "Fail", "Fail", "Pass", "Fail", "Pass", "Pass", "Pass" ],
         "Chocolate": ["Dark", "Dark", "Dark", "Dark", "Dark", "Dark", "Dark", "Dark", "Dark", "Dark", "Dark"],
         "Cholo": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1] }
targets = [1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0]

df_data = pd.DataFrame(data)

tree = ID3Model(df_data, targets)
pp = pprint.PrettyPrinter()
pp.pprint(tree.tree)

# print(tree.calc_entropy(2/3) + tree.calc_entropy(1/3))
