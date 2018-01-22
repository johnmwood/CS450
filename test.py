import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import cross_val_score, cross_val_predict


iris = datasets.load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
print(iris_df.head())

test_size = 0.40

# Show the data (the attributes of each instance)
# print(iris.data)
#
# # Show the target values (in numeric format) of each instance
# print(iris.target)
#
# # Show the actual target names that correspond to each number
# print(iris.target_names)
#
# print(train_test_split(iris.data))
# data_train, data_test, targets_train, targets_test = train_test_split(iris.data, iris.target, test_size=test_size)
#
# print(data_train)
# print(data_test)
# print(targets_train)
# print(targets_test)
