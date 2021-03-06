import pandas as pd


# read data in
adult_df = pd.read_csv('adult.csv', index_col=None, na_values=" ?")

column_names = ["age", "workclass", "fmlwgt", "education", "education-num",
                "marital_status", "occupation", "relationship", "race", "sex",
                "capital_gain", "capital_loss", "hours_per_week", "native_country",
                "income"]

adult_df.columns = column_names
categorical_data = adult_df.select_dtypes(include=['object']).copy()

# handle categorical data
for category in categorical_data.columns:
    categorical_data[category] = categorical_data[category].astype('category')
    categorical_data[category + "_code"] = categorical_data[category].cat.codes

adult_data = pd.merge(adult_df, categorical_data)

for column in adult_data.columns:
    print(adult_data[column].value_counts())
