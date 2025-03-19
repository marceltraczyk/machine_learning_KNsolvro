import pandas as pd

#converting json to data tables
df = pd.read_json("data/cocktail_dataset.json") 

# removing columns that i wont be used in data understanding
def remove_unnecessary_columns(df):
    columns_to_drop = ["imageUrl", "createdAt", "updatedAt", "instructions"]
    df = df.drop(columns=columns_to_drop, errors="ignore")
    return df

# take only names of ingredients
def extract_ingredient_names(df):
    df["ingredients"] = df["ingredients"].apply(
        lambda ingredients: ", ".join([ingredient["name"] for ingredient in ingredients])
    )
    return df

# creating empty list instead of NaN values in tags
def handle_NaN_values(df):
    df["tags"] = df["tags"].apply(lambda x: ", ".join(x if isinstance(x, list) else []))
    return df

# preprocessing the data
df = remove_unnecessary_columns(df)
df = handle_NaN_values(df)
df = extract_ingredient_names(df)

print(df)

# saving to a separate csv file in the data folder
def save_data(df, output_path):
    df.to_csv(output_path, index=False)

save_data(df, "data/processed_data.csv")


