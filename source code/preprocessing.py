import pandas as pd

df = pd.read_json("data/cocktail_dataset.json") 
# print(df)
# print(df.dtypes)

# removing columns that i wont use in data understanding
def remove_unnecessary_columns(df):
    columns_to_drop = ["imageUrl", "createdAt", "updatedAt", "instructions"]
    df = df.drop(columns=columns_to_drop, errors="ignore")
    return df

# take only names of ingredients
def extract_ingredient_names(df):
    df["ingredients"] = df["ingredients"].apply(
        lambda ingredients: [ingredient["name"] for ingredient in ingredients]
        if isinstance(ingredients, list) else ingredients
    )
    return df

# creating empty list instead of NaN values
def handle_NaN_values(df):
    df["tags"] = df["tags"].apply(lambda x: x if isinstance(x, list) else [])
    return df

df = remove_unnecessary_columns(df)
df = handle_NaN_values(df)
df = extract_ingredient_names(df)

print(df)
''''
we can check for spellings and things like that

print("Kategorie:", df["category"].unique())
print("Rodzaje szkła:", df["glass"].unique())
'''
def save_data(df, output_path):
    df.to_csv(output_path, index=False)

save_data(df, "data/processed_data.csv")


