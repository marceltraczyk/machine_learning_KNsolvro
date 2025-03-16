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
        lambda ingredients: ", ".join([ingredient["name"] for ingredient in ingredients])
        if isinstance(ingredients, list) else ""
    )
    return df

# creating empty list instead of NaN values in tags
def handle_NaN_values(df):
    df["tags"] = df["tags"].apply(lambda x: x if isinstance(x, list) else [])
    return df

def extract_tags(df):
    df["tags"] = df["tags"].apply(
        lambda tags: ", ".join(tags) if isinstance(tags, list) else ""
    )
    return df


df = remove_unnecessary_columns(df)
df = handle_NaN_values(df)
df = extract_ingredient_names(df)
df = extract_tags(df)

print(df)

def save_data(df, output_path):
    df.to_csv(output_path, index=False)

save_data(df, "data/processed_data.csv")


