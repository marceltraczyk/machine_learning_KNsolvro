import pandas as pd

def load_data(filepath):
    return pd.read_csv(filepath)

def count_unique_ingredients(df):
    unique_ingredients = set()
    for ingredient in df["ingredients"]:
        unique_ingredients.update(ingredient)
    
    return len(unique_ingredients)

def most_common_ingredients(df):
    ingredient_counts = {}
    for ingredients in df["ingredients"]:
        for ingredient in ingredients.strip("[]").replace("'", "").split(", "):
            ingredient_counts[ingredient] = ingredient_counts.get(ingredient, 0) + 1
    return sorted(ingredient_counts.items(), key=lambda x: x[1], reverse=True)[:10]

def count_unique_glass_types(df):
    return df["glass"].nunique()

def most_common_glass_types(df, top_n=10):
    return df["glass"].value_counts().head(top_n)

def count_alcoholic(df):
    return df["alcoholic"].sum()

def count_unique_categories(df):
    return df["category"].nunique()

def most_common_categories(df, top_n=10):
    return df["category"].value_counts().head(top_n)


file_path = "data/processed_data.csv"
df = load_data(file_path)

print(f"\nLiczba unikalnych składników: {count_unique_ingredients(df)}")
print("\n10 najczęściej występujących składników:")
for ing, count in most_common_ingredients(df):
    print(f"{ing}: {count}")

print(f"\nLiczba unikalnych typów szkła: {count_unique_glass_types(df)}")
print("\nNajczęściej używane typy szkła:")
print(most_common_glass_types(df))

print(f"\nLiczba unikalnych kategorii: {count_unique_categories(df)}")
print("\nNajczęściej wystepujące kategorie:")
print(most_common_categories(df))

print("\nLiczba koktajli alkoholowych")
print(count_alcoholic(df))