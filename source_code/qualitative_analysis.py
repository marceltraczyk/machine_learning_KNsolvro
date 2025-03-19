import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Wczytanie danych
df = pd.read_csv(r"C:\Users\marce\Desktop\machine_learning\data\processed_data.csv")

# 2. Konwersja kolumny "ingredients" na listy
# Jeśli wartość w kolumnie "ingredients" jest stringiem, dzielimy go na listę składników
df["ingredients"] = df["ingredients"].apply(lambda x: x.split(", "))

# 3. One-Hot Encoding składników
# Każdy koktajl (wiersz) będzie reprezentowany jako wektor binarny,
# gdzie każda kolumna odpowiada unikalnemu składnikowi
mlb = MultiLabelBinarizer()
ingredient_matrix = mlb.fit_transform(df["ingredients"])

# 4. Ustalenie celu (target) – przewidujemy typ szkła
y = df["glass"]

# 5. Podział danych na zbiór treningowy i testowy (70% trening, 30% test)
X_train, X_test, y_train, y_test = train_test_split(ingredient_matrix, y, test_size=0.25, random_state=42)

# 6. Model regresji logistycznej
log_model = LogisticRegression(max_iter=1000, random_state=42)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)
accuracy_log = accuracy_score(y_test, y_pred_log)

# 7. Model Random Forest
rf_model = RandomForestClassifier(n_estimators=1000, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
