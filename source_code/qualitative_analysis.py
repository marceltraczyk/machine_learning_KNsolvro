import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv(r"C:\Users\marce\Desktop\machine_learning\data\processed_data.csv")

df["ingredients"] = df["ingredients"].apply(lambda x: x.split(", "))

mlb = MultiLabelBinarizer()
ingredient_matrix = mlb.fit_transform(df["ingredients"])

y = df["glass"]

X_train, X_test, y_train, y_test = train_test_split(
    ingredient_matrix, y, test_size=0.25, random_state=42
)

log_model = LogisticRegression(max_iter=1000, random_state=42)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)
accuracy_log = accuracy_score(y_test, y_pred_log)

rf_model = RandomForestClassifier(n_estimators=1000, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)