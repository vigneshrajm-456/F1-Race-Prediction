
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import joblib

# Load cleaned data
df = pd.read_csv("data/f1_cleaned.csv")

# Use grid position and historical driver/team identities
features = ["year", "grid_position", "driver", "team"]
target = "winner"

X = df[features]
y = df[target]

# Split data: train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

numeric_features = ["year", "grid_position"]
categorical_features = ["driver", "team"]

preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

model = make_pipeline(
    preprocessor,
    RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
    )
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("MODEL TRAINED!")
print("Accuracy:", round(accuracy_score(y_test, predictions), 3))
print(classification_report(y_test, predictions, zero_division=0))

joblib.dump(model, "f1_winner_model.pkl")
print("Model saved: f1_winner_model.pkl")

