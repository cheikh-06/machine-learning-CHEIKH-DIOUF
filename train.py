import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib, json, os

df = pd.read_csv('data/flux_train.csv')
X = df.drop('attaque', axis=1)
y = df['attaque']
cat_features = ['protocol']
preprocessor = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)], remainder='passthrough')
clf = Pipeline([('preprocess', preprocessor), ('model', RandomForestClassifier(n_estimators=100, random_state=42))])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")
print(classification_report(y_test, y_pred))
os.makedirs('models', exist_ok=True)
joblib.dump(clf, 'models/modele_attaque.pkl')
with open('resultats_train.json', 'w') as f:
    json.dump({"accuracy": float(acc)}, f, indent=2)