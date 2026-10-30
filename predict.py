import pandas as pd, joblib, sys
MODEL_PATH = 'models/modele_attaque.pkl'
def predict_flux(csv_path):
    clf = joblib.load(MODEL_PATH)
    df = pd.read_csv(csv_path)
    X = df.drop(columns=['attaque'], errors='ignore') if 'attaque' in df.columns else df
    df['prediction'] = clf.predict(X)
    df['label'] = df['prediction'].map({0: 'normal', 1: 'attaque'})
    df.to_csv('predictions.csv', index=False)
    print(df['label'].value_counts())
if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'data/flux_train.csv'
    predict_flux(path)
