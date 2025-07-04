import glob
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib


# 1) CSV einlesen und DataFrame-Liste + Labels sammeln
def load_data(pattern="recordings/*.csv", delimiter=';'):
    dfs, labels = [], []
    for fn in glob.glob(pattern):
        df = pd.read_csv(fn, delimiter=delimiter)
        dfs.append(df)
        labels.append(df['spellName'].iloc[0])
    return dfs, np.array(labels)

# 2) Outlier entfernen mittels Z-Score (|z| > 3 → drop)
def remove_outliers(df, cols=['accX','accY','accZ','gyroX','gyroY','gyroZ']):
    zs = np.abs(stats.zscore(df[cols]))
    mask = (zs < 3).all(axis=1)
    return df[mask]

# 3) Lineare Interpolation fehlender Werte
def interpolate_df(df):
    return df.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')

# 4) Features extrahieren
def extract_features(df):
    cols = ['accX','accY','accZ','gyroX','gyroY','gyroZ']
    X = df[cols]
    feats = []
    feats += list(X.mean(axis=0))
    feats += list(X.std(axis=0))
    feats += list(X.max(axis=0))
    feats += list(X.min(axis=0))
    return np.array(feats)

# Komplett-Pipeline: von rohen DataFrames zu Feature-Matrix
def build_feature_matrix(dfs):
    X_list = []
    for df in dfs:
        # Outlier raus
        df_clean = remove_outliers(df)
        # Interpolieren
        df_interp = interpolate_df(df_clean)
        # Feature-Vector
        feats = extract_features(df_interp)
        X_list.append(feats)
    return np.vstack(X_list)

if __name__ == "__main__":
    # Lade
    dfs, y = load_data()
    # Feature-Matrix
    X = build_feature_matrix(dfs)

    # 5) Normalisierung
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # 6) Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_s, y_train)

    # 7) Evaluation
    print("=== Classification Report ===")
    print(classification_report(y_test, rf.predict(X_test_s)))
    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, rf.predict(X_test_s)))
    print("=== 5-Fold CV Accuracy ===")
    print(cross_val_score(rf, scaler.transform(X), y, cv=5).mean())

    # Speichern für späteren Einsatz im Duel-Client
    joblib.dump(scaler, "../python-client-wandduel/scaler.pkl")
    joblib.dump(rf, "../python-client-wandduel/gesture_rf.pkl")
