# IMPORTANT: Please make all your changes to this file
import numpy as np
import pandas as pd
import joblib
from scipy import stats

# 1) Spell-Namen auf deine Gesten umstellen
spellname1 = "sues (Schere)"
spellname2 = "GEAR 5 (Papier)"
spellname3 = "Annaniiiii (Stein)"

LABEL_TO_ID = {
    "sues (Schere)": 1,
    "GEAR 5 (Papier)"  : 2,
    "Annaniiiii (Stein)"     : 3,
}

# 2) model + scaler laden
scaler = joblib.load("scaler.pkl")
clf    = joblib.load("gesture_rf.pkl")

def remove_outliers(df,
    cols=['accX','accY','accZ','gyroX','gyroY','gyroZ']):
    zs = np.abs(stats.zscore(df[cols]))
    mask = (zs < 3).all(axis=1)
    return df[mask]

def interpolate_df(df):
    return df.interpolate(method='linear') \
        .fillna(method='bfill') \
        .fillna(method='ffill')

def extract_features(df):
    cols = ['accX','accY','accZ','gyroX','gyroY','gyroZ']
    A = df[cols].to_numpy()
    feats = []
    feats += list(A.mean(axis=0))
    feats += list(A.std(axis=0))
    feats += list(A.max(axis=0))
    feats += list(A.min(axis=0))
    return np.array(feats)

def process_spell(pandas_df: pd.DataFrame):
    # 1) Outlier entfernen
    df_clean  = remove_outliers(pandas_df)
    # 2) Interpolation
    df_interp = interpolate_df(df_clean)
    # 3) Feature-Extraktion
    feats = extract_features(df_interp).reshape(1, -1)
    # 4) Skalierung
    feats_s = scaler.transform(feats)
    # 5) Vorhersage (das liefert jetzt einen String, z.B. "sues (Schere)")
    predicted_label = clf.predict(feats_s)[0]
    # 6) Mappe String → Integer-ID
    spell_id   = LABEL_TO_ID.get(predicted_label, 0)
    # 7) Mappe ID → simpler Name
    spell_name = get_spellname(spell_id)
    return (spell_id, spell_name)

def get_spellname(id):
    if id == 1:
        return spellname1
    elif id == 2:
        return spellname2
    elif id == 3:
        return spellname3
    else:
        return "Unknown Spell"
