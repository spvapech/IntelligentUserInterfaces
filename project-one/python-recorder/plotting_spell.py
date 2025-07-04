import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

def extract_features(df):
    cols = ["accX","accY","accZ","gyroX","gyroY","gyroZ"]
    A = df[cols].to_numpy()
    return np.hstack([A.mean(axis=0), A.std(axis=0)])

# 1) Daten laden
paths = glob.glob("recordings/*.csv")
X_all, y_all = [], []
for fn in paths:
    df = pd.read_csv(fn, delimiter=';')
    X_all.append(extract_features(df))
    y_all.append(df["spellName"].iloc[0])
X_all = np.vstack(X_all)
y_all = np.array(y_all)

# 2) Welche Features projizieren wir? (z.B. mean accX vs mean accY)
feat_x, feat_y = 0, 1

# 3) Loop über jede Klasse
for spell in np.unique(y_all):
    # Indizes für diese Klasse
    idxs = np.where(y_all == spell)[0]
    # Test-Punkt = erstes Sample dieser Klasse
    test_idx = idxs[0]
    pt = X_all[test_idx, [feat_x, feat_y]]

    # Trainings-Daten für NN = alle Punkte außer dem Test-Punkt
    mask_train = np.ones(len(y_all), dtype=bool)
    mask_train[test_idx] = False
    X_train = X_all[mask_train][:, [feat_x, feat_y]]
    y_train = y_all[mask_train]

    # 4) NN-Finder trainieren
    nn = NearestNeighbors(n_neighbors=1).fit(X_train)
    _, idx_nn = nn.kneighbors([pt])
    nbr_pt    = X_train[idx_nn[0][0]]
    nbr_label = y_train[idx_nn[0][0]]

    # 5) Plot aufbauen
    plt.figure(figsize=(6,6))

    # 5a) alle anderen Klassen blass grau
    for cls in np.unique(y_train):
        cls_mask = (y_train == cls)
        color = 'gray' if cls != spell else 'C0'  # Ziel-Klasse in Farbe C0
        alpha = 0.3 if cls != spell else 0.8
        plt.scatter(
            X_train[cls_mask,0], X_train[cls_mask,1],
            c=color, alpha=alpha,
            label=(cls if cls==spell else None), s=40
        )

    # 5b) Test-Punkt als schwarzes Kreuz
    plt.scatter(pt[0], pt[1], marker='X', s=120, c='k', label="Test-Punkt")

    # 5c) Linie zum nächsten Nachbarn in Rot
    plt.plot(
        [pt[0], nbr_pt[0]],
        [pt[1], nbr_pt[1]],
        'r--', linewidth=2,
        label=f"NN → {nbr_label}"
    )

    # 6) Feinschliff
    plt.xlabel("Feature 0 (mean accX)")
    plt.ylabel("Feature 1 (mean accY)")
    plt.title(f"1-NN für Spell: {spell}")
    plt.legend(loc="best")
    plt.tight_layout()

    # 7) Speichern
    fn = f"nn_{spell.replace(' ','_')}.png"
    plt.savefig(fn)
    print("Gespeichert:", fn)
    plt.close()
