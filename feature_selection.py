import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, f1_score, accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score



def corr_groups(X: pd.DataFrame, threshold: float = 0.9):
    """
    Regroupe les colonnes corrélées (|corr| >= threshold) en composantes connexes et
    Retourne une liste de listes de noms de colonnes.
    """
    C = X.corr().abs()
    cols = list(X.columns)
    seen, groups = set(), []
    for c in cols:
        if c in seen: 
            continue
        # BFS minimaliste
        grp, stack = [], [c]
        seen.add(c)
        while stack:
            u = stack.pop()
            grp.append(u)
            neigh = C.index[(C[u] >= threshold) & (C.index != u)]
            for v in neigh:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        groups.append(sorted(grp))
    return groups

def shuffle_group(X: pd.DataFrame, group, random_state: int = 42):
    """
    Retourne une copie de X où tt les colonnes du groupe sont shuffles
    avec la même permutation (on casse le lien de ce groupe avec y).
    """
    Xp = X.copy()
    rng = np.random.default_rng(random_state)
    perm = rng.permutation(len(Xp))
    for c in group:
        Xp[c] = Xp[c].to_numpy()[perm]
    return Xp

# helper pour évaluer l’impact moyen d’un groupe sur accuracy et F1
def impact_per_group(pipe, X_te, y_te, groups, n_repeats=10, seed=0):
    base_acc = accuracy_score(y_te, pipe.predict(X_te))
    if hasattr(pipe, "predict_proba"):
        p = pipe.predict_proba(X_te)[:, 1]
        base_f1 = f1_score(y_te, (p >= 0.5).astype(int))
    else:
        base_f1 = f1_score(y_te, pipe.predict(X_te))

    rng = np.random.default_rng(seed)
    res = []
    for g in groups:
        acc_drops, f1_drops = [], []
        for _ in range(n_repeats):
            X_te_shuf = shuffle_group(X_te, g, random_state=int(rng.integers(0, 1e9)))
            if hasattr(pipe, "predict_proba"):
                p = pipe.predict_proba(X_te_shuf)[:, 1]
                yhat = (p >= 0.5).astype(int)
            else:
                yhat = pipe.predict(X_te_shuf)
            acc_drops.append(base_acc - accuracy_score(y_te, yhat))
            f1_drops.append(base_f1 - f1_score(y_te, yhat))
        res.append({"group": tuple(g), "size": len(g),
                    "acc_drop": float(np.mean(acc_drops)),
                    "f1_drop": float(np.mean(f1_drops))})
    df_imp = pd.DataFrame(res).sort_values(["f1_drop","acc_drop"], ascending=False).reset_index(drop=True)
    print(f"Baseline  Acc={base_acc:.3f}  F1={base_f1:.3f}")
    return df_imp
