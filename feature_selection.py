import numpy as np
import pandas as pd

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
