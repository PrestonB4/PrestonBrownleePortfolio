# Path: apps/nba_team/utils/team_selector.py

import pandas as pd
import numpy as np

def pick_lineup_enforced(df_pred: pd.DataFrame, score_col: str, require_pred_first: bool) -> pd.DataFrame:
    """
    Enforce positions with smart fallbacks:
      1) (optional) predicted=1 within the position by score_col
      2) any player within the position by score_col
      3) position family (PG<->SG, SF<->PF, PF<->C)
      4) absolute best remaining by score_col
    """
    order = ["PG", "SG", "SF", "PF", "C"]
    family = {"PG":["PG","SG"], "SG":["SG","PG"], "SF":["SF","PF"], "PF":["PF","SF","C"], "C":["C","PF"]}

    used = set()
    picks = []
    for pos in order:
        # Stage 1: predicted=1 within position
        if require_pred_first:
            pool1 = df_pred[(df_pred["position"] == pos) & (df_pred["prediction"] == 1)]
            pool1 = pool1.sort_values([score_col,"pred_proba","ts_pct"], ascending=[False,False,False])
            pool1 = pool1[~pool1.index.isin(used)]
            if not pool1.empty:
                idx = pool1.index[0]
                row = pool1.loc[idx].copy()
                row["position_assigned"] = pos
                row["position_note"] = ""
                picks.append(row); used.add(idx); continue

        # Stage 2: same position
        pool2 = df_pred[df_pred["position"] == pos].sort_values([score_col,"ts_pct","pred_proba"], ascending=[False,False,False])
        pool2 = pool2[~pool2.index.isin(used)]
        if not pool2.empty:
            idx = pool2.index[0]
            row = pool2.loc[idx].copy()
            row["position_assigned"] = pos
            row["position_note"] = "" if require_pred_first else ""
            picks.append(row); used.add(idx); continue

        # Stage 3: position family
        fam = family.get(pos, [pos])
        pool3 = df_pred[df_pred["position"].isin(fam)].sort_values([score_col,"ts_pct","pred_proba"], ascending=[False,False,False])
        pool3 = pool3[~pool3.index.isin(used)]
        if not pool3.empty:
            idx = pool3.index[0]
            row = pool3.loc[idx].copy()
            row["position_assigned"] = pos
            row["position_note"] = f"(fallback: from {row['position']})"
            picks.append(row); used.add(idx); continue

        # Stage 4: global best remaining
        pool4 = df_pred.sort_values([score_col,"ts_pct","pred_proba"], ascending=[False,False,False])
        pool4 = pool4[~pool4.index.isin(used)]
        if not pool4.empty:
            idx = pool4.index[0]
            row = pool4.loc[idx].copy()
            row["position_assigned"] = pos
            row["position_note"] = f"(global fallback: from {row['position']})"
            picks.append(row); used.add(idx)

    return pd.DataFrame(picks).reset_index(drop=True) if picks else pd.DataFrame()


def get_candidates_for_position(df_pred, position, score_col, require_pred_first):
    """Get all candidates for a specific position, sorted by score."""
    g = df_pred[df_pred["position"] == position].copy()
    if g.empty:
        return g
    if require_pred_first:
        g_opt  = g[g["prediction"] == 1].sort_values([score_col,"pred_proba","ts_pct"], ascending=[False,False,False])
        g_rest = g[g["prediction"] != 1].sort_values([score_col,"pred_proba","ts_pct"], ascending=[False,False,False])
        g = pd.concat([g_opt, g_rest], axis=0)
    else:
        g = g.sort_values([score_col,"ts_pct","pred_proba"], ascending=[False,False,False])

    return g.reset_index(drop=True)
