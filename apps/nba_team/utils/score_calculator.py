# Path: apps/nba_team/utils/score_calculator.py

import pandas as pd

def z_score(series: pd.Series) -> pd.Series:
    """Calculate z-scores for a series."""
    s = series.astype(float)
    m = s.mean()
    sd = s.std() or 1.0
    return (s - m) / sd


def calculate_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate off_score, def_score, and overall score for all players."""
    df = df.copy()

    # Z-score transformations
    fz_pts  = z_score(df["pts"])
    fz_ast  = z_score(df["ast"])
    fz_ts   = z_score(df["ts_pct"])
    fz_usg  = z_score(df["usg_pct"])
    fz_oreb = z_score(df["oreb_pct"])
    fz_dreb = z_score(df["dreb_pct"])
    fz_h    = z_score(df["player_height"])
    fz_w    = z_score(df["player_weight"])

    # Offensive score: PTS, AST, TS%, USG%, OREB%
    df["off_score"] = 0.40*fz_pts + 0.25*fz_ast + 0.20*fz_ts + 0.10*fz_usg + 0.05*fz_oreb

    # Defensive score: DREB%, height, weight (proxies for defense)
    df["def_score"] = 0.55*fz_dreb + 0.25*fz_h + 0.20*fz_w

    # Overall score: blend of offense and defense
    df["score"] = 0.5*df["off_score"] + 0.5*df["def_score"]

    return df


def get_mode_config(mode: str):
    """Get configuration for a specific team mode."""
    if mode == "Offense":
        return {
            "score_col": "off_score",
            "require_pred_first": False,
            "weights": {'pts':0.40, 'ast':0.25, 'ts_pct':0.20, 'usg_pct':0.10, 'oreb_pct':0.05},
            "description": "Offense favors high PTS/AST/TS% with responsibility (USG%) and extra chances (OREB%)."
        }
    elif mode == "Defense":
        return {
            "score_col": "def_score",
            "require_pred_first": False,
            "weights": {'dreb_pct':0.55, 'player_height':0.25, 'player_weight':0.20},
            "description": "Defense favors DREB% and size proxies (height/weight) as rim/board impact."
        }
    else:  # Overall
        return {
            "score_col": "score",
            "require_pred_first": True,
            "weights": {'net_rating':0.30, 'pts':0.25, 'ast':0.15, 'reb':0.15, 'ts_pct':0.10,
                       'oreb_pct':0.025, 'dreb_pct':0.025, 'usg_pct':0.025},
            "description": "Overall blends offense & defense (via net rating, PTS/AST/REB, TS%, usage, rebounding)."
        }
