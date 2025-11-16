# Path: app/streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models.ann_numpy import ANN

st.set_page_config(page_title="Optimal NBA Team Selector", page_icon="🏀", layout="wide")

# ============================================================
# Load data
# ============================================================
df = pd.read_csv("data/preprocessed_players.csv")
X_train = np.load("data/X_train.npy")
y_train = np.load("data/y_train.npy")
X_test  = np.load("data/X_test.npy")
y_test  = np.load("data/y_test.npy")
X_all   = np.load("data/X_all.npy")

try:
    with open("data/season_label.txt", "r") as f:
        season_label = f.read().strip()
except Exception:
    season_label = "Selected Seasons"

# ============================================================
# Train model + predictions (overall labels)
# ============================================================
model = ANN(layer_dims=[12, 32, 16, 1], learning_rate=0.01, epochs=1000)
model.fit(X_train, y_train)

y_pred_test = model.predict(X_test)
test_acc = (y_pred_test == y_test).mean() * 100

pred_all  = model.predict(X_all).flatten().astype(int)
proba_all = model.predict_proba(X_all).flatten()

df = df.copy()
df["prediction"] = pred_all
df["pred_proba"] = proba_all

# ============================================================
# Ensure Offense/Defense/Overall scores exist (fallback compute)
# ============================================================
def _z(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    m = s.mean()
    sd = s.std() or 1.0
    return (s - m) / sd

if any(c not in df.columns for c in ["off_score", "def_score", "score"]):
    fz_pts  = _z(df["pts"])
    fz_ast  = _z(df["ast"])
    fz_ts   = _z(df["ts_pct"])
    fz_usg  = _z(df["usg_pct"])
    fz_oreb = _z(df["oreb_pct"])
    fz_dreb = _z(df["dreb_pct"])
    fz_h    = _z(df["player_height"])
    fz_w    = _z(df["player_weight"])

    df["off_score"] = 0.40*fz_pts + 0.25*fz_ast + 0.20*fz_ts + 0.10*fz_usg + 0.05*fz_oreb
    df["def_score"] = 0.55*fz_dreb + 0.25*fz_h + 0.20*fz_w
    # "score" is overall (blend) to keep app compatibility
    if "score" not in df.columns:
        df["score"] = 0.5*df["off_score"] + 0.5*df["def_score"]

# ============================================================
# Helpers
# ============================================================
FMT = {
    "pred_proba":"{:.3f}",
    "pts":"{:.1f}", "ast":"{:.1f}", "reb":"{:.1f}",
    "net_rating":"{:.1f}", "ts_pct":"{:.3f}", "usg_pct":"{:.3f}",
    "ast_pct":"{:.3f}", "oreb_pct":"{:.3f}", "dreb_pct":"{:.3f}",
    "age":"{:.1f}", "player_height":"{:.1f}", "player_weight":"{:.1f}",
    "score":"{:.3f}", "off_score":"{:.3f}", "def_score":"{:.3f}"
}

def candidates_for_pos(df_pred, pos, score_col, require_pred_first):
    g = df_pred[df_pred["position"] == pos].copy()
    if g.empty:
        return g
    if require_pred_first:
        g_opt  = g[g["prediction"] == 1].sort_values([score_col,"pred_proba","ts_pct"], ascending=[False,False,False])
        g_rest = g[g["prediction"] != 1].sort_values([score_col,"pred_proba","ts_pct"], ascending=[False,False,False])
        g = pd.concat([g_opt, g_rest], axis=0)
    else:
        g = g.sort_values([score_col,"ts_pct","pred_proba"], ascending=[False,False,False])
    g = g.reset_index(drop=True)
    g.index = np.arange(1, len(g) + 1)
    cols = ["player_name","team_abbreviation","position","prediction","pred_proba",score_col,
            "pts","ast","reb","net_rating","ts_pct","usg_pct","ast_pct","oreb_pct","dreb_pct",
            "age","player_height","player_weight"]
    cols = [c for c in cols if c in g.columns]
    return g[cols]

def style_max_green(table: pd.DataFrame, numeric_cols, fmt_map):
    styler = table.style.format({k:v for k,v in fmt_map.items() if k in table.columns})
    for c in numeric_cols:
        if c in table.columns and pd.api.types.is_numeric_dtype(table[c]):
            styler = styler.apply(
                lambda s: ['color: green; font-weight: 700' if (pd.notnull(v) and v == s.max()) else '' for v in s],
                subset=[c]
            )
    return styler

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

def z_against_pool(col, team_series, pool_df):
    m = pool_df[col].mean()
    s = pool_df[col].std() or 1.0
    return (team_series.astype(float) - m) / s

# ============================================================
# Title + MODE selector
# ============================================================
st.title(f"🏀 Optimal NBA Team Selector — ANN (NumPy) | {season_label}")

team_mode = st.radio("Team type:", ["Overall", "Offense", "Defense"], index=0, horizontal=True)

if team_mode == "Offense":
    score_col = "off_score"
    require_pred_first = False
    hilite_cols = ["off_score","pts","ast","ts_pct","usg_pct","oreb_pct","pred_proba"]
    explain_weights = {'pts':0.40,'ast':0.25,'ts_pct':0.20,'usg_pct':0.10,'oreb_pct':0.05}
    mode_blurb = "Offense favors high **PTS/AST/TS%** with responsibility (**USG%**) and extra chances (**OREB%**)."
elif team_mode == "Defense":
    score_col = "def_score"
    require_pred_first = False
    hilite_cols = ["def_score","dreb_pct","player_height","player_weight","pred_proba","net_rating"]
    explain_weights = {'dreb_pct':0.55,'player_height':0.25,'player_weight':0.20}
    mode_blurb = "Defense favors **DREB%** and size proxies (**height/weight**) as rim/board impact."
else:  # Overall
    score_col = "score"
    require_pred_first = True
    hilite_cols = ["score","pred_proba","pts","ast","reb","net_rating","ts_pct","usg_pct","ast_pct","oreb_pct","dreb_pct"]
    explain_weights = {'net_rating':0.30,'pts':0.25,'ast':0.15,'reb':0.15,'ts_pct':0.10,'oreb_pct':0.025,'dreb_pct':0.025,'usg_pct':0.025}
    mode_blurb = "Overall blends offense & defense (via **net rating**, **PTS/AST/REB**, **TS%**, usage, rebounding)."

st.caption(f"**Mode:** {team_mode} — {mode_blurb}")

# ============================================================
# 1) AVAILABLE PLAYERS (stacked by position)
# ============================================================
st.subheader(f"All Candidates per Position — Sorted by **{score_col}**")
st.caption(("Predicted=1 shown first, then the rest." if require_pred_first else "Sorted purely by the selected score.")
           + " Best value per column is shown in green.")

for pos in ["PG","SG","SF","PF","C"]:
    st.markdown(f"### {pos}")
    tbl = candidates_for_pos(df, pos, score_col, require_pred_first)
    if tbl.empty:
        st.info(f"No players for {pos} in this pool.")
    else:
        st.write(style_max_green(tbl, hilite_cols + [score_col], FMT))

# ============================================================
# 2) PREDICTED TEAM (position-enforced) + WHY THIS TEAM (MODE)
# ============================================================
team_df = pick_lineup_enforced(df, score_col, require_pred_first)

col1, col2 = st.columns([2,1])
with col1:
    st.subheader(f"Predicted {team_mode} Team (One per Position)")
    show_cols = ["position_assigned","player_name","team_abbreviation","position",score_col,
                 "pts","reb","ast","net_rating","ts_pct","pred_proba","position_note"]
    show_cols = [c for c in show_cols if c in team_df.columns]
    st.dataframe(team_df[show_cols].reset_index(drop=True), use_container_width=True)

with col2:
    st.subheader("Model Snapshot")
    st.metric("Test Accuracy", f"{test_acc:.2f}%")
    st.write("Training set: 80 players")
    st.write("Validation/Test set: 20 players")
    st.caption("ANN is trained on 'overall' labels; Offense/Defense views use score ranking.")

# -------- Why this team for THIS mode (player-level) --------
st.subheader(f"Why these picks? ({team_mode} drivers)")
expl_cols = list(explain_weights.keys())

def safe_z(col):
    if col not in df.columns or col not in team_df.columns:
        return pd.Series([0.0]*len(team_df))
    return z_against_pool(col, team_df[col], df)

zs = {c: safe_z(c) for c in expl_cols}
rows = []
for i, row in team_df.iterrows():
    contrib = {c: float(zs[c].iloc[i]) * explain_weights[c] for c in expl_cols}
    top = sorted(contrib.items(), key=lambda kv: kv[1], reverse=True)[:3]
    rows.append({
        "position": row["position_assigned"],
        "player_name": row["player_name"],
        "score_used": score_col,
        "top1": f"{top[0][0]} ({top[0][1]:+.3f})",
        "top2": f"{top[1][0]} ({top[1][1]:+.3f})",
        "top3": f"{top[2][0]} ({top[2][1]:+.3f})"
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True)

# -------- Team-level drivers: why THIS TEAM is best for the mode --------
st.markdown("#### Team-level drivers")
team_means = {}
for c, w in explain_weights.items():
    if c in df.columns and c in team_df.columns:
        team_means[c] = float(z_against_pool(c, team_df[c], df).mean()) * w
if team_means:
    team_factors = sorted(team_means.items(), key=lambda kv: kv[1], reverse=True)
    team_view = pd.DataFrame(
        [{"stat": k, "weighted_team_z": v} for k, v in team_factors]
    )
    st.dataframe(team_view, use_container_width=True)
    st.caption("Higher weighted_team_z ⇒ this team is collectively strong (vs. pool) on that stat under current mode.")

# ============================================================
# 3) COURT (compact)
# ============================================================
st.subheader("Court Visualization (Positions)")

BACKGROUND = "#EAD8C0"   # light brown
COURT_LINE = "#C12020"   # red lines
LINE_W = 2.5

def draw_half_court(ax,
                    court_w=50.0, court_h=47.0,
                    key_w=16.0, key_h=19.0,
                    ft_radius=6.0,
                    rim_c=(25.0, 5.25), rim_r=0.75,
                    backboard_x1=22.0, backboard_x2=28.0, backboard_y=4.0,
                    arc_r=23.75, arc_deg_start=22, arc_deg_end=158,
                    corner_y_max=14.0, compact=True, top_margin=1.5,
                    court_color=COURT_LINE, background_color=BACKGROUND, line_width=LINE_W):

    ax.set_facecolor(background_color)

    key_x1 = (court_w - key_w) / 2
    key_x2 = (court_w + key_w) / 2

    ft_circle = plt.Circle((court_w/2, key_h), ft_radius, fill=False,
                           edgecolor=court_color, lw=line_width)
    rim = plt.Circle(rim_c, rim_r, fill=False, edgecolor=court_color, lw=line_width)

    theta = np.linspace(np.deg2rad(arc_deg_start), np.deg2rad(arc_deg_end), 180)
    x_arc = rim_c[0] + arc_r * np.cos(theta)
    y_arc = rim_c[1] + arc_r * np.sin(theta)
    arc_y_top = float(np.max(y_arc))
    y_top = min(court_h, max(key_h, arc_y_top, corner_y_max) + top_margin) if compact else court_h

    ax.plot([0, court_w], [0, 0], color=court_color, lw=line_width)
    ax.plot([0, 0], [0, y_top], color=court_color, lw=line_width)
    ax.plot([court_w, court_w], [0, y_top], color=court_color, lw=line_width)
    if not compact:
        ax.plot([0, court_w], [court_h, court_h], color=court_color, lw=line_width)

    ax.plot([key_x1, key_x2], [0, 0],         color=court_color, lw=line_width)
    ax.plot([key_x1, key_x1], [0, key_h],     color=court_color, lw=line_width)
    ax.plot([key_x2, key_x2], [0, key_h],     color=court_color, lw=line_width)
    ax.plot([key_x1, key_x2], [key_h, key_h], color=court_color, lw=line_width)

    ax.add_artist(ft_circle)
    ax.add_artist(rim)
    ax.plot([backboard_x1, backboard_x2], [backboard_y, backboard_y], color=court_color, lw=line_width)
    ax.plot(x_arc, y_arc, color=court_color, lw=line_width)

    cy = min(corner_y_max, y_top)
    ax.plot([rim_c[0]-22.0, rim_c[0]-22.0], [0, cy], color=court_color, lw=line_width)
    ax.plot([rim_c[0]+22.0, rim_c[0]+22.0], [0, cy], color=court_color, lw=line_width)

    ax.set_xlim(0, court_w)
    ax.set_ylim(0, y_top)
    ax.set_aspect('equal')
    ax.axis('off')

coords = {"PG": (25, 22), "SG": (15, 14), "C": (35, 14), "SF": (15, 6), "PF": (35, 6)}
fig, ax = plt.subplots(figsize=(7, 7))
fig.patch.set_facecolor(BACKGROUND)
draw_half_court(ax, compact=True, top_margin=1.5)

# Place names
name_by_pos = {p:"" for p in ["PG","SG","SF","PF","C"]}
for _, r in team_df.iterrows():
    name_by_pos[r["position_assigned"]] = r["player_name"]
for p, (x, y) in coords.items():
    ax.text(
        x, y, f"{p}\n{name_by_pos.get(p,'')}",
        ha='center', va='center', color='black',
        bbox=dict(boxstyle='round,pad=0.2', fc=BACKGROUND, ec='none')
    )
st.pyplot(fig)

# ============================================================
# Sidebar: Player Explorer
# ============================================================
st.sidebar.header("Player Explorer")
player = st.sidebar.selectbox("Select a player", df["player_name"].tolist())
prow = df[df["player_name"] == player].iloc[0]
st.sidebar.write("### Player Stats")
for k in ["position","age","player_height","player_weight","pts","reb","ast",
          "net_rating","ts_pct","usg_pct","ast_pct","oreb_pct","dreb_pct",
          "score","off_score","def_score","pred_proba","prediction"]:
    if k in prow:
        st.sidebar.write(f"**{k}**: {prow[k]}")

# ============================================================
# Stat Glossary
# ============================================================
st.markdown("---")
st.markdown("### Stat Glossary")
st.markdown(
    "- **prediction**: Model’s binary output (1 = predicted optimal, 0 = not optimal).\n"
    "- **pred_proba**: Model’s confidence (0–1) that a player is optimal (from the ANN trained on overall labels).\n"
    "- **score (Overall)**: 0.5·off_score + 0.5·def_score.\n"
    "- **off_score**: 0.40·PTS + 0.25·AST + 0.20·TS% + 0.10·USG% + 0.05·OREB% (z-scored).\n"
    "- **def_score**: 0.55·DREB% + 0.25·height + 0.20·weight (z-scored proxies).\n"
    "- Other stats: net_rating (team +/- per 100), ts_pct (true shooting), usg_pct (usage), "
    "ast_pct (assist%), oreb_pct/dreb_pct (rebound%), plus PTS/AST/REB per game."
)
