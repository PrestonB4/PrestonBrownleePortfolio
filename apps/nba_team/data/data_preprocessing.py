# Path: data/data_preprocessing.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SOURCE_CSV = "data/all_seasons.csv"
TARGET_SEASONS = ['2018-19','2019-20','2020-21','2021-22','2022-23']
POOL_SIZE = 100
TOP_PCT = 0.20

df = pd.read_csv(SOURCE_CSV)
df = df[df['season'].isin(TARGET_SEASONS)].copy()

# Keep most recent season per player inside window
def season_end_year(s: str) -> int:
    a, b = s.split('-'); b = int(b)
    century = 2000 if b < 50 else 1900
    return century + b
df['season_end'] = df['season'].apply(season_end_year)
df = df.sort_values(['player_name','season_end']).drop_duplicates('player_name', keep='last')

# Sample pool
df_sampled = df.sample(n=POOL_SIZE, random_state=42).reset_index(drop=True)

# Height/weight unit handling
mean_h = df_sampled['player_height'].mean()
height_cm = df_sampled['player_height'] * (100.0 if mean_h < 10 else 1.0)
mean_w = df_sampled['player_weight'].mean()
weight_kg = df_sampled['player_weight'] * (0.45359237 if mean_w > 150 else 1.0)

# Position inference (same as before, slightly stricter C rule)
def height_weight_to_pos(h_cm, w_kg):
    if h_cm < 190: return "PG"
    if h_cm < 198: return "SG"
    if h_cm < 204: return "SF"
    if h_cm >= 209 or w_kg >= 118: return "C"
    return "PF"
df_sampled['position'] = [height_weight_to_pos(h, w) for h, w in zip(height_cm, weight_kg)]

# Features
features = [
    'age','player_height','player_weight','pts','reb','ast',
    'net_rating','usg_pct','ts_pct','ast_pct','oreb_pct','dreb_pct'
]
for col in features:
    if df_sampled[col].isna().any():
        df_sampled[col] = df_sampled[col].fillna(df_sampled[col].median())

# z-score helper
def z(s): 
    s = s.astype(float)
    m = s.mean(); sd = s.std(ddof=0) or 1.0
    return (s - m) / sd

fz = {c: z(df_sampled[c]) for c in ['pts','ast','reb','net_rating','ts_pct','oreb_pct','dreb_pct','usg_pct']}
fz_h = z(height_cm)
fz_w = z(weight_kg)

# ---- NEW: Offense / Defense / Overall scores ----
off_score = (0.40*fz['pts'] + 0.25*fz['ast'] + 0.20*fz['ts_pct'] +
             0.10*fz['usg_pct'] + 0.05*fz['oreb_pct'])
def_score = (0.55*fz['dreb_pct'] + 0.25*fz_h + 0.20*fz_w)
overall_score = 0.5*off_score + 0.5*def_score   # you can change the mix

# Keep your original 'score' name as overall to stay app-compatible
df_sampled['off_score'] = off_score
df_sampled['def_score'] = def_score
df_sampled['score']     = overall_score  # overall

# Labels (top 20% in each)
def top_label(s): return (s >= np.quantile(s, 1 - TOP_PCT)).astype(int)
df_sampled['label_off']     = top_label(df_sampled['off_score'])
df_sampled['label_def']     = top_label(df_sampled['def_score'])
df_sampled['label_overall'] = top_label(df_sampled['score'])
df_sampled['label']         = df_sampled['label_overall']  # used by the ANN

# Scale inputs (same)
X = df_sampled[features].values
y = df_sampled['label'].values.reshape(-1, 1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Save artifacts
df_sampled.to_csv("data/preprocessed_players.csv", index=False)
np.save("data/X_train.npy", X_train)
np.save("data/X_test.npy", X_test)
np.save("data/y_train.npy", y_train)
np.save("data/y_test.npy", y_test)
np.save("data/X_all.npy", X_scaled)

season_label = TARGET_SEASONS[0] if len(TARGET_SEASONS)==1 else f"{TARGET_SEASONS[0]}–{TARGET_SEASONS[-1]}"
with open("data/season_label.txt","w") as f:
    f.write(season_label)
