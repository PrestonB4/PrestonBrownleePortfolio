# Path: apps/nba_team/nba_team_blueprint.py

from flask import Blueprint, jsonify, request
import pandas as pd
import numpy as np
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from models.ann_numpy import ANN
from utils.team_selector import pick_lineup_enforced, get_candidates_for_position
from utils.score_calculator import calculate_scores, get_mode_config

# Create blueprint
nba_bp = Blueprint('nba', __name__, url_prefix='/api/nba')

# Lazy loading - data and model loaded only when first used
_data_cache = None
_model_cache = None

def get_data():
    """Lazy load and process player data"""
    global _data_cache
    if _data_cache is None:
        base_path = os.path.dirname(__file__)

        # Load data
        df = pd.read_csv(os.path.join(base_path, "data", "preprocessed_players.csv"))
        X_train = np.load(os.path.join(base_path, "data", "X_train.npy"))
        y_train = np.load(os.path.join(base_path, "data", "y_train.npy"))
        X_test = np.load(os.path.join(base_path, "data", "X_test.npy"))
        y_test = np.load(os.path.join(base_path, "data", "y_test.npy"))
        X_all = np.load(os.path.join(base_path, "data", "X_all.npy"))

        try:
            with open(os.path.join(base_path, "data", "season_label.txt"), "r") as f:
                season_label = f.read().strip()
        except Exception:
            season_label = "Selected Seasons"

        # Train model
        model = ANN(layer_dims=[12, 32, 16, 1], learning_rate=0.01, epochs=1000)
        model.fit(X_train, y_train)

        # Get predictions
        y_pred_test = model.predict(X_test)
        test_acc = (y_pred_test == y_test).mean() * 100

        pred_all = model.predict(X_all).flatten().astype(int)
        proba_all = model.predict_proba(X_all).flatten()

        df["prediction"] = pred_all
        df["pred_proba"] = proba_all

        # Calculate scores
        df = calculate_scores(df)

        _data_cache = {
            "df": df,
            "model": model,
            "test_acc": test_acc,
            "season_label": season_label
        }

    return _data_cache


@nba_bp.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "ok", "app": "NBA Team Optimizer"})


@nba_bp.route('/model-stats', methods=['GET'])
def model_stats():
    """Get model statistics"""
    data = get_data()
    return jsonify({
        "test_accuracy": round(data["test_acc"], 2),
        "season_label": data["season_label"],
        "total_players": len(data["df"]),
        "architecture": "MLP: 12 -> 32 -> 16 -> 1",
        "training_size": 80,
        "test_size": 20
    })


@nba_bp.route('/team', methods=['GET'])
def get_team():
    """Get optimal team for a specific mode"""
    mode = request.args.get('mode', 'Overall')  # Overall, Offense, Defense
    data = get_data()
    df = data["df"]

    config = get_mode_config(mode)
    team_df = pick_lineup_enforced(
        df,
        config["score_col"],
        config["require_pred_first"]
    )

    # Convert to JSON-friendly format
    team_list = []
    for _, row in team_df.iterrows():
        team_list.append({
            "position_assigned": row["position_assigned"],
            "player_name": row["player_name"],
            "team_abbreviation": row["team_abbreviation"],
            "actual_position": row["position"],
            "score": round(float(row[config["score_col"]]), 3),
            "pts": round(float(row["pts"]), 1),
            "ast": round(float(row["ast"]), 1),
            "reb": round(float(row["reb"]), 1),
            "net_rating": round(float(row["net_rating"]), 1),
            "ts_pct": round(float(row["ts_pct"]), 3),
            "pred_proba": round(float(row["pred_proba"]), 3),
            "prediction": int(row["prediction"]),
            "position_note": row.get("position_note", ""),
            "age": round(float(row["age"]), 1),
            "height": round(float(row["player_height"]), 1),
            "weight": round(float(row["player_weight"]), 1)
        })

    return jsonify({
        "mode": mode,
        "description": config["description"],
        "team": team_list
    })


@nba_bp.route('/players', methods=['GET'])
def get_players():
    """Get all players for a specific position"""
    position = request.args.get('position', 'PG')  # PG, SG, SF, PF, C
    mode = request.args.get('mode', 'Overall')
    limit = request.args.get('limit', 10, type=int)

    data = get_data()
    df = data["df"]

    config = get_mode_config(mode)
    candidates = get_candidates_for_position(
        df,
        position,
        config["score_col"],
        config["require_pred_first"]
    )

    # Limit results
    candidates = candidates.head(limit)

    # Convert to JSON
    players_list = []
    for _, row in candidates.iterrows():
        players_list.append({
            "player_name": row["player_name"],
            "team_abbreviation": row["team_abbreviation"],
            "position": row["position"],
            "score": round(float(row[config["score_col"]]), 3),
            "pts": round(float(row["pts"]), 1),
            "ast": round(float(row["ast"]), 1),
            "reb": round(float(row["reb"]), 1),
            "ts_pct": round(float(row["ts_pct"]), 3),
            "pred_proba": round(float(row["pred_proba"]), 3),
            "prediction": int(row["prediction"])
        })

    return jsonify({
        "position": position,
        "mode": mode,
        "players": players_list
    })


@nba_bp.route('/player/<player_name>', methods=['GET'])
def get_player(player_name):
    """Get details for a specific player"""
    data = get_data()
    df = data["df"]

    player_row = df[df["player_name"] == player_name]

    if player_row.empty:
        return jsonify({"error": "Player not found"}), 404

    row = player_row.iloc[0]

    return jsonify({
        "player_name": row["player_name"],
        "team_abbreviation": row["team_abbreviation"],
        "position": row["position"],
        "age": round(float(row["age"]), 1),
        "height": round(float(row["player_height"]), 1),
        "weight": round(float(row["player_weight"]), 1),
        "pts": round(float(row["pts"]), 1),
        "ast": round(float(row["ast"]), 1),
        "reb": round(float(row["reb"]), 1),
        "net_rating": round(float(row["net_rating"]), 1),
        "ts_pct": round(float(row["ts_pct"]), 3),
        "usg_pct": round(float(row["usg_pct"]), 3),
        "ast_pct": round(float(row["ast_pct"]), 3),
        "oreb_pct": round(float(row["oreb_pct"]), 3),
        "dreb_pct": round(float(row["dreb_pct"]), 3),
        "score": round(float(row["score"]), 3),
        "off_score": round(float(row["off_score"]), 3),
        "def_score": round(float(row["def_score"]), 3),
        "pred_proba": round(float(row["pred_proba"]), 3),
        "prediction": int(row["prediction"])
    })
