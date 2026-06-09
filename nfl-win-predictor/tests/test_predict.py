from src.predict import predict_game


def test_predict_returns_valid_probability():
    result = predict_game("KC", "BAL")
    assert 0.0 <= result["home_win_prob"] <= 1.0
    assert abs(result["home_win_prob"] + result["away_win_prob"] - 1.0) < 1e-6
    assert len(result["shap_factors"]) == 3
    assert "home_stats" in result and "away_stats" in result
    assert len(result["home_players"]) == 3
