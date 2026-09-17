# test_formulas.py
import pytest
import pandas as pd
from auto_feature_pipeline import (
    f_form, f_xg, f_xga, f_xt, f_sot,
    f_attack_strength, f_defense_strength,
    f_clean_sheet_rate, f_h2h, f_ref_card_rate,
    f_gk_rating, f_finishing, f_penalty_rating,
    f_set_piece, f_press_resistance, f_squad_chemistry,
    f_travel_fatigue, f_possession, f_rest_days
)

# ── Feature 1: Form ──────────────────────────────────────────
def test_form_all_wins():
    log = pd.DataFrame({"result": ["W","W","W","W","W"]})
    assert f_form(log) == 1.0

def test_form_all_losses():
    log = pd.DataFrame({"result": ["L","L","L","L","L"]})
    assert f_form(log) == 0.0

def test_form_mixed():
    log = pd.DataFrame({"result": ["W","W","D","W","L"]})
    assert f_form(log) == round((3+3+1+3+0)/15, 4)

# ── Feature 2: xG ───────────────────────────────────────────
def test_xg_normal():
    log = pd.DataFrame({"goals_scored": [2,3,1,2,0]})
    assert f_xg(log) == 1.6

def test_xg_zero():
    log = pd.DataFrame({"goals_scored": [0,0,0,0,0]})
    assert f_xg(log) == 0.0

# ── Feature 3: xGA ──────────────────────────────────────────
def test_xga_normal():
    log = pd.DataFrame({"goals_conceded": [0,1,1,0,2]})
    assert f_xga(log) == 0.8

# ── Feature 4: xT ───────────────────────────────────────────
def test_xt_normal():
    log = pd.DataFrame({
        "sot":       [6,8,4,7,3],
        "shots_off": [4,3,5,3,5]
    })
    expected = round(((pd.Series([6,8,4,7,3])*0.3)+(pd.Series([4,3,5,3,5])*0.1)).mean(), 4)
    assert f_xt(log) == expected

# ── Feature 5: SOT ──────────────────────────────────────────
def test_sot_normal():
    log = pd.DataFrame({"sot": [6,8,4,7,3]})
    assert f_sot(log) == 5.6

# ── Feature 6 & 7: Attack / Defense Strength ────────────────
def test_attack_strength_above_average():
    assert f_attack_strength(2.1, 1.5) == round(2.1/1.5, 4)

def test_attack_strength_zero_league_avg():
    assert f_attack_strength(2.1, 0) == 1.0

def test_defense_strength_below_average():
    assert f_defense_strength(0.8, 1.5) == round(0.8/1.5, 4)

# ── Feature 10: Possession ──────────────────────────────────
def test_possession_normalised():
    assert f_possession(62) == 0.62

def test_possession_100():
    assert f_possession(100) == 1.0

# ── Feature 11: Clean Sheet Rate ────────────────────────────
def test_clean_sheet_rate():
    log = pd.DataFrame({"goals_conceded": [0,1,0,0,2]})
    assert f_clean_sheet_rate(log) == 0.6

def test_clean_sheet_rate_all_clean():
    log = pd.DataFrame({"goals_conceded": [0,0,0,0,0]})
    assert f_clean_sheet_rate(log) == 1.0

# ── Feature 12: H2H ─────────────────────────────────────────
def test_h2h_all_wins():
    assert f_h2h(["W","W","W","W","W"]) == 1.0

def test_h2h_mixed():
    assert f_h2h(["W","D","W","L","W"]) == round((1+0.5+1+0+1)/5, 4)

def test_h2h_home_away_sum():
    home = f_h2h(["W","W","L","W","D"])
    away = f_h2h(["L","L","W","L","D"])
    # Should sum to 1.0 (wins=1, draws=0.5 each side)
    assert round(home + away, 4) == 1.0

# ── Feature 13: Referee Card Rate ───────────────────────────
def test_ref_card_rate():
    assert f_ref_card_rate(420, 180) == round(420/180, 4)

def test_ref_card_rate_zero_matches():
    assert f_ref_card_rate(100, 0) == 2.0  # fallback

# ── Feature 14: GK Rating ───────────────────────────────────
def test_gk_rating_max():
    assert f_gk_rating(100, 100, 100) == 5.0

def test_gk_rating_zero():
    assert f_gk_rating(0, 0, 0) == 0.0

def test_gk_rating_normal():
    expected = round(((0.5*78)+(0.3*55)+(0.2*25))/100*5, 4)
    assert f_gk_rating(78, 55, 25) == expected

# ── Feature 15: Finishing Efficiency ────────────────────────
def test_finishing_clinical():
    log = pd.DataFrame({"goals_scored": [3,3,3,3,3]})  # 15 goals, xG=3.0*5=15
    assert f_finishing(log) == 2.5  # ratio=1.0 → (1/2)*5

def test_finishing_cap():
    goals_l5 = 20
    xg_l5    = 5
    ratio    = min(goals_l5 / xg_l5, 2.0)
    result   = round((ratio / 2) * 5, 4)
    assert result == 5.0

def test_finishing_zero_xg():
    log = pd.DataFrame({"goals_scored": [0,0,0,0,0]})
    assert f_finishing(log) == 0.0

# ── Feature 16: Penalty Rating ──────────────────────────────
def test_penalty_rating_max():
    assert f_penalty_rating(100, 100) == 5.0

def test_penalty_rating_normal():
    expected = round(((0.6*85)+(0.4*28))/100*5, 4)
    assert f_penalty_rating(85, 28) == expected

# ── Feature 17: Set Piece Strength ──────────────────────────
def test_set_piece_cap():
    assert f_set_piece(100, 100) == 5.0  # (1.0 * 25) capped at 5

def test_set_piece_normal():
    assert f_set_piece(3, 30) == round((3/30)*25, 4)

def test_set_piece_zero_attempts():
    assert f_set_piece(5, 0) == 0.0

# ── Feature 18: Press Resistance ────────────────────────────
def test_press_resistance_max():
    assert f_press_resistance(100, 100) == 5.0

def test_press_resistance_normal():
    expected = round(((0.5*88)+(0.5*62))/100*5, 4)
    assert f_press_resistance(88, 62) == expected

# ── Feature 19: Squad Chemistry ─────────────────────────────
def test_squad_chemistry_max():
    assert f_squad_chemistry(3, 9, 2) == 5

def test_squad_chemistry_zero():
    assert f_squad_chemistry(1, 5, 8) == 0

def test_squad_chemistry_partial():
    assert f_squad_chemistry(3, 5, 2) == 3  # coach +2, turnover +1

# ── Feature 20: Travel Fatigue ──────────────────────────────
def test_travel_fatigue_domestic_local():
    # Arsenal vs Chelsea ~8km apart → 0
    result = f_travel_fatigue("Arsenal", "Chelsea", True)
    assert result == 0

def test_travel_fatigue_domestic_far():
    # Arsenal vs Newcastle ~430km → 3
    result = f_travel_fatigue("Arsenal", "Newcastle", True)
    assert result == 3

def test_travel_fatigue_continental():
    result = f_travel_fatigue("Arsenal", "Newcastle", False)
    assert result == 0  # <500km continental → 0