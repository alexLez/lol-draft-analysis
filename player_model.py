import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split as tts


def calculate_diffs(match_data, pos, column):
    """ Regression works on the difference of a player to the opponent"""
    match_data[f"{pos}{column}_diff"] = match_data[f"{pos}{column}"] - \
                                                match_data[f"opp_{pos}{column}"]
    return match_data

def calculate_positional_differences(match_data):
    """Calculate the differences in the game for each role, at different stages of
    the game. """
    
    cols = ['early_game', "mid_game", "late_game", "ap", "ad"]
    pos_list = ['top_', 'jng_', 'mid_', 'bot_', 'sup_']

    for col in cols:
        for pos in pos_list:
            match_data = calculate_diffs(match_data, pos, col)


def fit_win_lane_model(match_data, lane, pos_list, use_draft):
    """Fits a lane win model given the lanes to interact with. If draft information is
    used, interactions with champion early/mid/late are included"""

    logit_formula = f"{lane}_lead_at_15 ~ "
    cols = ['early_game', "mid_game", "late_game"]
    pos_col_list = []
    for pos in pos_list:
        logit_formula += f" + {pos}_dif"
    
        if use_draft:
            for col in cols:
                logit_formula += f" + {pos}_{col}_diff"
                pos_col_list.append(f"{pos}_{col}_diff")

    if use_draft:
        # Positional interactions
        for i in pos_col_list:
            for j in pos_col_list:
                if i != j:
                    logit_formula += f" + {i}:{j}"

    train, test = tts(match_data, test_size=0.3, random_state=123)

    lane_lead_prob_model = smf.logit(
        formula=logit_formula, data=train
    ).fit_regularized(alpha=10, disp=0)

    name = "post_draft" if use_draft else "draft_agnostic"

    match_data[f'{name}_{lane}_lead_prob'] = lane_lead_prob_model.predict(
        match_data)


def fit_player_model(match_data, use_draft_info):

    """Fits the player model for all three lanes"""
    fit_win_lane_model(match_data, 'top', ['top', "jng"], use_draft=use_draft_info)
    fit_win_lane_model(match_data, 'mid', ['mid', 'jng'], use_draft=use_draft_info)
    fit_win_lane_model(match_data, 'bot', ['bot', 'jng', 'sup'], use_draft=use_draft_info)
    return match_data