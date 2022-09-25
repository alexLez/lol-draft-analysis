import statsmodels.formula.api as smf
import statsmodels.api as sm
import numpy as np

def train_draft_agnostic_model(train_data):
    """Train the draft agnostic model"""

    draft_agnostic_lead_prob_model = smf.logit(
        formula="result ~side + elo_diff + (draft_agnostic_bot_lead_prob + draft_agnostic_mid_lead_prob + draft_agnostic_top_lead_prob) - 1",
        data=train_data
    ).fit_regularized(alpha=1)
    return draft_agnostic_lead_prob_model

def train_post_draft_model(train, list_of_comps, match_data):
    """Train the post draft model"""

    # If there aren't more than 100 games of a matchup ignore
    string_list = ''
    for i in list_of_comps:
        if np.sum(abs(match_data[i])) >= 100:
            string_list += f' + (post_draft_bot_lead_prob + post_draft_mid_lead_prob + post_draft_top_lead_prob + {i})**2'
    formula = f"result ~ side {string_list} - 1"

    # The logit function from smf doesn't handle the large data well. This is equivilant
    post_draft_lead_prob_model = smf.glm(
            formula=formula,
            data=train, family=sm.families.Binomial()
           ).fit()    
           
    return post_draft_lead_prob_model

def predict_with_model(match_data, model, model_name):
    
    match_data[f'{model_name}_win_prob'] = model.predict(
        match_data)

    # Normalise the probabilities predicting both sides of the matchup
    match_data[f'{model_name}_win_prob'] = match_data[
                                                    f'{model_name}_win_prob'] / \
                                                match_data.groupby(
                                                    'gameid')[
                                                    f'{model_name}_win_prob'].transform(
                                                    'sum')