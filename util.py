import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_match_data():
    # Original data from https://oracleselixir.com 
    # We have transformed it slightly to pivot champions played by role to each game
    oracles_data = pd.read_csv("data/oracles-data.csv")

    #  Differences in player model scores. Also includes data on the champion played for convenience 
    player_model_data = pd.read_csv("data/player-diffs.csv")

    return oracles_data.merge(player_model_data, how='inner', on="id")

def print_match_and_player_data(match_data):
    columns = ["teamname", "opponent", "date", "league", "result",
               "top_dif","jng_dif","mid_dif","bot_dif","sup_dif"]
    return match_data[columns].head()

def print_player_lead_probabilities(match_data):
    columns = ["teamname","opponent",
               "post_draft_top_lead_prob","post_draft_mid_lead_prob", "post_draft_bot_lead_prob",
               "draft_agnostic_top_lead_prob","draft_agnostic_mid_lead_prob","draft_agnostic_bot_lead_prob"]
    return match_data[columns].head()

def print_team_reduced_drafts(match_drafts):
    columns = [
        "id", "team_Dives", "team_Tanks", "team_Damages", "team_Enchanters",
        "team_Picks", "team_Pokes", "team_Engages", "team_Splitpushs", "early_game", "mid_game", 
        "late_game", "ap", "ad", "no_damage_type"]
    return match_drafts[columns].head()

def print_draft_centroids(match_drafts):
    #Calculating the centroids manually gives a nice pandas output 
    return match_drafts.groupby(['team_comp']).mean()[
    ["team_Dives", "team_Tanks",
     "team_Damages",
     "team_Enchanters",
     "team_Picks",
     "team_Pokes", "team_Engages",
     "team_Splitpushs", "early_game", "mid_game",
     "late_game", 'no_damage_type']]

def merge_team_and_draft(match_data, draft_data, n_clusters=7):
    combined = match_data.merge(draft_data, how='inner', on='id')

    # Games are laid out as 
    # teamname   opp
    # team1      team2
    # team2      team1
    # This is a slighly hacky way to exploit this to generate a column for the 
    # opponent composition per game 
    combined['opp_comp_poss_1'] = combined['team_comp'].shift(1)
    combined['opp_comp_poss_2'] = combined['team_comp'].shift(-1)
    combined['opp_poss_1'] = combined['teamname'].shift(1)
    combined['opp_poss_2'] = combined['teamname'].shift(-1)
    combined['opp_comp'] = np.where(
    combined['opp_poss_1'] == combined['opponent'],
    combined['opp_comp_poss_1'], -1)
    combined['opp_comp'] = np.where(
    combined['opp_poss_2'] == combined['opponent'],
    combined['opp_comp_poss_2'], combined['opp_comp']).astype(int)
    
    combined = pd.get_dummies(combined,
                                columns=['team_comp', 'opp_comp'])

    list_of_comp_diffs = []
    for i in range(n_clusters):
        for j in range(n_clusters):
            list_of_comp_diffs.append(f'team_comp{i}_v_opp_comp{j}')
            combined[f'team_comp{i}_v_opp_comp{j}'] = np.where(
                (combined[f'team_comp_{i}'] == 1) & (
                        combined[f'opp_comp_{j}'] == 1), 1, 0)
            if i != j:
                combined[f'team_comp{i}_v_opp_comp{j}'] = np.where(
                    (combined[f'opp_comp_{i}'] == 1) & (
                            combined[f'team_comp_{j}'] == 1), -1,
                    combined[f'team_comp{i}_v_opp_comp{j}'])

    # Cleaning some bad data in the original Oracles dataset
    bad_games = combined.groupby('gameid').sum()
    list_of_bad_games = []
    for i in list_of_comp_diffs:
        for j in bad_games.index:
            if bad_games.at[j, i] not in [0, 2]:
                list_of_bad_games.append(j)
    combined = combined[
        ~combined['gameid'].isin(list_of_bad_games)]

    required_columns = ["gameid", "teamname", "opponent", "date", "league", "result",
    "side", "elo_diff","draft_agnostic_bot_lead_prob", "draft_agnostic_mid_lead_prob", 
    "draft_agnostic_top_lead_prob", "post_draft_bot_lead_prob",  "post_draft_mid_lead_prob", 
    "post_draft_top_lead_prob"]
    for i in range(n_clusters):
        for j in range(n_clusters):
            required_columns.append(f'team_comp{i}_v_opp_comp{j}')

    return combined[required_columns], list_of_comp_diffs

def print_win_prob_model_outputs(match_data):
    columns = ["teamname", "opponent", "result", 
               "draft_agnostic_win_prob", "post_draft_win_prob", "draft_diff"]
    return match_data[columns].head()

def calibration_plot(match_data):
    match_data['q'] = pd.cut(match_data['post_draft_win_prob'], 20)

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], label='perfectly calibrated',
             linestyle='dashed', color='grey')

    ax.plot(list(match_data.groupby('q').mean()['post_draft_win_prob']),
             list(match_data.groupby('q').mean()['result']),
             label='Draft Model')
    ax.legend()

    ax.set_title("Post draft model calibration plot")
    ax.set_xlabel("Predicted Win Probability")
    ax.set_ylabel("True Win Probability in each bin")
    ax.grid()

def plot_cumulative_expected_wins_added_by_draft(match_data, league, teams):
    lec = match_data[match_data['league'] == f'{league}']
    lec = lec[lec['date'] >= '2022-01-01'] # Most reason whole season
    lec['gamecount'] = lec.groupby('teamname')['draft_diff'].transform('cumcount')
    lec['wpa'] = lec.groupby('teamname')['draft_diff'].transform('cumsum')
    fig, ax = plt.subplots()
    colour_dict = teams
    for team in np.unique(lec['teamname']):
        if team in teams.keys():
            ax.plot(lec[lec['teamname'] == team]['gamecount'],
                    lec[lec['teamname'] == team]['wpa'],
                    label=team, linewidth=2, color=colour_dict[team])
        else:
            ax.plot(lec[lec['teamname'] == team]['gamecount'],
                    lec[lec['teamname'] == team]['wpa'],
                    label=team, color='Grey', alpha=0.4)
    ax.set_xlabel("Games played (2022 spring and summer split)")
    ax.set_ylabel("Cumulative expected wins")
    ax.grid()
    return ax