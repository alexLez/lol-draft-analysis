from sklearn.metrics import brier_score_loss as brier_score
from sklearn.metrics import accuracy_score as acc_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

def calibration_plot(match_data):
    match_data['q'] = pd.cut(match_data['post_draft_win_prob'], 20)

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], label='perfectly calibrated',
             linestyle='dashed', color='grey')

    ax.plot(list(match_data.groupby('q')['post_draft_win_prob'].mean()),
             list(match_data.groupby('q')['result'].mean()),
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

def print_model_results(test_data):

    def win_predictor_stats(model_name):
        brier = brier_score(test_data['result'], test_data[f'{model_name}_win_prob'])
        accuracy = acc_score(test_data['result'], round(test_data[f'{model_name}_win_prob']))
        return brier, accuracy

    def lane_lead_predictor_stats(model_name):
        average_brier = np.average([
        brier_score(test_data["mid_lead_at_15"], test_data[f"{model_name}_mid_lead_prob"]),
        brier_score(test_data["top_lead_at_15"], test_data[f"{model_name}_top_lead_prob"]),
        brier_score(test_data["bot_lead_at_15"], test_data[f"{model_name}_bot_lead_prob"])])

        average_accuracy = np.average([
        acc_score(test_data["mid_lead_at_15"], round(test_data[f"{model_name}_mid_lead_prob"])),
        acc_score(test_data["top_lead_at_15"], round(test_data[f"{model_name}_top_lead_prob"])),
        acc_score(test_data["bot_lead_at_15"], round(test_data[f"{model_name}_bot_lead_prob"]))])

        return average_brier, average_accuracy

    win_brier, win_accuracy = win_predictor_stats("draft_agnostic")
    lane_brier, lane_accuracy = lane_lead_predictor_stats("draft_agnostic")
    
    print(f"""Draft Agnostic models:
    
    Lane lead average Brier score: \t{lane_brier}
    Lane lead average accuracy: \t{lane_accuracy}
    Win prob Brier score: \t\t{win_brier}
    Win prob accuracy: \t\t\t{win_accuracy}
    """)

    win_brier, win_accuracy = win_predictor_stats("post_draft")
    lane_brier, lane_accuracy = lane_lead_predictor_stats("post_draft")
    
    print(f"""Post Draft models:
    
    Lane lead average Brier score: \t{lane_brier}
    Lane lead average accuracy: \t{lane_accuracy}
    Win prob Brier score: \t\t{win_brier}
    Win prob accuracy: \t\t\t{win_accuracy}
    """)

def print_bookie_stats():
    bookie_odds = pd.read_csv("data/b365_odds.tsv", sep='\t')
    bookie_odds["implied_win_prob"] = bookie_odds['implied_h_prob'] / (bookie_odds['implied_h_prob'] + bookie_odds['implied_a_prob'])
    print("After adjusting for hold:")
    print(f"""
    Bookie Brier score:
    {brier_score(bookie_odds['result'], bookie_odds['implied_win_prob'])}\n
    Bookie accuracy
    {acc_score(bookie_odds['result'], round(bookie_odds['implied_win_prob']))}
    """
    )

def get_teamcomps_for_lcs(match_data):
    new = match_data.iloc[:, 17:66].idxmax(axis=1)
    comp = new.str.split("_v_").to_list()
    comp = [int(x[0].replace("team_comp", "")) for x in comp]
    match_data['teamcomp'] = comp
    tc_games = match_data.groupby("teamname", as_index=True)["teamcomp"].value_counts().to_frame("tc_count").reset_index()   #["teamcomp"].value_counts()
    team_games = tc_games.pivot(index="teamname", columns="teamcomp", values="tc_count").fillna(0)
    return team_games

def show_team_comp_frequencies(LCS_teamcomps):
    tl_games = LCS_teamcomps.query("teamname=='Team Liquid'").values[0]
    all_LCS_games = LCS_teamcomps.sum(axis=0)

    num_total_LCS_games = LCS_teamcomps.sum().sum()
    num_TL_games = LCS_teamcomps.query("teamname=='Team Liquid'").values.sum()


    category_names = ["Cluster 0", "Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4", "Cluster 5", "Cluster 6"]
    results = {
        f'All LCS Games\nn={num_total_LCS_games}': all_LCS_games,
        f'Team Liquid Games\nn={num_TL_games}': tl_games
    }

    labels = list(results.keys())
    data = np.array(list(results.values()))
    totals = data.sum(axis=1)
    data[0] /= totals[0]
    data[1] /= totals[1]
    data_cum = data.cumsum(axis=1)


    fig, ax = plt.subplots(figsize=(9.2, 5))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, 1)

    for i, colname in enumerate(category_names):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        rects = ax.barh(labels, widths, left=starts, height=0.5,
                        label=colname)

        ax.bar_label(rects, labels=[str(round(100*x,1)) + "%" for x in widths], label_type='center', fmt="%.2f")
        ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1), loc='lower left', fontsize='small')

    fig.suptitle("Relative Frequency of Team Draft Clusters in the LCS")

def lane_lead_correlations(games):

    reg_b = linear_model.LinearRegression().fit(games[['draft_agnostic_bot_lead_prob']], games[['post_draft_bot_lead_prob']])
    reg_m = linear_model.LinearRegression().fit(games[['draft_agnostic_mid_lead_prob']], games[['post_draft_mid_lead_prob']])
    reg_t = linear_model.LinearRegression().fit(games[['draft_agnostic_top_lead_prob']], games[['post_draft_top_lead_prob']])

    r2_b = reg_b.score(games[['draft_agnostic_bot_lead_prob']], games[['post_draft_bot_lead_prob']])
    r2_m = reg_m.score(games[['draft_agnostic_mid_lead_prob']], games[['post_draft_mid_lead_prob']])
    r2_t = reg_t.score(games[['draft_agnostic_top_lead_prob']], games[['post_draft_top_lead_prob']])

    fig, ax = plt.subplots(1, 3, figsize=(21,7))

    ax[0].scatter(games["draft_agnostic_bot_lead_prob"], y=games["post_draft_bot_lead_prob"], label=f"R^2 = {r2_b}")
    ax[1].scatter(games["draft_agnostic_mid_lead_prob"], y=games["post_draft_mid_lead_prob"], label=f"R^2 = {r2_m}")
    ax[2].scatter(games["draft_agnostic_top_lead_prob"], y=games["post_draft_top_lead_prob"], label=f"R^2 = {r2_t}")

    ax[0].plot((0.2,0.8), (0.2,0.8), color='orange', label="y=x")
    ax[1].plot((0.2,0.8), (0.2,0.8), color='orange', label="y=x")
    ax[2].plot((0.2,0.8), (0.2,0.8), color='orange', label="y=x")

    ax[0].legend()
    ax[1].legend()
    ax[2].legend()

    ax[0].set_title("Bot Lane")
    ax[1].set_title("Mid Lane")
    ax[2].set_title("Top Lane")

    ax[0].set_ylabel("Probability of Lane Lead (Post Draft)")
    ax[0].set_xlabel("Probability of Lane Lead (Draft Agnostic)")
    ax[1].set_xlabel("Probability of Lane Lead (Draft Agnostic)")
    ax[2].set_xlabel("Probability of Lane Lead (Draft Agnostic)")

    fig.suptitle("Predicted Post-Draft Lane Leads vs Draft-Agnostic", fontsize="x-large")

    return r2_b, r2_m, r2_t

def plot_jojopyun_lane_lead_above_expected(match_data):
    fig, ax = plt.subplots()
    match_data["mid_wins"] = match_data["mid_lead_at_15"] - match_data["post_draft_mid_lead_prob"]
    wins_above = match_data[["mid_wins"]].cumsum()
    ax.plot(range(len(wins_above)), wins_above)

    ax.set_title("Jojopyun Mid Lane Lead Above Expected")
    ax.set_xlabel("2022 Season Games")
    ax.set_ylabel("Mid leads above expected")
    ax.grid()
    ax.axhline(0, color='r', label="Expected performance")
    ax.legend();

