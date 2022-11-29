import csv
import pandas as pd
from scipy.stats import zscore
from sklearn.cluster import KMeans

class MappedChampion:
    """Representation a champion to their feature reduced mapped form, as defined
    in data/champ_roles.csv
    """
    def __init__(self, champion):
        self.name =       champion[0]
        self.role =       champion[1]
        self.early_game = int(champion[2])
        self.mid_game =   int(champion[3])
        self.late_game =  int(champion[4])
        self.ap =         int(champion[5])
        self.ad =         int(champion[6])

    def __repr__(self):
        powerspike = "early game" if self.early_game else "mid game" if self.mid_game else "late game"
        damage_type = "AD" if self.ad else "AP" if self.ap else "None"
        return f"{self.name} - Role: {self.role}, Powerspike: {powerspike}, DamageType {damage_type}"

def load_champion_mapping():
    """Returns a dictionary mapping each champion to their reduced form"""

    champ_reduction_dict = {}

    with open("data/champion_roles.csv") as champ_file:
        champions = csv.reader(champ_file)
    
        for i, champion in enumerate(champions):
            if i == 0: continue # csv header
            mapped_champ = MappedChampion(champion)
            champ_reduction_dict[mapped_champ.name] = mapped_champ

    return champ_reduction_dict

def get_blank_row():
    return {   
        "team_Dives" :     0, 
        "team_Tanks" :     0,
        "team_Damages" :   0,
        "team_Enchanters": 0,
        "team_Picks":      0,
        "team_Pokes":      0, 
        "team_Engages":    0,
        "team_Splitpushs": 0, 
        
        "early_game":      0,
        "mid_game":        0, 
        "late_game":       0,
        
        "ap":              0,
        "ad":              0,
        "no_damage_type":  0
        }

def add_champ_to_blank_row(row, champ):
    row[f"team_{champ.role}s"] += 1
    row["early_game"] += champ.early_game
    row["mid_game"]   += champ.mid_game
    row["late_game"]  += champ.late_game
    row["ap"]         += champ.ap
    row["ad"]         += champ.ad


def reduce_team_drafts(oracles_data, champ_map):
    """Create team draft representation by summing all comprising champion feature vectors"""

    draft_df = pd.DataFrame()

    # There are more efficient ways to do this...
    for _, row in oracles_data.iterrows():
        blank_row = get_blank_row()
        for role in ["top", "jng", "mid", "bot", "sup"]:
            add_champ_to_blank_row(blank_row, champ_map[row[f"{role}_champion"]])

        blank_row["id"] = row.id
        # A single damage type is easily drafted against so deserves 
        # it's own column for significance
        if blank_row["ap"] == 0 or blank_row["ad"] == 0:
            blank_row["no_damage_type"] = 1
        
        draft_df = pd.concat([draft_df, pd.DataFrame.from_dict([blank_row])], ignore_index=True, axis=0, join='outer')

    return draft_df

def cluster_drafts(draft_df, n_clusters=7):
    # Normalise features for a consistent distance or some roles are over represented
    cluster_columns = zscore(draft_df[
               ["team_Dives", "team_Tanks",
                "team_Damages",
                "team_Enchanters",
                "team_Picks",
                "team_Pokes", "team_Engages",
                "team_Splitpushs", "early_game",
                "late_game",
                "mid_game", 'no_damage_type']])

    cluster_model = KMeans(n_clusters=n_clusters).fit(cluster_columns)
    draft_df['team_comp'] = cluster_model.fit_predict(cluster_columns).astype(int)
    return draft_df

def print_centroids(draft_df):
    # Calculating the centroids manually gives a nice pandas output 
    return draft_df.groupby(['team_comp']).mean()[
    ["team_Dives", "team_Tanks",
     "team_Damages",
     "team_Enchanters",
     "team_Picks",
     "team_Pokes", "team_Engages",
     "team_Splitpushs", "early_game", "mid_game",
     "late_game", 'no_damage_type']]