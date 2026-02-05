from typing import List, Dict
import pandas as pd

class SeasonRules:
    @staticmethod
    def update_standings(standings: Dict[str, Dict], home: str, away: str, winner: str):
        if winner == home:
            standings[home]['Wins'] += 1
            standings[away]['Losses'] += 1
        elif winner == away:
            standings[away]['Wins'] += 1
            standings[home]['Losses'] += 1
        else:
            standings[home]['Ties'] += 1
            standings[away]['Ties'] += 1

    @staticmethod
    def determine_seeds(standings: Dict[str, Dict], divisions: Dict[str, str], conference_teams: Dict[str, List[str]]) -> Dict[str, List[str]]:
        teams_data = []
        for team, stats in standings.items():
            wins = stats['Wins']
            losses = stats['Losses']
            ties = stats['Ties']
            total = wins + losses + ties
            win_pct = (wins + 0.5 * ties) / total if total > 0 else 0
            
            teams_data.append({
                'Team': team,
                'Wins': wins,
                'WinPct': win_pct,
                'Division': divisions.get(team),
                'Conference': 'AFC' if team in conference_teams.get('AFC', []) else 'NFC'
            })
            
        df = pd.DataFrame(teams_data)
        
        seeds = {'AFC': [], 'NFC': []}
        
        for conf in ['AFC', 'NFC']:
            conf_teams = df[df['Conference'] == conf].copy()
            if conf_teams.empty:
                continue

            div_winners = []
            wild_card_pool = []
            
            conf_teams['Random'] = 0.0
            
            conf_teams = conf_teams.sort_values(by=['WinPct'], ascending=False)
            
            if 'Division' in conf_teams.columns and conf_teams['Division'].notna().all():
                 for div, group in conf_teams.groupby('Division'):
                     sorted_group = group.sort_values(by=['WinPct'], ascending=False)
                     winner = sorted_group.iloc[0]
                     div_winners.append(winner)
                     wild_card_pool.append(sorted_group.iloc[1:])
            else:
                sorted_conf = conf_teams.sort_values(by=['WinPct'], ascending=False)
                seeds[conf] = sorted_conf.head(7)['Team'].tolist()
                continue

            div_winners_df = pd.DataFrame(div_winners).sort_values(by=['WinPct'], ascending=False)
            ranked_winners = div_winners_df['Team'].tolist()
            
            if wild_card_pool:
                wc_df = pd.concat(wild_card_pool).sort_values(by=['WinPct'], ascending=False)
                ranked_wc = wc_df.head(3)['Team'].tolist()
            else:
                ranked_wc = []
                
            seeds[conf] = ranked_winners + ranked_wc
            
        return seeds
