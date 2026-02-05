from typing import Dict, List
import pandas as pd

class Evaluator:
    @staticmethod
    def aggregate_and_print(results: Dict[str, Dict], n_sims: int, teams_map: Dict):
        summary_data = []
        
        for team, data in results.items():
            made_playoffs = data.get('MadePlayoffs', 0)
            won_division = data.get('WonDivision', 0)
            won_sb = data.get('WonSuperBowl', 0)
            
            prob_playoffs = (made_playoffs / n_sims) * 100
            prob_division = (won_division / n_sims) * 100
            prob_sb = (won_sb / n_sims) * 100
            
            seed_counts = data.get('SeedCounts', {})
            if seed_counts:
                likely_seed = max(seed_counts, key=seed_counts.get)
            else:
                likely_seed = "-"
                
            summary_data.append({
                'Team': team,
                'Full Name': teams_map.get(team, {}).get('FullName', team),
                'Conference': teams_map.get(team, {}).get('Conference', '-'),
                'Division': teams_map.get(team, {}).get('Division', '-'),
                'Playoff %': round(prob_playoffs, 1),
                'Div Win %': round(prob_division, 1),
                'SB Win %': round(prob_sb, 1),
                'Proj Seed': likely_seed
            })
            
        df = pd.DataFrame(summary_data)
        
        df = df.sort_values(by=['SB Win %', 'Playoff %'], ascending=False)
        
        print("\n=== NFL SEASON SIMULATION (Advanced) ===")
        print(f"Based on {n_sims} Simulations (Model: Elo + Dynamic Updates)")
        print("-" * 75)
        
        cols = ['Team', 'Playoff %', 'Div Win %', 'SB Win %', 'Proj Seed']
        
        for conf in ['AFC', 'NFC']:
            print(f"\n{conf} PROJECTIONS")
            print(df[df['Conference'] == conf][cols].to_string(index=False))
            print("-" * 55)
