from typing import Dict, List, Any

class PythagoreanModel:
    def __init__(self, exponent: float = 2.37):
        self.exponent = exponent
        self.stats: Dict[str, Dict[str, float]] = {}

    def train(self, games: List[Dict]):
        temp_stats = {}
        
        for game in games:
            if game.get('Status') != 'Final':
                continue
                
            home = game['HomeTeam']
            away = game['AwayTeam']
            h_score = game.get('HomeScore', 0)
            a_score = game.get('AwayScore', 0)
            
            if home not in temp_stats: temp_stats[home] = {'PF': 0, 'PA': 0}
            if away not in temp_stats: temp_stats[away] = {'PF': 0, 'PA': 0}
            
            temp_stats[home]['PF'] += h_score
            temp_stats[home]['PA'] += a_score
            temp_stats[away]['PF'] += a_score
            temp_stats[away]['PA'] += h_score
            
        self.stats = {}
        for team, data in temp_stats.items():
            pf = data['PF']
            pa = data['PA']
            if pf == 0 and pa == 0:
                wp = 0.5
            else:
                wp = (pf ** self.exponent) / ((pf ** self.exponent) + (pa ** self.exponent))
            self.stats[team] = wp

    def get_win_pct(self, team: str) -> float:
        return self.stats.get(team, 0.5)

    def get_win_probability(self, team_a: str, team_b: str, is_home: bool = False) -> float:
        pa = self.get_win_pct(team_a)
        pb = self.get_win_pct(team_b)
        
        if pa == 0: pa = 0.01
        if pa == 1: pa = 0.99
        if pb == 0: pb = 0.01
        if pb == 1: pb = 0.99
        
        odds_a = pa / (1.0 - pa)
        odds_b = pb / (1.0 - pb)
        
        hfa_mult = 1.0
        if is_home:
            hfa_mult = 1.5 
            
        odds_match = (odds_a / odds_b) * hfa_mult
        
        prob = odds_match / (1.0 + odds_match)
        return prob
