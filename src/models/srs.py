from typing import Dict, List, Any

class SRSModel:
    def __init__(self):
        self.ratings: Dict[str, float] = {}

    def train(self, games: List[Dict], iterations: int = 10):
        margins = {}
        opponents = {}
        
        for game in games:
            if game.get('Status') != 'Final': 
                continue
                
            home = game['HomeTeam']
            away = game['AwayTeam']
            h_score = game.get('HomeScore', 0)
            a_score = game.get('AwayScore', 0)
            margin = h_score - a_score
            
            if home not in margins: 
                margins[home] = []
                opponents[home] = []
            if away not in margins: 
                margins[away] = []
                opponents[away] = []
                
            margins[home].append(margin)
            margins[away].append(-margin)
            
            opponents[home].append(away)
            opponents[away].append(home)
            
        avg_margins = {t: sum(m)/len(m) for t, m in margins.items() if m}
        
        current_srs = avg_margins.copy()
        
        for _ in range(iterations):
            new_srs = {}
            for team in current_srs:
                opps = opponents.get(team, [])
                if not opps:
                    new_srs[team] = current_srs[team]
                    continue
                    
                opp_srs_sum = sum(current_srs.get(opp, 0.0) for opp in opps)
                sos = opp_srs_sum / len(opps)
                
                new_srs[team] = avg_margins[team] + sos
            
            current_srs = new_srs
            
        self.ratings = current_srs

    def get_rating(self, team: str) -> float:
        return self.ratings.get(team, 0.0)

    def get_win_probability(self, team_a: str, team_b: str, is_home: bool = False) -> float:
        ra = self.get_rating(team_a)
        rb = self.get_rating(team_b)
        
        hfa = 2.5 if is_home else 0.0
        
        pred_margin = (ra - rb) + hfa
        
        elo_diff_equiv = pred_margin * 25.0
        prob = 1.0 / (1.0 + 10 ** (-elo_diff_equiv / 400.0))
        
        return prob
