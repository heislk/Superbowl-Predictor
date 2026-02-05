from typing import Dict, List, Any

class QBEloModel:
    def __init__(self, base_rating: float = 1400.0, k_factor: float = 20.0):
        self.ratings: Dict[str, float] = {}
        self.base_rating = base_rating
        self.k_factor = k_factor
        
    def get_rating(self, qb_name: str) -> float:
        if not qb_name or qb_name == '':
            return self.base_rating
        return self.ratings.get(qb_name, self.base_rating)

    def train(self, games: List[Dict]):
        sorted_games = sorted(games, key=lambda x: (x['Season'], x['Week']))
        
        for game in sorted_games:
            if game.get('Status') != 'Final': continue
            
            home_qb = game.get('home_qb_name')
            away_qb = game.get('away_qb_name')
            
            if not home_qb or not away_qb:
                continue
                
            h_score = game.get('HomeScore', 0)
            a_score = game.get('AwayScore', 0)
            
            winner_qb = None
            if h_score > a_score:
                winner_qb = home_qb
                loser_qb = away_qb
                score = 1.0
            elif a_score > h_score:
                winner_qb = away_qb
                loser_qb = home_qb
                score = 0.0
            else:
                score = 0.5
            
            ra = self.get_rating(home_qb)
            rb = self.get_rating(away_qb)
            
            hfa = 30
            
            diff = rb - (ra + hfa)
            prob_a = 1.0 / (1.0 + 10 ** (diff / 400.0))
            
            change = self.k_factor * (score - prob_a)
            
            self.ratings[home_qb] = ra + change
            self.ratings[away_qb] = rb - change

    def get_win_probability(self, home_qb: str, away_qb: str, is_home: bool = False) -> float:
        ra = self.get_rating(home_qb)
        rb = self.get_rating(away_qb)
        
        hfa = 30 if is_home else 0
        
        diff = rb - (ra + hfa)
        prob = 1.0 / (1.0 + 10 ** (diff / 400.0))
        return prob
