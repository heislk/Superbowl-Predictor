import math
from typing import Dict, List, Tuple

class EloModel:
    def __init__(self, base_rating: float = 1500.0, k_factor: float = 20.0, hfa: float = 65.0):
        self.base_rating = base_rating
        self.k_factor = k_factor
        self.hfa = hfa
        self.ratings: Dict[str, float] = {}

    def get_rating(self, team: str) -> float:
        return self.ratings.get(team, self.base_rating)

    def set_rating(self, team: str, rating: float):
        self.ratings[team] = rating

    def get_win_probability(self, team_rating: float, opponent_rating: float, is_home: bool = False) -> float:
        adv = self.hfa if is_home else 0.0
        
        if is_home:
            eff_team_rating = team_rating + self.hfa
            eff_opp_rating = opponent_rating
        else:
            eff_team_rating = team_rating
            eff_opp_rating = opponent_rating + self.hfa
            
        diff = eff_opp_rating - eff_team_rating
        probability = 1.0 / (1.0 + math.pow(10, diff / 400.0))
        return probability

    def update_ratings(self, team_a: str, team_b: str, winner: str, is_neutral: bool = False):
        ra = self.get_rating(team_a)
        rb = self.get_rating(team_b)
        
        prob_a = self.get_win_probability(ra, rb, is_home=not is_neutral)
        
        if winner == "TIE":
            score_a = 0.5
        elif winner == team_a:
            score_a = 1.0
        else:
            score_a = 0.0
            
        change = self.k_factor * (score_a - prob_a)
        
        self.ratings[team_a] = ra + change
        self.ratings[team_b] = rb - change

    def train(self, games: List[Dict]):
        for game in games:
            if game.get('Status') != 'Final':
                continue
            
            home = game['HomeTeam']
            away = game['AwayTeam']
            
            home_score = game.get('HomeScore', 0) or 0
            away_score = game.get('AwayScore', 0) or 0
            
            if home_score > away_score:
                result = "HOME"
            elif away_score > home_score:
                result = "AWAY"
            else:
                result = "TIE"
                
            h_to = game.get('home_turnovers', 0)
            a_to = game.get('away_turnovers', 0)
            
            self._update_single_game(home, away, result, h_to, a_to)

    def _update_single_game(self, home_team: str, away_team: str, result: str, home_turnovers=0, away_turnovers=0):
        ra = self.ratings.get(home_team, self.base_rating)
        rb = self.ratings.get(away_team, self.base_rating)
        
        ea = 1.0 / (1.0 + 10 ** ((rb - (ra + self.hfa)) / 400.0))
        
        if result == "HOME":
            sa = 1.0
        elif result == "AWAY":
            sa = 0.0
        else:
            sa = 0.5
            
        multiplier = 1.0
        
        if sa == 1.0 and home_turnovers > away_turnovers:
            multiplier = 0.5
        elif sa == 0.0 and away_turnovers > home_turnovers:
            multiplier = 0.5
        elif sa == 0.0 and home_turnovers > away_turnovers:
             multiplier = 0.7
        elif sa == 1.0 and away_turnovers > home_turnovers:
             multiplier = 0.7
            
        change = self.k_factor * (sa - ea) * multiplier
        
        self.ratings[home_team] = ra + change
        self.ratings[away_team] = rb - change
