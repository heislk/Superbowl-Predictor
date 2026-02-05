from typing import Dict, List, Any

class RecentFormModel:
    def __init__(self, window: int = 5):
        self.window = window
        self.ratings: Dict[str, float] = {}

    def train(self, games: List[Dict]):
        team_games = {}
        
        sorted_games = sorted(games, key=lambda x: (x['Season'], x['Week']))
        
        for game in sorted_games:
            if game.get('Status') != 'Final': continue
            
            home = game['HomeTeam']
            away = game['AwayTeam']
            h_score = game.get('HomeScore', 0)
            a_score = game.get('AwayScore', 0)
            if h_score is None: h_score = 0
            if a_score is None: a_score = 0
            
            margin = h_score - a_score
            
            if home not in team_games: team_games[home] = []
            if away not in team_games: team_games[away] = []
            
            team_games[home].append(margin)
            team_games[away].append(-margin)
            
        for team, margins in team_games.items():
            recent = margins[-self.window:]
            if not recent:
                self.ratings[team] = 0.0
                continue
                
            weighted_sum = 0
            weight_total = 0
            for i, m in enumerate(recent):
                w = i + 1
                weighted_sum += m * w
                weight_total += w
                
            self.ratings[team] = weighted_sum / weight_total

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
