from typing import Dict, List, Any

class PowerRatingModel:
    def __init__(self):
        self.off_ratings: Dict[str, float] = {}
        self.def_ratings: Dict[str, float] = {}
        self.league_avg_score = 22.0

    def train(self, games: List[Dict], iterations: int = 10):
        scores_for = {}
        scores_allowed = {}
        opponents = {}
        
        all_scores = []
        
        for game in games:
            if game.get('Status') != 'Final': continue
            home = game['HomeTeam']
            away = game['AwayTeam']
            h_score = game.get('HomeScore', 0)
            a_score = game.get('AwayScore', 0)
            
            all_scores.extend([h_score, a_score])
            
            if home not in scores_for:
                scores_for[home] = []
                scores_allowed[home] = []
                opponents[home] = []
            if away not in scores_for:
                scores_for[away] = []
                scores_allowed[away] = []
                opponents[away] = []
                
            scores_for[home].append(h_score)
            scores_allowed[home].append(a_score)
            opponents[home].append(away)
            
            scores_for[away].append(a_score)
            scores_allowed[away].append(h_score)
            opponents[away].append(home)
            
        if all_scores:
            self.league_avg_score = sum(all_scores) / len(all_scores)
            
        for team in scores_for:
            avg_pts = sum(scores_for[team])/len(scores_for[team])
            avg_all = sum(scores_allowed[team])/len(scores_allowed[team])
            self.off_ratings[team] = avg_pts - self.league_avg_score
            self.def_ratings[team] = self.league_avg_score - avg_all
            
        for _ in range(iterations):
            new_off = {}
            new_def = {}
            
            for team in self.off_ratings:
                opps = opponents.get(team, [])
                if not opps:
                    new_off[team] = self.off_ratings[team]
                    new_def[team] = self.def_ratings[team]
                    continue
                
                avg_scored = sum(scores_for[team])/len(scores_for[team])
                avg_opp_def = sum(self.def_ratings.get(o, 0) for o in opps) / len(opps)
                
                new_off[team] = (avg_scored - self.league_avg_score) + avg_opp_def
                
                avg_allowed = sum(scores_allowed[team])/len(scores_allowed[team])
                avg_opp_off = sum(self.off_ratings.get(o, 0) for o in opps) / len(opps)
                
                new_def[team] = (self.league_avg_score - avg_allowed) + avg_opp_off
                
            self.off_ratings = new_off
            self.def_ratings = new_def

    def get_win_probability(self, team_a: str, team_b: str, is_home: bool = False) -> float:
        a_off = self.off_ratings.get(team_a, 0)
        b_def = self.def_ratings.get(team_b, 0)
        
        b_off = self.off_ratings.get(team_b, 0)
        a_def = self.def_ratings.get(team_a, 0)
        
        hfa = 2.5 if is_home else 0.0
        
        pred_a = self.league_avg_score + a_off - b_def + hfa
        pred_b = self.league_avg_score + b_off - a_def
        
        margin = pred_a - pred_b
        elo_diff_equiv = margin * 25.0
        prob = 1.0 / (1.0 + 10 ** (-elo_diff_equiv / 400.0))
        return prob
