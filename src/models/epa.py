from typing import Dict, List, Any

class EPAModel:
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha 
        self.off_pass_epa: Dict[str, float] = {}
        self.off_rush_epa: Dict[str, float] = {}
        self.def_pass_epa: Dict[str, float] = {}
        self.def_rush_epa: Dict[str, float] = {}

    def train(self, games: List[Dict]):
        sorted_games = sorted(games, key=lambda x: x['Week'])
        
        for g in sorted_games:
            if g.get('Status') != 'Final': continue
            
            home = g['HomeTeam']
            away = g['AwayTeam']
            
            h_pe = g.get('home_pass_epa', 0)
            h_re = g.get('home_rush_epa', 0)
            a_pe = g.get('away_pass_epa', 0)
            a_re = g.get('away_rush_epa', 0)
            
            self._update(self.off_pass_epa, home, h_pe)
            self._update(self.off_rush_epa, home, h_re)
            self._update(self.def_pass_epa, away, h_pe)
            self._update(self.def_rush_epa, away, h_re)
            
            self._update(self.off_pass_epa, away, a_pe)
            self._update(self.off_rush_epa, away, a_re)
            self._update(self.def_pass_epa, home, a_pe)
            self._update(self.def_rush_epa, home, a_re)

    def _update(self, rating_dict, team, value):
        curr = rating_dict.get(team, 0.0) 
        new_val = (curr * (1.0 - self.alpha)) + (value * self.alpha)
        rating_dict[team] = new_val

    def get_win_probability(self, home_team: str, away_team: str, is_home=True) -> float:
        h_pass_exp = (self.off_pass_epa.get(home_team, 0) + self.def_pass_epa.get(away_team, 0)) / 2
        h_rush_exp = (self.off_rush_epa.get(home_team, 0) + self.def_rush_epa.get(away_team, 0)) / 2
        h_total_epa = h_pass_exp + h_rush_exp
        
        a_pass_exp = (self.off_pass_epa.get(away_team, 0) + self.def_pass_epa.get(home_team, 0)) / 2
        a_rush_exp = (self.off_rush_epa.get(away_team, 0) + self.def_rush_epa.get(home_team, 0)) / 2
        a_total_epa = a_pass_exp + a_rush_exp
        
        net_epa = h_total_epa - a_total_epa
        
        if is_home:
            net_epa += 2.5 
            
        elo_diff = net_epa * 25.0
        return 1.0 / (1.0 + 10 ** (-elo_diff / 400.0))
