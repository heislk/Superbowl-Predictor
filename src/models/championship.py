import numpy as np
from typing import Dict, List
from sklearn.ensemble import RandomForestClassifier

class ChampionshipPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            n_jobs=1
        )
        self.trained = False
        self.feature_names = []
        
        self.divisions = {
            'BUF': 'AFC_East', 'MIA': 'AFC_East', 'NE': 'AFC_East', 'NYJ': 'AFC_East',
            'BAL': 'AFC_North', 'CIN': 'AFC_North', 'CLE': 'AFC_North', 'PIT': 'AFC_North',
            'HOU': 'AFC_South', 'IND': 'AFC_South', 'JAX': 'AFC_South', 'TEN': 'AFC_South',
            'DEN': 'AFC_West', 'KC': 'AFC_West', 'LV': 'AFC_West', 'LAC': 'AFC_West',
            'DAL': 'NFC_East', 'NYG': 'NFC_East', 'PHI': 'NFC_East', 'WAS': 'NFC_East',
            'CHI': 'NFC_North', 'DET': 'NFC_North', 'GB': 'NFC_North', 'MIN': 'NFC_North',
            'ATL': 'NFC_South', 'CAR': 'NFC_South', 'NO': 'NFC_South', 'TB': 'NFC_South',
            'ARI': 'NFC_West', 'LAR': 'NFC_West', 'LA': 'NFC_West', 'SF': 'NFC_West', 'SEA': 'NFC_West'
        }
        self.dome_teams = {'ATL', 'NO', 'DET', 'MIN', 'LV', 'LAR', 'LA', 'ARI', 'DAL', 'HOU', 'IND'}
        
    def extract_features(self, game: Dict, elo_model, qb_model, epa_model, form_model,
                        for_prediction=False) -> np.ndarray:
        home = game['HomeTeam']
        away = game['AwayTeam']
        week = game.get('Week', 1)
        
        features = []
        
        if not self.feature_names:
            self.feature_names = [
                'vegas_line', 'vegas_close', 'vegas_medium',
                'elo_diff', 'qb_diff', 'epa_diff', 'epa_weighted',
                'rest_diff', 'short_rest_home', 'short_rest_away',
                'form_diff', 'division_game', 'dome_outdoors',
                'week_num', 'early_season', 'late_season',
                'rest_x_home', 'div_x_late', 'epa_x_elo',
                'vegas_x_rest', 'close_x_div',
                'turnover_margin'
            ]
        
        vegas = game.get('spread_line', 0.0)
        features.extend([
            vegas,
            1 if abs(vegas) < 3.0 else 0,
            1 if 3.0 <= abs(vegas) < 7.0 else 0
        ])
        
        h_elo = elo_model.get_rating(home)
        a_elo = elo_model.get_rating(away)
        elo_diff = h_elo - a_elo
        features.append(elo_diff)
        
        h_qb = game.get('home_qb_name')
        a_qb = game.get('away_qb_name')
        qb_diff = 0.0
        if h_qb and a_qb:
            qb_diff = qb_model.get_rating(h_qb) - qb_model.get_rating(a_qb)
        features.append(qb_diff)
        
        h_rush_epa = game.get('home_rush_epa', 0.0)
        h_pass_epa = game.get('home_pass_epa', 0.0)
        a_rush_epa = game.get('away_rush_epa', 0.0)
        a_pass_epa = game.get('away_pass_epa', 0.0)
        
        epa_diff = (h_rush_epa + h_pass_epa) - (a_rush_epa + a_pass_epa)
        
        epa_confidence = min(1.0, max(0.0, (week - 1) / 9.0))
        epa_weighted = epa_diff * epa_confidence
        
        features.extend([epa_diff, epa_weighted])
        
        h_rest = game.get('HomeRest', 7)
        a_rest = game.get('AwayRest', 7)
        rest_diff = h_rest - a_rest
        features.extend([
            rest_diff,
            1 if h_rest <= 5 else 0,
            1 if a_rest <= 5 else 0
        ])
        
        if form_model:
            h_prob = form_model.get_win_probability(home, away, is_home=True)
            features.append(h_prob - 0.5)
        else:
            features.append(0.0)
        
        is_div = self.divisions.get(home) == self.divisions.get(away)
        roof = game.get('roof', 'unknown')
        features.extend([
            1 if is_div else 0,
            1 if home in self.dome_teams and roof == 'outdoors' else 0
        ])
        
        features.extend([
            week / 18.0,
            1 if week <= 4 else 0,
            1 if week >= 13 else 0
        ])
        
        features.extend([
            rest_diff * (1 if vegas < 0 else -1),
            (1 if is_div else 0) * (1 if week >= 13 else 0),
            epa_weighted * (elo_diff / 400.0),
            vegas * rest_diff,
            (1 if abs(vegas) < 3 else 0) * (1 if is_div else 0)
        ])
        
        if for_prediction:
            features.append(0.0)
        else:
            h_to = game.get('home_turnovers', 0)
            a_to = game.get('away_turnovers', 0)
            features.append(a_to - h_to)
        
        return np.array(features)
    
    def train(self, games: List[Dict], elo_model, qb_model, epa_model, form_model):
        X = []
        y = []
        
        for game in games:
            if game.get('Status') != 'Final':
                continue
            
            features = self.extract_features(game, elo_model, qb_model, epa_model, form_model, for_prediction=False)
            X.append(features)
            
            y.append(1 if game['HomeScore'] > game['AwayScore'] else 0)
        
        if len(X) == 0:
            return
        
        X = np.array(X)
        y = np.array(y)
        
        self.model.fit(X, y)
        self.trained = True
    
    def predict(self, game: Dict, elo_model, qb_model, epa_model, form_model) -> float:
        if not self.trained:
            return 0.5
        
        features = self.extract_features(game, elo_model, qb_model, epa_model, form_model, for_prediction=True)
        X = features.reshape(1, -1)
        
        prob = self.model.predict_proba(X)[0][1]
        return prob
    
    def get_feature_importance(self) -> Dict[str, float]:
        if not self.trained:
            return {}
        
        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance))
