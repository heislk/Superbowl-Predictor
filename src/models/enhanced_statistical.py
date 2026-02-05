import numpy as np
from typing import Dict, List
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

class EnhancedStatisticalModel:
    
    def __init__(self):
        self.model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        self.scaler = StandardScaler()
        self.trained = False
        
        self.h2h_wins = defaultdict(lambda: defaultdict(int))
        self.h2h_games = defaultdict(lambda: defaultdict(int))
        
        self.home_record = defaultdict(lambda: {'wins': 0, 'games': 0})
        self.away_record = defaultdict(lambda: {'wins': 0, 'games': 0})
        
        self.recent_opponent_elo = defaultdict(list)
        
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
        
    def update_context(self, games, elo_model):
        self.h2h_wins.clear()
        self.h2h_games.clear()
        self.home_record.clear()
        self.away_record.clear()
        self.recent_opponent_elo.clear()
        
        for game in games:
            if game.get('Status') != 'Final':
                continue
            
            home = game['HomeTeam']
            away = game['AwayTeam']
            home_won = game['HomeScore'] > game['AwayScore']
            
            self.h2h_games[home][away] += 1
            self.h2h_games[away][home] += 1
            if home_won:
                self.h2h_wins[home][away] += 1
            else:
                self.h2h_wins[away][home] += 1
            
            self.home_record[home]['games'] += 1
            self.away_record[away]['games'] += 1
            if home_won:
                self.home_record[home]['wins'] += 1
            else:
                self.away_record[away]['wins'] += 1
            
            away_elo = elo_model.get_rating(away)
            home_elo = elo_model.get_rating(home)
            
            self.recent_opponent_elo[home].append(away_elo)
            self.recent_opponent_elo[away].append(home_elo)
            
            if len(self.recent_opponent_elo[home]) > 3:
                self.recent_opponent_elo[home].pop(0)
            if len(self.recent_opponent_elo[away]) > 3:
                self.recent_opponent_elo[away].pop(0)
    
    def extract_features(self, game, elo_model, qb_model, epa_model, form_model, for_prediction=False):
        home = game['HomeTeam']
        away = game['AwayTeam']
        week = game.get('Week', 1)
        
        features = []
        
        vegas = game.get('spread_line', 0.0)
        features.extend([
            vegas,
            1 if abs(vegas) < 3.0 else 0,
            1 if 3.0 <= abs(vegas) < 7.0 else 0
        ])
        
        h_elo = elo_model.get_rating(home)
        a_elo = elo_model.get_rating(away)
        features.append(h_elo - a_elo)
        
        h_qb = game.get('home_qb_name')
        a_qb = game.get('away_qb_name')
        qb_diff = qb_model.get_rating(h_qb) - qb_model.get_rating(a_qb) if h_qb and a_qb else 0.0
        features.append(qb_diff)
        
        h_rush_epa = game.get('home_rush_epa', 0.0)
        h_pass_epa = game.get('home_pass_epa', 0.0)
        a_rush_epa = game.get('away_rush_epa', 0.0)
        a_pass_epa = game.get('away_pass_epa', 0.0)
        
        epa_diff = (h_rush_epa + h_pass_epa) - (a_rush_epa + a_pass_epa)
        features.append(epa_diff)
        
        h_rest = game.get('HomeRest', 7)
        a_rest = game.get('AwayRest', 7)
        features.extend([
            h_rest - a_rest,
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
        
        features.append(week / 18.0)
        
        h2h_total = self.h2h_games[home][away]
        h2h_home_wins = self.h2h_wins[home][away]
        
        if h2h_total > 0:
            h2h_win_pct = h2h_home_wins / h2h_total
            features.append(h2h_win_pct - 0.5)
            features.append(min(h2h_total / 6.0, 1.0))
        else:
            features.append(0.0)
            features.append(0.0)
        
        h_home_games = self.home_record[home]['games']
        h_home_wins = self.home_record[home]['wins']
        h_home_pct = h_home_wins / h_home_games if h_home_games > 0 else 0.5
        
        a_away_games = self.away_record[away]['games']
        a_away_wins = self.away_record[away]['wins']
        a_away_pct = a_away_wins / a_away_games if a_away_games > 0 else 0.5
        
        features.append(h_home_pct - a_away_pct)
        
        h_recent_opp = self.recent_opponent_elo[home]
        a_recent_opp = self.recent_opponent_elo[away]
        
        h_avg_opp = np.mean(h_recent_opp) if len(h_recent_opp) > 0 else 1500
        a_avg_opp = np.mean(a_recent_opp) if len(a_recent_opp) > 0 else 1500
        
        features.append((h_avg_opp - 1500) / 100.0)
        features.append((a_avg_opp - 1500) / 100.0)
        
        if is_div and h2h_total > 0:
            h2h_variance = h2h_win_pct * (1 - h2h_win_pct)
            features.append(1.0 - h2h_variance * 4.0)
        else:
            features.append(0.0)
        
        if for_prediction:
            features.append(0.0)
        else:
            h_to = game.get('home_turnovers', 0)
            a_to = game.get('away_turnovers', 0)
            features.append(a_to - h_to)
        
        return np.array(features)
    
    def train(self, games, elo_model, qb_model, epa_model, form_model):
        self.update_context(games, elo_model)
        
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
        X_scaled = self.scaler.fit_transform(X)
        
        self.model.fit(X_scaled, y)
        self.trained = True
    
    def predict(self, game, elo_model, qb_model, epa_model, form_model) -> float:
        if not self.trained:
            return 0.5
        
        features = self.extract_features(game, elo_model, qb_model, epa_model, form_model, for_prediction=True)
        X = features.reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        prob = self.model.predict_proba(X_scaled)[0][1]
        return prob
    
    def get_feature_importance(self) -> Dict[str, float]:
        if not self.trained:
            return {}
        
        feature_names = [
            'vegas_line', 'vegas_close', 'vegas_medium', 'elo_diff', 'qb_diff', 'epa_diff',
            'rest_diff', 'short_rest_home', 'short_rest_away', 'form_diff',
            'division_game', 'dome_outdoors', 'week_normalized',
            'h2h_advantage', 'h2h_confidence', 'home_away_split',
            'home_opp_strength', 'away_opp_strength', 'div_rivalry_intensity',
            'turnover_margin'
        ]
        
        coeffs = self.model.coef_[0]
        return dict(zip(feature_names, coeffs))
