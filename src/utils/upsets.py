class UpsetDetector:
    def __init__(self):
        self.divisions = {
            'BUF': 'AFC_East', 'MIA': 'AFC_East', 'NE': 'AFC_East', 'NYJ': 'AFC_East',
            'BAL': 'AFC_North', 'CIN': 'AFC_North', 'CLE': 'AFC_North', 'PIT': 'AFC_North',
            'HOU': 'AFC_South', 'IND': 'AFC_South', 'JAX': 'AFC_South', 'TEN': 'AFC_South',
            'DEN': 'AFC_West', 'KC': 'AFC_West', 'LV': 'AFC_West', 'LAC': 'AFC_West',
            'DAL': 'NFC_East', 'NYG': 'NFC_East', 'PHI': 'NFC_East', 'WAS': 'NFC_East',
            'CHI': 'NFC_North', 'DET': 'NFC_North', 'GB': 'NFC_North', 'MIN': 'NFC_North',
            'ATL': 'NFC_South', 'CAR': 'NFC_South', 'NO': 'NFC_South', 'TB': 'NFC_South',
            'ARI': 'NFC_West', 'LAR': 'NFC_West', 'SF': 'NFC_West', 'SEA': 'NFC_West'
        }

    def is_trap_game(self, home_team, away_team, vegas_line, home_rest, away_rest, week):
        risk_score = 0
        
        is_div = self.divisions.get(home_team) == self.divisions.get(away_team)
        
        if is_div:
            risk_score += 1
            if vegas_line < -3.0:
                risk_score += 1.5
        
        if vegas_line < 0 and home_rest < away_rest:
            risk_score += 1
            if home_rest <= 5:
                risk_score += 2
        
        if week <= 3:
            risk_score += 1
            
        if risk_score >= 2.5:
            return True
            
        return False
