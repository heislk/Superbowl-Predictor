import numpy as np
import sys
from typing import Dict, List, Tuple
from src.models.srs import SRSModel
from src.models.power import PowerRatingModel
from src.models.pythagorean import PythagoreanModel
from src.models.recent_form import RecentFormModel
from src.models.qb_elo import QBEloModel
from src.models.enhanced_statistical import EnhancedStatisticalModel
from src.models.championship import ChampionshipPredictor

class SuperBowl2025Predictor:
    
    def __init__(self, elo_model, epa_model, srs_model, power_model, pyth_model, 
                 form_model, qb_model, enhanced_model, champ_model, primary_qbs):
        self.elo_model = elo_model
        self.epa_model = epa_model
        self.srs_model = srs_model
        self.power_model = power_model
        self.pyth_model = pyth_model
        self.form_model = form_model
        self.qb_model = qb_model
        self.enhanced_model = enhanced_model
        self.champ_model = champ_model
        self.primary_qbs = primary_qbs
        
        self.playoff_hfa = 30
        
        self.afc_teams = {}
        self.nfc_teams = {}
    
    def determine_playoff_teams(self, games):
        wins = {}
        
        for game in games:
            if game.get('Status') != 'Final':
                continue
            
            home = game['HomeTeam']
            away = game['AwayTeam']
            
            if home not in wins:
                wins[home] = {'wins': 0, 'losses': 0, 'pf': 0, 'pa': 0}
            if away not in wins:
                wins[away] = {'wins': 0, 'losses': 0, 'pf': 0, 'pa': 0}
            
            if game['HomeScore'] > game['AwayScore']:
                wins[home]['wins'] += 1
                wins[away]['losses'] += 1
            else:
                wins[away]['wins'] += 1
                wins[home]['losses'] += 1
            
            wins[home]['pf'] += game['HomeScore']
            wins[home]['pa'] += game['AwayScore']
            wins[away]['pf'] += game['AwayScore']
            wins[away]['pa'] += game['HomeScore']
        
        divisions = {
            'AFC_East': ['BUF', 'MIA', 'NE', 'NYJ'],
            'AFC_North': ['BAL', 'CIN', 'CLE', 'PIT'],
            'AFC_South': ['HOU', 'IND', 'JAX', 'TEN'],
            'AFC_West': ['DEN', 'KC', 'LV', 'LAC'],
            'NFC_East': ['DAL', 'NYG', 'PHI', 'WAS'],
            'NFC_North': ['CHI', 'DET', 'GB', 'MIN'],
            'NFC_South': ['ATL', 'CAR', 'NO', 'TB'],
            'NFC_West': ['ARI', 'LA', 'SF', 'SEA'],
        }
        
        afc_divs = ['AFC_East', 'AFC_North', 'AFC_South', 'AFC_West']
        nfc_divs = ['NFC_East', 'NFC_North', 'NFC_South', 'NFC_West']
        
        def get_div_winner(div_name):
            teams = divisions[div_name]
            team_records = [(t, wins.get(t, {'wins': 0})['wins']) for t in teams]
            team_records.sort(key=lambda x: -x[1])
            return team_records[0][0]
        
        def get_wild_cards(div_list, div_winners, n=3):
            all_teams = []
            for div in div_list:
                for team in divisions[div]:
                    if team not in div_winners:
                        all_teams.append((team, wins.get(team, {'wins': 0})['wins']))
            all_teams.sort(key=lambda x: -x[1])
            return [t[0] for t in all_teams[:n]]
        
        afc_div_winners = [get_div_winner(d) for d in afc_divs]
        afc_div_winners_sorted = sorted(afc_div_winners, key=lambda t: -wins.get(t, {'wins': 0})['wins'])
        afc_wild_cards = get_wild_cards(afc_divs, afc_div_winners)
        
        afc_playoff = afc_div_winners_sorted + afc_wild_cards
        self.afc_teams = {i+1: team for i, team in enumerate(afc_playoff)}
        
        nfc_div_winners = [get_div_winner(d) for d in nfc_divs]
        nfc_div_winners_sorted = sorted(nfc_div_winners, key=lambda t: -wins.get(t, {'wins': 0})['wins'])
        nfc_wild_cards = get_wild_cards(nfc_divs, nfc_div_winners)
        
        nfc_playoff = nfc_div_winners_sorted + nfc_wild_cards
        self.nfc_teams = {i+1: team for i, team in enumerate(nfc_playoff)}
        
        return wins
    
    def win_probability(self, team_a: str, team_b: str, neutral: bool = False) -> float:
        elo_a = self.elo_model.get_rating(team_a)
        elo_b = self.elo_model.get_rating(team_b)
        if not neutral: elo_a += self.playoff_hfa
        prob_elo = 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))
        
        prob_epa = self.epa_model.get_win_probability(team_a, team_b, is_home=not neutral)
        prob_srs = self.srs_model.get_win_probability(team_a, team_b, is_home=not neutral)
        prob_power = self.power_model.get_win_probability(team_a, team_b, is_home=not neutral)
        prob_pyth = self.pyth_model.get_win_probability(team_a, team_b, is_home=not neutral)
        
        qb_a = self.primary_qbs.get(team_a)
        qb_b = self.primary_qbs.get(team_b)
        if qb_a and qb_b:
            prob_qb = self.qb_model.get_win_probability(qb_a, qb_b, is_home=not neutral)
        else:
            prob_qb = 0.5
            
        prob_form = self.form_model.get_win_probability(team_a, team_b, is_home=not neutral)
        
        mock_game = {
            'HomeTeam': team_a, 'AwayTeam': team_b, 'Status': 'Final',
            'Season': 2025, 'Week': 22, 
            'spread_line': 0.0,
            'home_qb_name': qb_a, 'away_qb_name': qb_b,
            'HomeRest': 7, 'AwayRest': 7,
            'roof': 'dome' if neutral else 'outdoors' 
        }
        
        prob_enhanced = self.enhanced_model.predict(
            mock_game, self.elo_model, self.qb_model, self.epa_model, self.form_model
        )
        
        prob_champ = self.champ_model.predict(
            mock_game, self.elo_model, self.qb_model, self.epa_model, self.form_model
        )
        
        final_prob = (
            (prob_elo * 0.15) +
            (prob_epa * 0.05) +
            (prob_srs * 0.15) +
            (prob_power * 0.10) +
            (prob_pyth * 0.10) +
            (prob_qb * 0.15) +
            (prob_form * 0.05) +
            (prob_enhanced * 0.15) +
            (prob_champ * 0.10)
        )
        
        return final_prob
    
    def simulate_game(self, team_a: str, team_b: str, neutral: bool = False) -> str:
        prob_a = self.win_probability(team_a, team_b, neutral)
        return team_a if np.random.random() < prob_a else team_b
    
    def simulate_conference_playoffs(self, teams: Dict[int, str]) -> str:
        wc_winners = []
        wc_winners.append(self.simulate_game(teams[2], teams[7]))
        wc_winners.append(self.simulate_game(teams[3], teams[6]))
        wc_winners.append(self.simulate_game(teams[4], teams[5]))
        
        def get_seed(team):
            for s, t in teams.items():
                if t == team:
                    return s
            return 8
        
        remaining = [(teams[1], 1)] + [(w, get_seed(w)) for w in wc_winners]
        remaining.sort(key=lambda x: x[1])
        
        div_winner_1 = self.simulate_game(remaining[0][0], remaining[3][0])
        div_winner_2 = self.simulate_game(remaining[1][0], remaining[2][0])
        
        seed_1 = get_seed(div_winner_1)
        seed_2 = get_seed(div_winner_2)
        if seed_1 < seed_2:
            return self.simulate_game(div_winner_1, div_winner_2)
        else:
            return self.simulate_game(div_winner_2, div_winner_1)
    
    def simulate_super_bowl(self, n_simulations: int = 10000) -> Dict[str, float]:
        all_teams = list(self.afc_teams.values()) + list(self.nfc_teams.values())
        results = {team: 0 for team in all_teams}
        
        for i in range(n_simulations):
            afc_champ = self.simulate_conference_playoffs(self.afc_teams)
            nfc_champ = self.simulate_conference_playoffs(self.nfc_teams)
            sb_winner = self.simulate_game(afc_champ, nfc_champ, neutral=True)
            results[sb_winner] += 1
            
            if (i + 1) % 10 == 0:
                progress = (i + 1) / n_simulations * 100
                sys.stdout.write(f"\rSimulating: [{int(progress/2) * '=':<50}] {progress:.1f}%")
                sys.stdout.flush()
        
        sys.stdout.write("\n")
        return {team: count / n_simulations for team, count in results.items()}
    
    def predict(self, games, n_simulations: int = 50000):
        wins = self.determine_playoff_teams(games)
        
        print("\n" + "="*70)
        print("SUPER BOWL LX PREDICTION (2025 SEASON)")
        print("="*70)
        print("Running Multi-Model Ensemble Simulation (9 Models)")
        print("="*70 + "\n")
        
        print("AFC PLAYOFF TEAMS:")
        for seed, team in self.afc_teams.items():
            w = wins.get(team, {'wins': 0, 'losses': 0})
            elo = self.elo_model.get_rating(team)
            print(f"  #{seed}: {team} ({w['wins']}-{w['losses']}) - Elo: {elo:.0f}")
        
        print("\nNFC PLAYOFF TEAMS:")
        for seed, team in self.nfc_teams.items():
            w = wins.get(team, {'wins': 0, 'losses': 0})
            elo = self.elo_model.get_rating(team)
            print(f"  #{seed}: {team} ({w['wins']}-{w['losses']}) - Elo: {elo:.0f}")
        
        print("\n" + "-"*70)
        print(f"Running {n_simulations:,} playoff simulations...")
        print("-"*70 + "\n")
        
        probs = self.simulate_super_bowl(n_simulations)
        sorted_probs = sorted(probs.items(), key=lambda x: -x[1])
        
        print("SUPER BOWL WIN PROBABILITY:\n")
        print(f"{'Rank':>4}  {'Team':>4}  {'Prob':>8}  {'Odds':>8}")
        print("-" * 35)
        
        for i, (team, prob) in enumerate(sorted_probs, 1):
            if prob > 0.001:
                odds = f"+{int(100/prob - 100)}" if prob < 0.5 else f"-{int(100*prob/(1-prob))}"
                print(f"{i:4d}  {team:>4}  {prob*100:7.1f}%  {odds:>8}")
        
        winner = sorted_probs[0][0]
        win_prob = sorted_probs[0][1]
        
        print("\n" + "="*70)
        print(f"PREDICTION: {winner} wins Super Bowl LX")
        print(f"Probability: {win_prob*100:.1f}%")
        print("="*70)
        
        return sorted_probs


def get_primary_qbs(games):
    qbs = {}
    sorted_games = sorted(games, key=lambda x: x['Week'], reverse=True)
    for g in sorted_games:
        h = g['HomeTeam']
        a = g['AwayTeam']
        hq = g.get('home_qb_name')
        aq = g.get('away_qb_name')
        if h not in qbs and hq: qbs[h] = hq
        if a not in qbs and aq: qbs[a] = aq
    return qbs

def predict_super_bowl_2025():
    from src.data.client import NFLVerseClient
    from src.models.elo import EloModel
    from src.models.epa import EPAModel
    
    client = NFLVerseClient()
    schedule_2025 = client.get_schedules(2025)
    completed = [g for g in schedule_2025 if g['Status'] == 'Final']
    
    print("Training models...")
    elo = EloModel(k_factor=50, hfa=40)
    elo.train(completed)
    
    epa = EPAModel()
    epa.train(completed)
    
    srs = SRSModel()
    srs.train(completed)
    
    power = PowerRatingModel()
    power.train(completed)
    
    pyth = PythagoreanModel()
    pyth.train(completed)
    
    form = RecentFormModel()
    form.train(completed)
    
    qb = QBEloModel()
    qb.train(completed)
    
    enhanced = EnhancedStatisticalModel()
    enhanced.train(completed, elo, qb, epa, form)
    
    champ = ChampionshipPredictor()
    champ.train(completed, elo, qb, epa, form)
    
    primary_qbs = get_primary_qbs(completed)
    
    predictor = SuperBowl2025Predictor(
        elo, epa, srs, power, pyth, form, qb, enhanced, champ, primary_qbs
    )
    return predictor.predict(completed, n_simulations=10000)


if __name__ == "__main__":
    predict_super_bowl_2025()
