import random
import copy
import logging
from typing import List, Dict
from src.models.predictor import GamePredictor
from src.simulation.rules import SeasonRules

logger = logging.getLogger(__name__)

class SeasonSimulator:
    def __init__(self, games: List[Dict], teams: List[Dict], predictor: GamePredictor):
        self.games = games
        self.teams_map = {t['Key']: t for t in teams}
        self.original_predictor = predictor
        
        self.divisions = {t['Key']: t.get('Conference') + ' ' + t.get('Division') for t in teams}
        self.conferences = {'AFC': [], 'NFC': []}
        for t in teams:
            conf = t.get('Conference')
            if conf in self.conferences:
                self.conferences[conf].append(t['Key'])
                
        self.primary_qbs = {}
        sorted_games = sorted(games, key=lambda x: x['Week'], reverse=True)
        for g in sorted_games:
            h = g['HomeTeam']
            a = g['AwayTeam']
            hq = g.get('home_qb_name')
            aq = g.get('away_qb_name')
            if h not in self.primary_qbs and hq: self.primary_qbs[h] = hq
            if a not in self.primary_qbs and aq: self.primary_qbs[a] = aq
            
    def simulate(self, n_simulations: int = 1000, start_week: int = None) -> Dict:
        logger.info(f"Starting {n_simulations} simulations from Week {start_week if start_week else 'Current'}...")
        
        completed_games = []
        pending_games = []
        
        for game in self.games:
            is_historical = False
            if start_week is not None:
                if game['Week'] < start_week:
                    is_historical = True
            else:
                if game.get('Status') == 'Final':
                    is_historical = True
            
            if is_historical:
                completed_games.append(game)
            else:
                pending_games.append(game)
                
        base_standings = {ticker: {'Wins': 0, 'Losses': 0, 'Ties': 0} for ticker in self.teams_map}
        
        for game in completed_games:
            home = game.get('HomeTeam')
            away = game.get('AwayTeam')
            h_score = game.get('HomeScore', 0)
            a_score = game.get('AwayScore', 0)
            
            winner = "TIE"
            if h_score > a_score: winner = home
            elif a_score > h_score: winner = away
            
            SeasonRules.update_standings(base_standings, home, away, winner)
        
        team_results = {t: {'MadePlayoffs': 0, 'WonDivision': 0, 'WonSuperBowl': 0, 'SeedCounts': {}} for t in self.teams_map}

        for i in range(n_simulations):
            self._run_single_simulation(base_standings, pending_games, team_results)
            
        return team_results

    def _run_single_simulation(self, base_standings: Dict, pending_games: List[Dict], results: Dict):
        current_standings = copy.deepcopy(base_standings)
        
        has_elo = self.original_predictor.elo_model is not None
        has_pyth = self.original_predictor.pyth_model is not None
        has_srs = self.original_predictor.srs_model is not None
        has_form = self.original_predictor.form_model is not None
        has_power = self.original_predictor.power_model is not None
        has_qb = self.original_predictor.qb_model is not None
        
        current_ratings = {}
        if has_elo:
            current_ratings = self.original_predictor.elo_model.ratings.copy()

        def get_r(team): 
            if has_elo:
                 return current_ratings.get(team, self.original_predictor.elo_model.base_rating)
            return 1500.0

        def update_r(team, new_r):
            if has_elo:
                current_ratings[team] = new_r
            
        def get_prob(home, away, h_qb=None, a_qb=None):
            weights = self.original_predictor.weights
            total_prob = 0.0
            
            if has_elo:
                 ra = get_r(home)
                 rb = get_r(away)
                 diff = (rb) - (ra + 65) 
                 p_elo = 1.0 / (1.0 + 10 ** (diff / 400.0))
                 total_prob += p_elo * weights.get('elo', 0)
            
            if has_pyth:
                total_prob += self.original_predictor.pyth_model.get_win_probability(home, away, is_home=True) * weights.get('pyth', 0)
            if has_srs:
                total_prob += self.original_predictor.srs_model.get_win_probability(home, away, is_home=True) * weights.get('srs', 0)
            if has_form:
                total_prob += self.original_predictor.form_model.get_win_probability(home, away, is_home=True) * weights.get('form', 0)
            if has_power:
                total_prob += self.original_predictor.power_model.get_win_probability(home, away, is_home=True) * weights.get('power', 0)
            
            if has_qb:
                if h_qb and a_qb:
                    p_qb = self.original_predictor.qb_model.get_win_probability(h_qb, a_qb, is_home=True)
                else:
                    p_qb = 0.5
                total_prob += p_qb * weights.get('qb', 0)
                
            return total_prob

        for game in pending_games:
            home = game['HomeTeam']
            away = game['AwayTeam']
            h_qb = game.get('home_qb_name')
            a_qb = game.get('away_qb_name')
            
            p_home = get_prob(home, away, h_qb, a_qb)
            
            if random.random() < p_home:
                winner = home
                if has_elo:
                    change = 20 * (1.0 - p_home) 
                    ra = get_r(home)
                    rb = get_r(away)
                    update_r(home, ra + change)
                    update_r(away, rb - change)
            else:
                winner = away
                if has_elo:
                    change = 20 * (0.0 - p_home)
                    ra = get_r(home)
                    rb = get_r(away)
                    update_r(home, ra + change)
                    update_r(away, rb - change)
            
            SeasonRules.update_standings(current_standings, home, away, winner)
            
        seeds = SeasonRules.determine_seeds(current_standings, self.divisions, self.conferences)
        
        for conf, seed_list in seeds.items():
            for rank, team in enumerate(seed_list, 1):
                if team in results:
                    results[team]['MadePlayoffs'] += 1
                    results[team]['SeedCounts'][rank] = results[team]['SeedCounts'].get(rank, 0) + 1
                    if rank <= 4:
                         results[team]['WonDivision'] += 1
                         
        sb_winner = self._simulate_playoffs(seeds, current_ratings, get_prob)
        if sb_winner and sb_winner in results:
            results[sb_winner]['WonSuperBowl'] += 1

    def _simulate_playoffs(self, seeds: Dict, ratings: Dict, prob_func) -> str:
        
        def sim_game(home, away):
             h_qb = self.primary_qbs.get(home)
             a_qb = self.primary_qbs.get(away)
             
             p = prob_func(home, away, h_qb, a_qb) 
             return home if random.random() < p else away

        conf_winners = []
        
        for conf in ['AFC', 'NFC']:
            conf_seeds = seeds[conf]
            if len(conf_seeds) < 7: continue 
            
            winner_2_7 = sim_game(conf_seeds[1], conf_seeds[6])
            winner_3_6 = sim_game(conf_seeds[2], conf_seeds[5])
            winner_4_5 = sim_game(conf_seeds[3], conf_seeds[4])
            
            team_to_seed = {t: i+1 for i, t in enumerate(conf_seeds)}
            survivors = [winner_2_7, winner_3_6, winner_4_5]
            survivors.sort(key=lambda t: team_to_seed[t], reverse=True)
            
            lowest_seed = survivors[0]
            other_two = survivors[1:]
            
            div_winner_1 = sim_game(conf_seeds[0], lowest_seed)
            
            home_div = other_two[1] if team_to_seed[other_two[1]] < team_to_seed[other_two[0]] else other_two[0]
            away_div = other_two[0] if home_div == other_two[1] else other_two[1]
            div_winner_2 = sim_game(home_div, away_div)
            
            finalists = [div_winner_1, div_winner_2]
            home_conf = finalists[0] if team_to_seed[finalists[0]] < team_to_seed[finalists[1]] else finalists[1]
            away_conf = finalists[1] if home_conf == finalists[0] else finalists[0]
            
            conf_champ = sim_game(home_conf, away_conf)
            conf_winners.append(conf_champ)
            
        if len(conf_winners) == 2:
            h_qb = self.primary_qbs.get(conf_winners[0])
            a_qb = self.primary_qbs.get(conf_winners[1])
            p = prob_func(conf_winners[0], conf_winners[1], h_qb, a_qb)
            return conf_winners[0] if random.random() < p else conf_winners[1]
            
        return None
