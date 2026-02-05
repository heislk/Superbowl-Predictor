import argparse
import sys
import logging
import random
from datetime import datetime
from src.config import DEFAULT_SEASON, SIMULATION_RUNS
from src.data.client import NFLVerseClient
from src.data import storage
from src.models.elo import EloModel
from src.models.pythagorean import PythagoreanModel
from src.models.srs import SRSModel
from src.models.recent_form import RecentFormModel
from src.models.power import PowerRatingModel
from src.models.qb_elo import QBEloModel
from src.models.hfa import DynamicHFAModel
from src.models.bias import BiasModel
from src.models.epa import EPAModel
from src.utils.upsets import UpsetDetector
from src.models.predictor import GamePredictor
from src.simulation.engine import SeasonSimulator
from src.simulation.evaluator import Evaluator

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def print_predictions(games, predictor):
    print("\n=== GAME PREDICTIONS (Ensemble + Vegas) ===")
    print(f"Model Spread vs Vegas Line analysis")
    print("-" * 125)
    print(f"{'Matchup':<25} | {'Pred Score':<15} | {'Win %':<8} | {'Model':<8} | {'Vegas':<8} | {'Value':<8} | {'Ratings'}")
    print("-" * 125)
    
    for game in games:
        home = game['HomeTeam']
        away = game['AwayTeam']
        h_rest = game.get('HomeRest', 7)
        a_rest = game.get('AwayRest', 7)
        h_qb = game.get('home_qb_name')
        a_qb = game.get('away_qb_name')
        
        vegas_line = game.get('spread_line', 0.0)
        
        if h_rest is None: h_rest = 7
        if a_rest is None: a_rest = 7
        
        pred = predictor.predict_matchup(home, away, is_neutral=False, 
                                         home_rest=int(h_rest), away_rest=int(a_rest),
                                         home_qb=h_qb, away_qb=a_qb,
                                         vegas_line=float(vegas_line))
        p_home = pred['HomeWinProbability'] * 100
        spread = pred['EstimatedSpread']
        score_home = pred['PredictedHomeScore']
        score_away = pred['PredictedAwayScore']
        
        matchup_str = f"{away} @ {home}"
        score_str = f"{away} {score_away} - {home} {score_home}"
        
        model_str = f"{spread:.1f}"
        vegas_str = f"{vegas_line:.1f}"
        
        diff = spread - vegas_line
        value_str = f"{diff:.1f}"
        
        if p_home > 50:
            win_prob_str = f"{home} {p_home:.0f}%"
        else:
            win_prob_str = f"{away} {100-p_home:.0f}%"
            
        rating_info = f"{pred['HomeRating']}/{pred['AwayRating']}"
            
        print(f"{matchup_str:<25} | {score_str:<15} | {win_prob_str:<8} | {model_str:<8} | {vegas_str:<8} | {value_str:<8} | {rating_info}")
    
    print("-" * 125)

    print("\n=== CONTRARIAN PICKS (Model Disagrees with Vegas) ===")
    print("-" * 80)
    print(f"{'Matchup':<25} | {'Vegas Fav':<15} | {'Model Pick':<15} | {'Value'}")
    print("-" * 80)
    
    found_contrarian = False
    for game in games:
        home = game['HomeTeam']
        away = game['AwayTeam']
        
        vegas_line = game.get('spread_line', 0.0)
        
        h_rest = game.get('HomeRest', 7)
        if h_rest is None: h_rest = 7
        a_rest = game.get('AwayRest', 7)
        if a_rest is None: a_rest = 7
        h_qb = game.get('home_qb_name')
        a_qb = game.get('away_qb_name')

        pred = predictor.predict_matchup(home, away, is_neutral=False, 
                                         home_rest=int(h_rest), away_rest=int(a_rest),
                                         home_qb=h_qb, away_qb=a_qb,
                                         vegas_line=float(vegas_line))
        spread = pred['EstimatedSpread']
        
        if spread < 0 and vegas_line > 0:
            found_contrarian = True
            vegas_fav = f"{away} ({-vegas_line:.1f})"
            model_pick = f"{home} ({-spread:.1f})"
            diff = abs(spread - vegas_line)
            print(f"{away} @ {home:<21} | {vegas_fav:<15} | {model_pick:<15} | {diff:.1f}")
            
        elif spread > 0 and vegas_line < 0:
            found_contrarian = True
            vegas_fav = f"{home} ({vegas_line:.1f})"
            model_pick = f"{away} ({spread:.1f})"
            diff = abs(spread - vegas_line)
            print(f"{away} @ {home:<21} | {vegas_fav:<15} | {model_pick:<15} | {diff:.1f}")

    if not found_contrarian:
        print("No disagreements found for this week.")
    print("-" * 80)

def run_backtest(schedule, teams, model_type='ensemble'):
    print("\n=== RUNNING SEASON BACKTEST ===")
    print("Simulating season week-by-week (No Future Knowledge)...")
    
    weeks = sorted(list(set(g['Week'] for g in schedule if g['Status'] == 'Final' and g['Season'] == DEFAULT_SEASON)))
    
    total_games = 0
    correct_picks = 0
    ae_spread = 0
    spread_count = 0
    
    contrarian_wins = 0
    contrarian_losses = 0
    
    for week in weeks:
        print(f"Testing Week {week}...", end='\r')
        
        train_data = [g for g in schedule if g['Week'] < week and g['Status'] == 'Final']
        
        elo = EloModel(k_factor=50, hfa=40)
        pyth = PythagoreanModel()
        srs = SRSModel()
        form = RecentFormModel()
        power = PowerRatingModel()
        qb = QBEloModel()
        hfa = DynamicHFAModel()
        epa = EPAModel()
        
        elo.train(train_data)
        pyth.train(train_data)
        srs.train(train_data)
        form.train(train_data)
        power.train(train_data)
        qb.train(train_data)
        hfa.train(train_data)
        epa.train(train_data)
        
        predictor = GamePredictor(elo, pyth, srs, form, power, qb, hfa, epa_model=epa, upset_detector=UpsetDetector())
        
        week_games = [g for g in schedule if g['Week'] == week and g['Status'] == 'Final']
        
        for g in week_games:
            home = g['HomeTeam']
            away = g['AwayTeam']
            actual_home_score = g['HomeScore']
            actual_away_score = g['AwayScore']
            actual_margin = actual_home_score - actual_away_score
            actual_winner = home if actual_home_score > actual_away_score else away
            
            h_rest = g.get('HomeRest', 7) or 7
            a_rest = g.get('AwayRest', 7) or 7
            h_qb = g.get('home_qb_name')
            a_qb = g.get('away_qb_name')
            
            vegas_val = g.get('spread_line', 0.0)
            p = predictor.predict_matchup(home, away, home_rest=int(h_rest), away_rest=int(a_rest),
                                          home_qb=h_qb, away_qb=a_qb,
                                          vegas_line=float(vegas_val))
            
            pred_spread = p['EstimatedSpread']
            pred_winner = home if pred_spread < 0 else away
            
            if pred_winner == actual_winner:
                correct_picks += 1
            total_games += 1
            
            error = abs(pred_spread - (-actual_margin))
            ae_spread += error
            spread_count += 1
            
            vegas = g.get('spread_line', 0.0)
            if vegas != 0.0:
                if pred_spread < 0 and vegas > 0:
                    if actual_winner == home: contrarian_wins += 1
                    else: contrarian_losses += 1
                elif pred_spread > 0 and vegas < 0:
                    if actual_winner == away: contrarian_wins += 1
                    else: contrarian_losses += 1

    print("\n" + "-"*50)
    print(f"BACKTEST RESULTS (Weeks {min(weeks)}-{max(weeks)})")
    print("-"*50)
    print(f"Total Games Analyzed: {total_games}")
    if total_games > 0:
        print(f"Straight Up Accuracy: {correct_picks}/{total_games} ({correct_picks/total_games:.1%})")
        print(f"Avg Spread Error:     {ae_spread/spread_count:.2f} points")
    
    print("-"*50)
    print("CONTRARIAN PERFORMANCE (Model vs Vegas)")
    total_contrarian = contrarian_wins + contrarian_losses
    if total_contrarian > 0:
        print(f"Record when disagreeing: {contrarian_wins}-{contrarian_losses}")
        print(f"Win Rate: {contrarian_wins/total_contrarian:.1%}")
    else:
        print("No major disagreements found in sample.")
    print("-"*50)

def main():
    parser = argparse.ArgumentParser(description="NFL Playoff & Championship Predictor (NFLVerse)")
    parser.add_argument("--season", type=int, default=DEFAULT_SEASON, help="Season to simulate")
    parser.add_argument("--sims", type=int, default=SIMULATION_RUNS, help="Number of simulations")
    parser.add_argument("--refresh", action="store_true", help="Force refresh of data from NFLVerse")
    parser.add_argument("--week", type=int, help="Start simulation from this week (Time Travel)")
    parser.add_argument("--predict", action="store_true", help="Predict games for the specified week")
    parser.add_argument("--backtest", action="store_true", help="Run historical backtest validation")
    parser.add_argument("--model", type=str, choices=['elo', 'pyth', 'srs', 'form', 'power', 'ensemble'], default='ensemble', help="Model to use")
    
    args = parser.parse_args()
    
    client = NFLVerseClient()
    logger.info(f"Fetching data for {args.season}...")
    
    teams = client.get_teams(force_refresh=args.refresh)
    schedule = client.get_schedules(args.season, force_refresh=args.refresh)
    
    if not teams or not schedule:
        logger.error("Failed to acquire data.")
        sys.exit(1)
        
    active_team_abbrs = set([g['HomeTeam'] for g in schedule] + [g['AwayTeam'] for g in schedule])
    teams = [t for t in teams if t['Key'] in active_team_abbrs]
    
    if args.backtest:
        run_backtest(schedule, teams, args.model)
        return

    logger.info(f"Loaded {len(teams)} teams and {len(schedule)} games.")

    train_limit_week = args.week if args.week else 100 
    training_schedule = [g for g in schedule if g['Week'] < train_limit_week] if args.week else schedule
        
    elo_model = None
    pyth_model = None
    srs_model = None
    form_model = None
    power_model = None
    qb_model_inst = None
    
    is_ens = (args.model == 'ensemble')
    
    if args.model == 'elo' or is_ens:
        logger.info("Training Elo Model...")
        elo_model = EloModel(k_factor=50, hfa=40)
        elo_model.train(training_schedule)
        
    if args.model == 'pyth' or is_ens:
        logger.info("Training Pythagorean Model...")
        pyth_model = PythagoreanModel()
        pyth_model.train(training_schedule)
        
    if args.model == 'srs' or is_ens:
        logger.info("Training SRS Model...")
        srs_model = SRSModel()
        srs_model.train(training_schedule)
        
    if args.model == 'form' or is_ens:
        logger.info("Training Recent Form Model...")
        form_model = RecentFormModel()
        form_model.train(training_schedule)
        
    if args.model == 'power' or is_ens:
        logger.info("Training Power Rating Model...")
        power_model = PowerRatingModel()
        power_model.train(training_schedule)
        
    if is_ens or 'qb' in args.model:
        logger.info("Training QB Elo Model...")
        qb_model_inst = QBEloModel()
        qb_model_inst.train(training_schedule)
    
    hfa_model = None
    if is_ens: 
        logger.info("Training Dynamic HFA Model...")
        hfa_model = DynamicHFAModel()
        hfa_model.train(training_schedule)
    
    predictor = GamePredictor(
        elo_model=elo_model, 
        pyth_model=pyth_model,
        srs_model=srs_model,
        form_model=form_model,
        power_model=power_model,
        qb_model=qb_model_inst if qb_model_inst else None,
        hfa_model=hfa_model
    )

    if is_ens:
        logger.info("Using 6-Model Ensemble (Elo, SRS, Power, Pyth, QB, Form)")

    if args.predict:
        target_week = args.week if args.week else 1
        if not args.week:
             for wk in range(1, 23):
                 pending = [g for g in schedule if g['Week'] == wk and g['Status'] != 'Final']
                 if pending:
                     target_week = wk
                     break
                     
        logger.info(f"Generating Predictions for Week {target_week}...")
        week_games = [g for g in schedule if g['Week'] == target_week]
        print_predictions(week_games, predictor)
        
    else:
        logger.info(f"Starting {args.sims} simulations from Week {args.week if args.week else 'Current'}...")
        simulator = SeasonSimulator(schedule, teams, predictor)
        results = simulator.simulate(n_simulations=args.sims, start_week=args.week)
        Evaluator.aggregate_and_print(results, args.sims, simulator.teams_map)
    
if __name__ == "__main__":
    main()
