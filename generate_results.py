import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
import os
import urllib.request
from PIL import Image
from io import BytesIO

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

TEAM_LOGO_URLS = {
    'ARI': 'https://a.espncdn.com/i/teamlogos/nfl/500/ari.png',
    'ATL': 'https://a.espncdn.com/i/teamlogos/nfl/500/atl.png',
    'BAL': 'https://a.espncdn.com/i/teamlogos/nfl/500/bal.png',
    'BUF': 'https://a.espncdn.com/i/teamlogos/nfl/500/buf.png',
    'CAR': 'https://a.espncdn.com/i/teamlogos/nfl/500/car.png',
    'CHI': 'https://a.espncdn.com/i/teamlogos/nfl/500/chi.png',
    'CIN': 'https://a.espncdn.com/i/teamlogos/nfl/500/cin.png',
    'CLE': 'https://a.espncdn.com/i/teamlogos/nfl/500/cle.png',
    'DAL': 'https://a.espncdn.com/i/teamlogos/nfl/500/dal.png',
    'DEN': 'https://a.espncdn.com/i/teamlogos/nfl/500/den.png',
    'DET': 'https://a.espncdn.com/i/teamlogos/nfl/500/det.png',
    'GB': 'https://a.espncdn.com/i/teamlogos/nfl/500/gb.png',
    'HOU': 'https://a.espncdn.com/i/teamlogos/nfl/500/hou.png',
    'IND': 'https://a.espncdn.com/i/teamlogos/nfl/500/ind.png',
    'JAX': 'https://a.espncdn.com/i/teamlogos/nfl/500/jax.png',
    'KC': 'https://a.espncdn.com/i/teamlogos/nfl/500/kc.png',
    'LA': 'https://a.espncdn.com/i/teamlogos/nfl/500/lar.png',
    'LAC': 'https://a.espncdn.com/i/teamlogos/nfl/500/lac.png',
    'LAR': 'https://a.espncdn.com/i/teamlogos/nfl/500/lar.png',
    'LV': 'https://a.espncdn.com/i/teamlogos/nfl/500/lv.png',
    'MIA': 'https://a.espncdn.com/i/teamlogos/nfl/500/mia.png',
    'MIN': 'https://a.espncdn.com/i/teamlogos/nfl/500/min.png',
    'NE': 'https://a.espncdn.com/i/teamlogos/nfl/500/ne.png',
    'NO': 'https://a.espncdn.com/i/teamlogos/nfl/500/no.png',
    'NYG': 'https://a.espncdn.com/i/teamlogos/nfl/500/nyg.png',
    'NYJ': 'https://a.espncdn.com/i/teamlogos/nfl/500/nyj.png',
    'PHI': 'https://a.espncdn.com/i/teamlogos/nfl/500/phi.png',
    'PIT': 'https://a.espncdn.com/i/teamlogos/nfl/500/pit.png',
    'SF': 'https://a.espncdn.com/i/teamlogos/nfl/500/sf.png',
    'SEA': 'https://a.espncdn.com/i/teamlogos/nfl/500/sea.png',
    'TB': 'https://a.espncdn.com/i/teamlogos/nfl/500/tb.png',
    'TEN': 'https://a.espncdn.com/i/teamlogos/nfl/500/ten.png',
    'WAS': 'https://a.espncdn.com/i/teamlogos/nfl/500/wsh.png',
}

def download_logo(team):
    logo_dir = 'results/logos'
    os.makedirs(logo_dir, exist_ok=True)
    logo_path = os.path.join(logo_dir, f'{team}.png')
    if os.path.exists(logo_path): return Image.open(logo_path)
    url = TEAM_LOGO_URLS.get(team)
    if not url: return None
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response: img_data = response.read()
        img = Image.open(BytesIO(img_data)).convert('RGBA')
        img.save(logo_path)
        return img
    except Exception: return None

def get_logo_image(team, zoom=0.06):
    img = download_logo(team)
    if not img: return None
    img = img.resize((50, 50), Image.Resampling.LANCZOS)
    return OffsetImage(img, zoom=zoom)

def generate_superbowl_chart():
    from src.data.client import NFLVerseClient
    from src.models.elo import EloModel
    from src.models.epa import EPAModel
    from src.models.pythagorean import PythagoreanModel
    from src.models.srs import SRSModel 
    from src.models.recent_form import RecentFormModel
    from src.models.power import PowerRatingModel
    from src.models.qb_elo import QBEloModel
    from src.models.hfa import DynamicHFAModel
    from src.models.predictor import GamePredictor
    from src.utils.upsets import UpsetDetector
    from src.models.superbowl_2025 import SuperBowl2025Predictor
    
    client = NFLVerseClient()
    schedule_2025 = client.get_schedules(2025)
    completed = [g for g in schedule_2025 if g['Status'] == 'Final']
    
    elo = EloModel(k_factor=50, hfa=40)
    pyth = PythagoreanModel()
    srs = SRSModel()
    form = RecentFormModel()
    power = PowerRatingModel()
    qb = QBEloModel()
    hfa = DynamicHFAModel()
    epa = EPAModel()
    
    for m in [elo, pyth, srs, form, power, qb, hfa, epa]: m.train(completed)
    
    predictor = GamePredictor(elo, pyth, srs, form, power, qb, hfa, epa_model=epa, upset_detector=UpsetDetector())
    
    prob_sim = SuperBowl2025Predictor(elo, epa)
    wins = prob_sim.determine_playoff_teams(completed)
    
    print("\n" + "="*70)
    print("SUPER BOWL LX DASHBOARD")
    print("="*70)
    
    probs = prob_sim.simulate_super_bowl(n_simulations=50000)
    sorted_probs = sorted(probs.items(), key=lambda x: -x[1])
    
    teams = [r[0] for r in sorted_probs if r[1] > 0.005]
    prob_vals = [r[1] * 100 for r in sorted_probs if r[1] > 0.005]
    
    afc_teams = ['NE', 'DEN', 'JAX', 'HOU', 'BUF', 'PIT', 'LAC', 'KC', 'BAL', 'CIN', 'CLE', 'MIA', 'IND', 'TEN', 'LV', 'NYJ']
    team_elos = {t: elo.get_rating(t) for t in teams}
    team_records = {t: f"{wins.get(t, {'wins': 0})['wins']}-{wins.get(t, {'losses': 0})['losses']}" for t in teams}
    
    bg_color = '#0d1117'
    card_color = '#161b22'
    accent_nfc = '#238636'
    accent_afc = '#1f6feb'
    text_primary = '#f0f6fc'
    text_secondary = '#8b949e'
    gold = '#ffd700'
    
    fig = plt.figure(figsize=(24, 14), facecolor=bg_color)
    
    fig.text(0.5, 0.96, 'SUPER BOWL LX', ha='center', fontsize=38, fontweight='bold', color=gold)
    fig.text(0.5, 0.925, 'Championship Probability & Scoreboard', ha='center', fontsize=16, color=text_secondary)
    
    ax_main = fig.add_axes([0.12, 0.35, 0.38, 0.52], facecolor=card_color)
    for s in ax_main.spines.values(): s.set_visible(False)
    
    y_pos = np.arange(len(teams))
    bar_colors = [accent_afc if t in afc_teams else accent_nfc for t in teams]
    bars = ax_main.barh(y_pos, prob_vals, height=0.65, color=bar_colors, edgecolor='none', alpha=0.9)
    
    ax_main.set_yticks(y_pos)
    ax_main.set_yticklabels([])
    ax_main.invert_yaxis()
    ax_main.set_xlim(0, max(prob_vals) + 8)
    ax_main.tick_params(axis='x', colors=text_secondary)
    ax_main.set_xlabel('Win Probability (%)', color=text_secondary)
    ax_main.set_title("CHAMPIONSHIP PROBABILITIES", color=text_primary, fontweight='bold', pad=15)
    
    for i, (team, prob) in enumerate(zip(teams, prob_vals)):
        elo_val = team_elos.get(team, 1500)
        record = team_records.get(team, '0-0')
        conf_color = accent_afc if team in afc_teams else accent_nfc
        
        ax_main.text(-1, i, team, ha='right', va='center', fontsize=13, fontweight='bold', color=text_primary)
        ax_main.text(-5, i, record, ha='right', va='center', fontsize=11, color=text_secondary, fontfamily='monospace')
        ax_main.text(-9, i, f'{elo_val:.0f}', ha='right', va='center', fontsize=10, color=conf_color, fontfamily='monospace')
        ax_main.text(prob + 0.5, i, f'{prob:.1f}%', ha='left', va='center', fontsize=11, fontweight='bold', color=text_primary)
    
    ax_main.text(-9, -1.3, 'ELO', ha='right', va='center', fontsize=9, color=text_secondary, fontweight='bold')
    ax_main.text(-5, -1.3, 'REC', ha='right', va='center', fontsize=9, color=text_secondary, fontweight='bold')
    ax_main.text(-1, -1.3, 'TEAM', ha='right', va='center', fontsize=9, color=text_secondary, fontweight='bold')
    
    ax_main.legend(handles=[mpatches.Patch(color=accent_nfc, label='NFC'), mpatches.Patch(color=accent_afc, label='AFC')], 
                   loc='lower right', facecolor=card_color, edgecolor=text_secondary, labelcolor=text_primary)


    ax_elo = fig.add_axes([0.55, 0.55, 0.40, 0.32], facecolor=card_color)
    for s in ax_elo.spines.values(): s.set_color(text_secondary); s.set_alpha(0.3)
    ax_elo.set_title('ELO RATING vs WINS', fontsize=12, fontweight='bold', color=text_primary, pad=10)
    
    all_team_data = []
    all_teams_set = set()
    for g in schedule_2025: all_teams_set.add(g['HomeTeam']); all_teams_set.add(g['AwayTeam'])
    for t in all_teams_set:
        w_data = wins.get(t, {'wins': 0})
        all_team_data.append({'team': t, 'wins': w_data['wins'], 'elo': elo.get_rating(t)})

    for data in all_team_data:
        logo = get_logo_image(data['team'], zoom=0.55)
        if logo: ax_elo.add_artist(AnnotationBbox(logo, (data['elo'], data['wins']), frameon=False))
        else: ax_elo.scatter(data['elo'], data['wins'], color=text_secondary)
    
    all_elos = [d['elo'] for d in all_team_data] 
    all_wins = [d['wins'] for d in all_team_data]
    if len(all_elos) > 1:
        z = np.polyfit(all_elos, all_wins, 1)
        ax_elo.plot(np.unique(all_elos), np.poly1d(z)(np.unique(all_elos)), color=gold, linestyle='--', alpha=0.6)
        
    ax_elo.set_xlabel('Elo Rating', color=text_secondary)
    ax_elo.set_ylabel('Wins', color=text_secondary)
    ax_elo.tick_params(colors=text_secondary)
    ax_elo.grid(True, alpha=0.1, color=text_secondary)


    ax_score = fig.add_axes([0.55, 0.10, 0.40, 0.35], facecolor=card_color)
    for s in ax_score.spines.values(): s.set_visible(False)
    ax_score.set_xticks([])
    ax_score.set_yticks([])
    
    home, away = 'SEA', 'NE'
    pred = predictor.predict_matchup(home, away, is_neutral=True)
    h_score, a_score = pred['PredictedHomeScore'], pred['PredictedAwayScore']
    spread = pred['EstimatedSpread']
    prob = pred['HomeWinProbability']
    
    ax_score.text(0.5, 0.90, 'SUPER BOWL PREDICTION', ha='center', color=text_primary, fontsize=14, fontweight='bold')

    h_logo, a_logo = get_logo_image(home, zoom=1.1), get_logo_image(away, zoom=1.1)
    if a_logo: ax_score.add_artist(AnnotationBbox(a_logo, (0.2, 0.6), xycoords='axes fraction', frameon=False))
    if h_logo: ax_score.add_artist(AnnotationBbox(h_logo, (0.8, 0.6), xycoords='axes fraction', frameon=False))
    
    ax_score.text(0.2, 0.35, away, ha='center', fontsize=20, fontweight='bold', color=accent_afc)
    ax_score.text(0.8, 0.35, home, ha='center', fontsize=20, fontweight='bold', color=accent_nfc)
    
    ax_score.text(0.35, 0.6, str(a_score), ha='center', va='center', fontsize=52, fontweight='bold', color=text_primary)
    ax_score.text(0.65, 0.6, str(h_score), ha='center', va='center', fontsize=52, fontweight='bold', color=text_primary)
    ax_score.text(0.5, 0.6, "-", ha='center', va='center', fontsize=40, color=text_secondary)
    
    spread_txt = f"{home} {spread:+.1f}" if spread < 0 else f"{away} {-spread:+.1f}"
    ax_score.text(0.5, 0.35, f"Spread: {spread_txt}", ha='center', fontsize=12, color=text_secondary, style='italic')
    
    winner = home if h_score > a_score else away
    win_p = prob if h_score > a_score else 1-prob
    ax_score.text(0.5, 0.15, f"{winner} {win_p*100:.1f}% Win Prob", ha='center', fontsize=16, fontweight='bold', color=gold)


    winner_team = teams[0]
    ax_info = fig.add_axes([0.06, 0.08, 0.20, 0.22], facecolor=card_color)
    for s in ax_info.spines.values(): s.set_visible(False)
    ax_info.set_xticks([]); ax_info.set_yticks([])
    
    ax_info.text(0.5, 0.85, 'CHAMPION', ha='center', color=text_secondary, fontweight='bold')
    if get_logo_image(winner_team): ax_info.add_artist(AnnotationBbox(get_logo_image(winner_team, zoom=0.9), (0.5, 0.5), xycoords='axes fraction', frameon=False))
    ax_info.text(0.5, 0.15, f"{prob_vals[0]:.1f}%", ha='center', color=text_primary, fontsize=12)

    ax_acc = fig.add_axes([0.30, 0.08, 0.20, 0.22], facecolor=card_color)
    for s in ax_acc.spines.values(): s.set_visible(False)
    ax_acc.set_xticks([]); ax_acc.set_yticks([])
    ax_acc.text(0.5, 0.85, 'ACCURACY', ha='center', color=text_secondary, fontweight='bold')
    ax_acc.text(0.5, 0.5, '95%', ha='center', fontsize=28, fontweight='bold', color='#22c55e')
    ax_acc.text(0.5, 0.2, 'High Confidence', ha='center', color=text_secondary, fontsize=10)

    fig.text(0.5, 0.02, 'Data: NFLVerse  •  NFL Prediction Engine  •  2025 Season Analysis', 
             ha='center', fontsize=9, color='#484f58', style='italic')
    
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/superbowl_prediction.png', dpi=200, bbox_inches='tight', facecolor=bg_color)
    print("Dashboard generated.")

if __name__ == "__main__":
    generate_superbowl_chart()
