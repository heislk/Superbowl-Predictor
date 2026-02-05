import nfl_data_py as nfl
import pandas as pd
import logging
from typing import List, Dict, Optional, Any
from src.data import storage

logger = logging.getLogger(__name__)

class NFLVerseClient:
    def __init__(self):
        pass

    def get_teams(self, force_refresh: bool = False) -> List[Dict]:
        filename = "teams_nflverse.json"
        
        if not force_refresh:
            data = storage.load_json(filename, processed=False)
            if data:
                return data

        try:
            logger.info("Fetching Team Data from NFLVerse...")
            df = nfl.import_team_desc()
            df = df.fillna('')
            
            df_renamed = df.rename(columns={
                'team_abbr': 'Key',
                'team_conf': 'Conference',
                'team_division': 'Division',
                'team_name': 'FullName'
            })
            
            data = df_renamed.to_dict(orient='records')
            storage.save_json(filename, data, processed=False)
            return data
            
        except Exception as e:
            logger.error(f"Error fetching teams: {e}")
            return []

    def get_schedules(self, season: int, force_refresh: bool = False) -> List[Dict]:
        filename = f"schedules_nflverse_{season}.json"
        
        if not force_refresh:
            data = storage.load_json(filename, processed=False)
            if data:
                return data

        try:
            logger.info(f"Fetching Schedule for {season} from NFLVerse...")
            df = nfl.import_schedules([season])
            
            if 'game_type' in df.columns:
                df = df[df['game_type'] == 'REG']
            
            try:
                logger.info("Fetching Weekly Player Stats for Granular Metrics...")
                p_df = nfl.import_weekly_data([season])
                
                p_df['turnovers'] = p_df['interceptions'] + p_df['rushing_fumbles_lost'] + p_df['receiving_fumbles_lost'] + p_df['sack_fumbles_lost']
                
                team_stats = p_df.groupby(['season', 'week', 'recent_team'])[['rushing_yards', 'passing_yards', 'turnovers', 'rushing_epa', 'passing_epa']].sum().reset_index()
                
                df = df.merge(team_stats, left_on=['season', 'week', 'home_team'], right_on=['season', 'week', 'recent_team'], how='left')
                df = df.rename(columns={
                    'rushing_yards': 'home_rush_yards', 
                    'passing_yards': 'home_pass_yards',
                    'turnovers': 'home_turnovers',
                    'rushing_epa': 'home_rush_epa',
                    'passing_epa': 'home_pass_epa'
                })
                if 'recent_team' in df.columns: df = df.drop(columns=['recent_team'])
                
                df = df.merge(team_stats, left_on=['season', 'week', 'away_team'], right_on=['season', 'week', 'recent_team'], how='left', suffixes=('', '_away'))
                
                df = df.rename(columns={
                    'rushing_yards': 'away_rush_yards', 
                    'passing_yards': 'away_pass_yards',
                    'turnovers': 'away_turnovers',
                    'rushing_epa': 'away_rush_epa',
                    'passing_epa': 'away_pass_epa'
                })
                if 'recent_team' in df.columns: df = df.drop(columns=['recent_team'])
                
                df = df.fillna({
                    'home_rush_yards': 0, 'home_pass_yards': 0, 'home_turnovers': 0, 'home_rush_epa': 0, 'home_pass_epa': 0,
                    'away_rush_yards': 0, 'away_pass_yards': 0, 'away_turnovers': 0, 'away_rush_epa': 0, 'away_pass_epa': 0
                })
                
            except Exception as e:
                logger.error(f"Failed to merge granular stats: {e}")
                cols = ['home_rush_yards', 'home_pass_yards', 'home_turnovers', 'home_rush_epa', 'home_pass_epa',
                        'away_rush_yards', 'away_pass_yards', 'away_turnovers', 'away_rush_epa', 'away_pass_epa']
                for c in cols:
                    df[c] = 0.0

            df['home_score'] = df['home_score'].fillna(0)
            df['away_score'] = df['away_score'].fillna(0)
            
            def get_status(row):
                if pd.notnull(row['result']):
                    return 'Final'
                return 'Scheduled'
            
            df['Status'] = df.apply(get_status, axis=1)
            
            if 'home_rest' not in df.columns: df['home_rest'] = 7
            if 'away_rest' not in df.columns: df['away_rest'] = 7
            
            if 'spread_line' in df.columns:
                 df['spread_line'] = -df['spread_line']
            else:
                 df['spread_line'] = 0.0

            df_renamed = df.rename(columns={
                'season': 'Season',
                'week': 'Week',
                'home_team': 'HomeTeam',
                'away_team': 'AwayTeam',
                'home_score': 'HomeScore',
                'away_score': 'AwayScore',
                'game_id': 'GameKey',
                'gameday': 'Date',
                'home_rest': 'HomeRest',
                'away_rest': 'AwayRest',
                'spread_line': 'spread_line',
                'home_rush_yards': 'home_rush_yards',
                'home_pass_yards': 'home_pass_yards',
                'home_turnovers': 'home_turnovers',
                'home_rush_epa': 'home_rush_epa',
                'home_pass_epa': 'home_pass_epa',
                'away_rush_yards': 'away_rush_yards',
                'away_pass_yards': 'away_pass_yards',
                'away_turnovers': 'away_turnovers',
                'away_rush_epa': 'away_rush_epa',
                'away_pass_epa': 'away_pass_epa'
            })
            
            cols = ['Season', 'Week', 'HomeTeam', 'AwayTeam', 'HomeScore', 'AwayScore', 'GameKey', 'Status', 'Date', 'HomeRest', 'AwayRest', 'spread_line', 'home_qb_name', 'away_qb_name',
                    'home_rush_yards', 'home_pass_yards', 'home_turnovers', 'home_rush_epa', 'home_pass_epa', 
                    'away_rush_yards', 'away_pass_yards', 'away_turnovers', 'away_rush_epa', 'away_pass_epa']
            data = df_renamed[cols].to_dict(orient='records')
            
            storage.save_json(filename, data, processed=False)
            return data
            
        except Exception as e:
            logger.error(f"Error fetching schedule: {e}")
            return []
