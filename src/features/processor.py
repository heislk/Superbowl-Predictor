import pandas as pd
import numpy as np
from typing import List, Dict

class FeatureProcessor:
    def __init__(self):
        pass

    def process_team_stats(self, games: List[Dict], teams: List[Dict]) -> pd.DataFrame:
        if not games:
            return pd.DataFrame()

        df_games = pd.DataFrame(games)
        
        completed_games = df_games[df_games['Status'] == 'Final'].copy()
        
        if completed_games.empty:
            return pd.DataFrame()

        home_games = completed_games[['Season', 'Week', 'HomeTeam', 'AwayTeam', 'HomeScore', 'AwayScore']].copy()
        home_games.columns = ['Season', 'Week', 'Team', 'Opponent', 'PointsFor', 'PointsAllowed']
        home_games['IsHome'] = 1
        home_games['Won'] = (home_games['PointsFor'] > home_games['PointsAllowed']).astype(int)
        
        away_games = completed_games[['Season', 'Week', 'AwayTeam', 'HomeTeam', 'AwayScore', 'HomeScore']].copy()
        away_games.columns = ['Season', 'Week', 'Team', 'Opponent', 'PointsFor', 'PointsAllowed']
        away_games['IsHome'] = 0
        away_games['Won'] = (away_games['PointsFor'] > away_games['PointsAllowed']).astype(int)
        
        team_games = pd.concat([home_games, away_games], ignore_index=True)
        
        stats = team_games.groupby('Team').agg(
            Games=('Won', 'count'),
            Wins=('Won', 'sum'),
            PointsFor=('PointsFor', 'sum'),
            PointsAllowed=('PointsAllowed', 'sum')
        ).reset_index()
        
        stats['Losses'] = stats['Games'] - stats['Wins']
        stats['WinPct'] = stats['Wins'] / stats['Games']
        stats['PPG'] = stats['PointsFor'] / stats['Games']
        stats['PAPG'] = stats['PointsAllowed'] / stats['Games']
        stats['PointDiff'] = stats['PointsFor'] - stats['PointsAllowed']
        stats['NetPointDiff'] = stats['PointDiff'] / stats['Games']
        
        return stats

    def calculate_rest_days(self, games: List[Dict]) -> pd.DataFrame:
        if not games:
            return pd.DataFrame()
            
        df = pd.DataFrame(games).sort_values(['Season', 'Date']) 
        
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            
            home_sched = df[['Season', 'Week', 'HomeTeam', 'Date']].rename(columns={'HomeTeam': 'Team'})
            away_sched = df[['Season', 'Week', 'AwayTeam', 'Date']].rename(columns={'AwayTeam': 'Team'})
            full_sched = pd.concat([home_sched, away_sched]).sort_values(['Team', 'Date'])
            
            full_sched['PrevGameDate'] = full_sched.groupby('Team')['Date'].shift(1)
            full_sched['RestDays'] = (full_sched['Date'] - full_sched['PrevGameDate']).dt.days
            
            full_sched['RestDays'] = full_sched['RestDays'].fillna(7) 
            
            return full_sched[['Season', 'Week', 'Team', 'RestDays']]
            
        return pd.DataFrame()
