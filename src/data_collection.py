import nflreadpy as nfl
import pandas as pd

games = nfl.load_schedules(seasons=[2019,2020,2021,2022,2023,2024,2025])

#converting to pandas

games = games.to_pandas()

games['home_win'] = (games['home_score'] > games['away_score']).astype(int)

# games with no score
games = games.dropna(subset=['home_score', 'away_score'])

games.to_csv('data/raw/games.csv', index=False)
print(f"Total games: {len(games)}")
print(f"Home win rate: {games['home_win'].mean():.1%}")
print(f"Columns available: {games.columns.tolist()}")
