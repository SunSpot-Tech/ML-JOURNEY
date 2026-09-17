from fbref_cache_builder import LEAGUES, build_team_stats, parse_fbref_team_tables, parse_schedule

assert LEAGUES["Championship"]["comp_id"] == "10"
assert LEAGUES["Championship"]["schedule_short"] is True

SCHEDULE = '''<table><thead><tr><th>Wk</th><th>Date</th><th>Home</th><th>Score</th><th>Away</th><th>Referee</th></tr></thead><tbody><tr><td>1</td><td>2026-08-15</td><td>Arsenal</td><td>2–0</td><td>Chelsea</td><td>Ref</td></tr><tr><td>2</td><td>2026-08-22</td><td>Chelsea</td><td>1–1</td><td>Arsenal</td><td>Ref</td></tr></tbody></table>'''
STATS = '''<table><thead><tr><th>Squad</th><th>Poss</th><th>Cmp%</th><th>Save%</th></tr></thead><tbody><tr><td>Arsenal</td><td>57.4</td><td>86.2</td><td>72.0</td></tr><tr><td>Chelsea</td><td>51.1</td><td>83.5</td><td>68.0</td></tr></tbody></table>'''

matches = parse_schedule(SCHEDULE)
assert len(matches) == 2
assert matches[0]["HomeTeam"] == "Arsenal"
assert matches[0]["FTR"] == "H"

stats_rows = parse_fbref_team_tables(STATS)
stats = build_team_stats(matches, stats_rows)
assert stats["Arsenal"]["avg_goals"] == 1.5
assert stats["Arsenal"]["possession"] == 57.4
assert stats["Arsenal"]["pass_accuracy"] == 86.2
assert stats["Arsenal"]["gk_save_pct"] == 72.0
print("FBref parser tests passed")
