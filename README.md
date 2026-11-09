# Hourly chart transition model

A forecast of Roblox's Top Trending chart (device computer, country US): for each of its top 10 places, a probability
for every game at every hour of the 24 hours after the history ends, written as one forecast scored against each
hourly read with a log score.

`python3 main.py` reads `$MAKEFUN_INPUT_DIR` (else `/input`): `history.ndjson` and `task.json`, and writes
`$MAKEFUN_OUTPUT_DIR` (else `/output`)`/forecast.json`. Python 3.13, standard library only, no network, deterministic.

## How it forecasts

- **Per game, per hour:** each game on the last chart gets a distribution over its future rank in log-rank space. The
  mean drifts with the game's measured momentum (a damped log-rank trend, weaker for games already in the top 10,
  whose moves barely trend) and with its player surge (players now against the same hours a day earlier, both
  relative to the chart's median, which takes out the daily cycle). An explosive surge, several times the players
  of the day before, lifts a game outside the top 10 further and faster: the chart's daily-user measure lags the
  live player count, so the rank catches up within hours. The spread grows with the horizon, with a heavy-tailed
  second component.
- **Exits and newcomers:** each game may leave the chart (a hazard by rank band), and each place keeps some mass for
  games that are not on the chart yet. When a top-10 game leaves, every game below it moves up one place at once,
  so part of the top-10 exit rate is modelled as explicit dropout scenarios: each is the same model run on the chart
  without that game. A game that just broke into the Top Playing Now top 20 with a player surge, while off the
  trending chart, is named as a likely newcomer near the top.
- **One game per place:** the games-by-places matrix is balanced by Sinkhorn iterations, so each place holds one game
  and each game one place.
- **One forecast for the whole window:** the hourly matrices are averaged over the 24 slots, which is the best single
  forecast under a log score.
- **Naming:** every game of the last chart is named at every place, so a place's unnamed rest is shared only by the
  games that are new to the chart.

The parameters were tuned on rolling-origin backtests of the public archive and of hourly reads of the chart (every
read as the last of a history, scored on the reads of the next 24 hours with the contest's scoring rule), then hedged
toward wider tails.

## Licence

MIT, see `LICENSE`.
