# Market Analysis Summary


Analysis timestamp: 2025-04-13 15:52:06.869377


Data period: 2025-04-12 00:00:00 to 2025-04-14 00:00:00


## Data Quality Summary


### Data Coverage
- Time span: 2 days 00:00:00
- Effective coverage: 100.0%

## Statistical Analysis

### Stationarity Tests

- price: Non-stationary (p-value: 0.5755)
- volume: Stationary (p-value: 0.0000)

### Granger Causality Tests

- ['volume'] -> ['price']: Not significant (min p-value: 1.0000)
- ['rsi'] -> ['price']: Not significant (min p-value: 1.0000)
- ['macd'] -> ['price']: Not significant (min p-value: 0.9994)
- ['bb_upper', 'bb_lower'] -> ['price']: Not significant (min p-value: 0.9997)

## HMM Market Regime Analysis

### Overall State Distribution:
- State 0: 2657 periods (92.2%)
- State 1: 224 periods (7.8%)

### Recent State Analysis:

Last 5 periods:
- State 0: 4 periods (80.0%)
- State 1: 1 periods (20.0%)

Last 10 periods:
- State 0: 9 periods (90.0%)
- State 1: 1 periods (10.0%)

Last 20 periods:
- State 0: 19 periods (95.0%)
- State 1: 1 periods (5.0%)

Most Recent: Transition from State 0 to State 1

## HMM State Characteristics

### State 1

#### price
- mean: 2285.9152
- std: 2659.4264
- median: 2033.5000
- min: 2033.5000
- max: 30504.5000
#### volume
- mean: 613.2634
- std: 5263.9910
- median: 0.0000
- min: 0.0000
- max: 46300.0000
#### volatility_5
- mean: 0.0324
- std: 0.1954
- median: 0.0000
- min: 0.0000
- max: 1.2111
#### rsi
- mean: 0.8929
- std: 9.4068
- median: 0.0000
- min: 0.0000
- max: 100.0000
#### macd
- mean: -364.7833
- std: 1327.6296
- median: -0.0002
- min: -7183.6291
- max: 3701.5385

### State 0

#### price
- mean: 17453.0593
- std: 14186.1716
- median: 30504.5000
- min: 2033.5000
- max: 30504.5000
#### volume
- mean: 0.0000
- std: 0.0000
- median: 0.0000
- min: 0.0000
- max: 0.0000
#### volatility_5
- mean: 0.0000
- std: 0.0000
- median: 0.0000
- min: 0.0000
- max: 0.0000
#### rsi
- mean: 54.1588
- std: 49.8267
- median: 100.0000
- min: 0.0000
- max: 100.0000
#### macd
- mean: 0.0000
- std: 0.0000
- median: 0.0000
- min: -0.0000
- max: 0.0000


## HMM State Persistence

### State 1
- Average duration: 0 days 00:12:54.887892376
- Number of transitions: 3.0

### State 0
- Average duration: 0 days 00:01:05.015060240
- Number of transitions: 2.0


## Traditional Market Regimes

### Volatility Regime
Current: high

Distribution:
- medium: 2875 periods (99.8%)
- high: 6 periods (0.2%)
- low: 0 periods (0.0%)

### Trend Regime
Current: strong_up

Distribution:
- weak_sideways: 2875 periods (99.8%)
- strong_down: 5 periods (0.2%)
- strong_up: 1 periods (0.0%)

## Anomaly Detection

Total anomalies detected: 0 (0.0% of all observations)

Recent periods:
- Last 5 periods: 0 anomalies (0.0%)
- Last 10 periods: 0 anomalies (0.0%)
- Last 20 periods: 0 anomalies (0.0%)

## Trading Signals

Current Signal: Bullish (strength: 20.0)

Recent Signal Analysis:
- Last 5 periods: Bullish (avg strength: 20.0, max: 20.0)
- Last 10 periods: Bullish (avg strength: 20.0, max: 20.0)
- Last 20 periods: Bullish (avg strength: 20.0, max: 20.0)