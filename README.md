# RL Trading Agent 🤖📈

A **Reinforcement Learning trading agent** that learns to make **Buy / Sell / Hold** decisions on real forex and crypto price data (BTC-USD, EUR-USD). Agents are evaluated against a passive **buy-and-hold** baseline across multiple risk-adjusted performance metrics.

---

## 📁 Project Structure

```
rl_trading_agent/
├── config.py                 # Central configuration (paths, hyper-parameters)
├── data/                     # Auto-generated CSV splits
├── env/
│   └── trading_env.py        # Custom Gymnasium environment
├── agents/
│   ├── dqn_agent.py          # DQN agent wrapper (Stable-Baselines3)
│   └── ppo_agent.py          # PPO agent wrapper (Stable-Baselines3)
├── train.py                  # Training script with MLflow logging
├── evaluate.py               # Test-set evaluation & metric table
├── backtest.py               # Three-panel backtest visualisation
├── utils/
│   ├── data_loader.py        # yfinance download, clean, split pipeline
│   └── indicators.py         # Technical indicators & feature normalisation
├── results/                  # Models, metrics, plots, MLflow runs
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

```bash
# 1. Clone or enter the project directory
cd rl_trading_agent

# 2. Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Linux / macOS

# 3. Install dependencies
pip install -r requirements.txt
```

> **Python ≥ 3.10** is required.

---

## 🚀 How to Train

```bash
python train.py                   # Train on BTC-USD (default)
python train.py --ticker EURUSD=X  # Train on EUR/USD
```

This will:

1. Download 5 years of daily OHLCV data via **yfinance** (if not cached).
2. Compute technical indicators and normalise features.
3. Train a **DQN** agent for 200 000 timesteps.
4. Train a **PPO** agent for 200 000 timesteps.
5. Evaluate on the validation set every 10 000 steps and save the best checkpoint.
6. Log all hyperparameters and metrics to **MLflow** (`results/mlruns/`).
7. Save final models as `results/dqn_best.zip` and `results/ppo_best.zip`.

---

## 📊 How to Evaluate

```bash
python evaluate.py
python evaluate.py --ticker EURUSD=X
```

Runs both trained agents **and** a buy-and-hold baseline on the **held-out test set** (last 15 % of data). Prints a comparison table and saves `results/metrics.json`.

---

## 📈 Backtesting Visualisation

```bash
python backtest.py
python backtest.py --ticker EURUSD=X
```

Generates a three-panel chart saved to `results/backtest_plot.png`:

| Panel | Content |
|-------|---------|
| 1     | Portfolio value over time (DQN vs PPO vs Buy & Hold) |
| 2     | Asset price with coloured Buy ▲ / Sell ▼ markers |
| 3     | Drawdown curves for each strategy |

---

## 📋 Results

| Metric             | DQN      | PPO      | Buy & Hold |
|--------------------|----------|----------|------------|
| Total Return (%)   | *TBD*    | *TBD*    | *TBD*      |
| Sharpe Ratio       | *TBD*    | *TBD*    | *TBD*      |
| Max Drawdown (%)   | *TBD*    | *TBD*    | *TBD*      |
| Win Rate (%)       | *TBD*    | *TBD*    | *TBD*      |
| Total Trades       | *TBD*    | *TBD*    | 1          |
| Calmar Ratio       | *TBD*    | *TBD*    | *TBD*      |

> Run `python evaluate.py` to populate these numbers.  Results are saved in `results/metrics.json`.

---

## 🧠 Reward Function Design

The agent receives a reward equal to the **logarithmic return** of its portfolio at each timestep:

```
reward_t = ln(V_t / V_{t-1})
```

### Why log returns?

| Property | Benefit |
|----------|---------|
| **Additivity** | Log returns sum across time, making cumulative performance easy to reason about. |
| **Symmetry** | A +5 % gain followed by a −5 % loss nets close to zero, unlike arithmetic returns. |
| **Scale-invariance** | Rewards remain comparable regardless of portfolio size. |
| **Tail penalisation** | Large losses produce disproportionately negative rewards, discouraging excessive risk. |

### Transaction cost penalty

Every trade incurs a **0.1 % proportional cost** deducted from cash.  This:

- Prevents the agent from "churning" — flipping positions every step for marginal gain.
- Forces the agent to only trade when the expected gain exceeds the cost.
- Reflects realistic market conditions (spread + commission).

### Position model

The agent maintains a **discrete position** (long / flat / short).  Switching from long to short requires closing the existing position first, incurring **two** transaction costs.  This asymmetry encourages conviction: the agent learns to hold profitable positions rather than trade on noise.

---

## 🔒 Reproducibility

- All random seeds (NumPy, PyTorch, environment, SB3) are controlled via `config.py → seed`.
- Train / val / test splits are **chronological** — no shuffling.
- The `RobustScaler` is fitted **only on training data**, preventing data leakage.

---

## 📦 Dependencies

See `requirements.txt` for the full pinned list.  Key packages:

- `stable-baselines3` — DQN and PPO implementations
- `gymnasium` — environment interface
- `yfinance` — real market data
- `mlflow` — experiment tracking
- `scikit-learn` — feature scaling
- `matplotlib` — visualisation

---

## License

MIT
