from trading_env import TradingEnv
from dqn_agent import DQNAgent
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf  # For raw B&H prices
from tensorflow.keras.models import load_model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Silence TF warnings
import argparse
import sys

# Load trained model (set epsilon=0 for deterministic decisions)
state_size = 30 * 12  # 360 (window * features)
agent = DQNAgent(state_size, 3)

# CLI: allow specifying a model path or an epoch
parser = argparse.ArgumentParser(description='Evaluate trained DQN trading agent')
parser.add_argument('--model', type=str, help='Path to a saved model file (e.g. /path/to/dqn_model.keras)')
parser.add_argument('--epoch', type=int, help='Load checkpoint by epoch number (e.g. --epoch 100 loads dqn_model_epoch_100.keras)')
parser.add_argument('--model_name', type=str, default='dqn_model', help='Base model name used for checkpoints')
args = parser.parse_args()

# Locate model file relative to this script
here = os.path.dirname(os.path.abspath(__file__))
if args.model:
    # Use exactly the provided model path
    candidates = [args.model]
elif args.epoch is not None:
    # Prefer explicit epoch checkpoint, fall back to base model name
    candidates = [
        os.path.join(here, f"{args.model_name}_epoch_{args.epoch}.keras"),
        os.path.join(here, f"{args.model_name}.keras"),
    ]
else:
    candidates = [
        os.path.join(here, 'dqn_model.keras'),
        os.path.join(here, 'dqn_model.h5'),
        os.path.join(here, 'dqn_model_weights.h5'),
        os.path.join(here, 'dqn_weights.h5'),
    ]
model_loaded = False
for path in candidates:
    if os.path.exists(path):
        try:
            if path.endswith('.keras') or path.endswith('.h5'):
                # If it's a full model file
                agent.model = load_model(path)
            else:
                # weights-only
                agent.model.load_weights(path)
            print(f"Loaded model from: {path}")
            model_loaded = True
            break
        except Exception as e:
            print(f"Found candidate model {path} but failed to load: {e}")

if not model_loaded:
    print("ERROR: No usable model file found. Tried:\n" + "\n".join(candidates))
    print("Place a Keras saved model (dqn_model.keras or dqn_model.h5) or weights file (dqn_model_weights.h5) in the rl_trading_agent folder.")
    raise SystemExit(1)

agent.epsilon = 0  # Greedy: Always pick best action, no exploration

# Test environment (normalized for RL)
test_csv = os.path.join(here, 'aapl_test.csv')
if not os.path.exists(test_csv):
    print(f"ERROR: test CSV not found at: {test_csv}")
    raise SystemExit(1)
test_env = TradingEnv(test_csv, window_size=30)

def evaluate_agent(agent, env):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    done = False
    net_worths = [env.initial_balance]  # Track portfolio value over time
    actions_log = []  # Optional: Log actions for inspection
    while not done:
        action = agent.act(state)
        actions_log.append(action)  # 0=Hold, 1=Buy, 2=Sell
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        state = next_state
        net_worths.append(env.net_worth)
    return net_worths, actions_log

def buy_and_hold(initial_balance, raw_prices_aligned):
    # Ensure input is a 1-D numeric numpy array
    raw = np.asarray(raw_prices_aligned, dtype=float).flatten()
    if raw.size == 0:
        return np.array([initial_balance], dtype=float)
    initial_price = float(raw[0])
    bah = initial_balance * (raw / initial_price)
    # Prepend initial balance so lengths align with RL's net_worths list
    bah_values = np.concatenate(([initial_balance], bah))
    return bah_values  # numpy array, len = 1 + len(aligned)

def calculate_metrics(values):
    # Accept lists or numpy arrays; coerce to 1-D float array
    values = np.asarray(values, dtype=float).flatten()
    if len(values) < 2:
        return {'Cumulative Returns (%)': 0, 'Sharpe Ratio': 0, 'Max Drawdown (%)': 0, 'Win Rate (%)': 0, 'Profit Factor': 0}
    returns = np.diff(values) / values[:-1]
    if len(returns) == 0 or np.isnan(returns).all():
        return {'Cumulative Returns (%)': 0, 'Sharpe Ratio': 0, 'Max Drawdown (%)': 0, 'Win Rate (%)': 0, 'Profit Factor': 0}
    cum_returns = (values[-1] / values[0] - 1) * 100
    std_returns = np.std(returns)
    sharpe = np.mean(returns) / std_returns * np.sqrt(252) if std_returns > 1e-8 else 0
    peak = np.maximum.accumulate(values)
    drawdowns = (peak - values) / peak * 100
    max_drawdown = np.max(drawdowns)
    wins = np.sum(returns > 0)
    win_rate = (wins / len(returns)) * 100
    pos_returns = np.sum(returns[returns > 0])
    neg_returns = abs(np.sum(returns[returns < 0]))
    profit_factor = pos_returns / neg_returns if neg_returns > 1e-8 else float('inf')
    return {
        'Cumulative Returns (%)': cum_returns,
        'Sharpe Ratio': sharpe,
        'Max Drawdown (%)': max_drawdown,
        'Win Rate (%)': win_rate,
        'Profit Factor': profit_factor
    }

# Fetch raw Close prices for B&H (full period, unnormalized)
raw_test_prices = yf.download('AAPL', start='2021-01-01', end='2023-12-31', auto_adjust=False)['Close'].values  # .values = np.array floats
# Align: Use from window to window + num_steps (but compute after RL for exact)
print("Running backtest on unseen data (2021-2023 AAPL)...")
rl_values, actions = evaluate_agent(agent, test_env)
num_steps = len(rl_values) - 1  # steps progressed by RL
start_idx = test_env.window_size
available_prices = len(raw_test_prices) - start_idx
if available_prices <= 0:
    print(f"ERROR: raw_test_prices has no data after window start (start_idx={start_idx}).")
    raise SystemExit(1)

# Trim to the minimum available length to keep RL and B&H comparable
usable_steps = min(num_steps, available_prices)
if usable_steps != num_steps:
    print(f"Warning: trimming RL run from {num_steps} steps to {usable_steps} to match available price data.")
    # Trim RL outputs (keep initial balance + usable_steps values)
    rl_values = rl_values[:usable_steps + 1]
    actions = actions[:usable_steps]
    num_steps = usable_steps

end_idx = start_idx + num_steps
raw_test_prices_aligned = raw_test_prices[start_idx:end_idx]
print(f"RL steps: {len(rl_values)-1}, Raw prices used: {len(raw_test_prices_aligned)}")

# Prepare series: coerce to 1-D numeric arrays and trim to same length
bah_values = buy_and_hold(test_env.initial_balance, raw_test_prices_aligned)
rl_arr = np.asarray(rl_values, dtype=float).flatten()
bah_arr = np.asarray(bah_values, dtype=float).flatten()
print(f"rl_arr shape: {rl_arr.shape}, bah_arr shape: {bah_arr.shape}")
min_len = min(len(rl_arr), len(bah_arr))
if len(rl_arr) != len(bah_arr):
    print(f"Trimming series to min length {min_len} for metrics/plotting.")
    rl_arr = rl_arr[:min_len]
    bah_arr = bah_arr[:min_len]

# Metrics (now aligned & uniform)
rl_metrics = calculate_metrics(rl_arr)
bah_metrics = calculate_metrics(bah_arr)

# Plot: Portfolio growth comparison
plt.figure(figsize=(12, 6))
plt.plot(rl_arr, label='RL Agent', linewidth=2)
plt.plot(bah_arr, label='Buy-and-Hold', linewidth=2)
plt.title('Backtest Results: RL Agent vs. Buy-and-Hold (AAPL 2021-2023)')
plt.xlabel('Trading Days')
plt.ylabel('Portfolio Value ($)')
plt.legend()
plt.grid(True)
# Save plot next to the script for later inspection and also show it interactively
out_plot = os.path.join(here, 'evaluation_plot.png')
plt.savefig(out_plot, bbox_inches='tight')
plt.close()
print(f"Saved plot to: {out_plot}")

# Pretty-format metrics and print immediately
def pretty_metrics(metrics_dict):
    m = metrics_dict.copy()
    # Display Max Drawdown as negative value
    if 'Max Drawdown (%)' in m:
        m['Max Drawdown (%)'] = -abs(m['Max Drawdown (%)'])
    return m

rl_pretty = pretty_metrics(rl_metrics)
bah_pretty = pretty_metrics(bah_metrics)

comparison = pd.DataFrame({'RL Agent': rl_pretty, 'Buy-and-Hold': bah_pretty})
comparison = comparison.reindex(['Cumulative Returns (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 'Win Rate (%)', 'Profit Factor'])

print('\nRunning backtest...\n')
print('Performance Metrics:')
print(comparison.round(2).to_string())

# Quick action summary (see what it predicted)
action_names = ['Hold', 'Buy', 'Sell']
buys = actions.count(1)
sells = actions.count(2)
holds = actions.count(0)
print(f"\nTrade Summary: {buys} Buys, {sells} Sells, {holds} Holds (Total steps: {len(actions)})")