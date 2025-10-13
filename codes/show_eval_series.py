import argparse
import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from trading_env import TradingEnv
import yfinance as yf

parser = argparse.ArgumentParser(description='Run a single evaluation episode and dump series to CSV')
parser.add_argument('--epoch', type=int, required=True)
parser.add_argument('--model_name', type=str, default='dqn_model')
args = parser.parse_args()

here = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(here, f"{args.model_name}_epoch_{args.epoch}.keras")
if not os.path.exists(model_path):
    raise SystemExit(f"Model not found: {model_path}")

print('Loading model:', model_path)
model = load_model(model_path)

test_csv = os.path.join(here, 'aapl_test.csv')
if not os.path.exists(test_csv):
    raise SystemExit('Test CSV not found: ' + test_csv)

# Run evaluation loop similar to evaluate.py so we capture net_worths
env = TradingEnv(test_csv, window_size=30)
state = env.reset()
state_size = state.size
done = False
rl_networths = [env.initial_balance]
actions = []
while not done:
    s_in = np.reshape(state, (1, state_size))
    q = model.predict(s_in, verbose=0)[0]
    a = int(np.argmax(q))
    actions.append(a)
    state, reward, done, _ = env.step(a)
    rl_networths.append(env.net_worth)

# Align buy-and-hold series using the same logic as evaluate.py
raw_test_prices = yf.download('AAPL', start='2021-01-01', end='2023-12-31', auto_adjust=False)['Close'].values
start_idx = env.window_size
num_steps = len(rl_networths) - 1
available_prices = len(raw_test_prices) - start_idx
usable_steps = min(num_steps, max(0, available_prices))
if usable_steps != num_steps:
    rl_networths = rl_networths[:usable_steps + 1]
    actions = actions[:usable_steps]
raw_aligned = raw_test_prices[start_idx:start_idx + usable_steps]

def buy_and_hold(initial_balance, raw_prices_aligned):
    raw = np.asarray(raw_prices_aligned, dtype=float).flatten()
    if raw.size == 0:
        return np.array([initial_balance], dtype=float)
    initial_price = float(raw[0])
    bah = initial_balance * (raw / initial_price)
    return np.concatenate(([initial_balance], bah))

bah_values = buy_and_hold(env.initial_balance, raw_aligned)
rl_arr = np.asarray(rl_networths, dtype=float).flatten()
bah_arr = np.asarray(bah_values, dtype=float).flatten()

# Trim to equal length
min_len = min(len(rl_arr), len(bah_arr))
rl_arr = rl_arr[:min_len]
bah_arr = bah_arr[:min_len]

# Normalize to start at 1 for easier comparison
rl_norm = rl_arr / rl_arr[0]
bah_norm = bah_arr / bah_arr[0]

out_csv = os.path.join(here, f'eval_epoch_{args.epoch}_series.csv')
cols = {}
if len(rl_norm) > 0:
    cols['rl_networth'] = rl_norm
if len(bah_norm) > 0:
    cols['bah_networth'] = bah_norm

if cols:
    maxlen = max(len(v) for v in cols.values())
    for k in cols:
        if len(cols[k]) < maxlen:
            cols[k] = np.pad(cols[k], (0, maxlen - len(cols[k])), mode='edge')
    df = pd.DataFrame(cols)
    df.to_csv(out_csv, index=False)
    print('Saved series to:', out_csv)

# Print quick stats
print('\nQuick stats for epoch', args.epoch)
if len(rl_norm) > 0:
    print('RL networth start/last/min/max:', float(rl_norm[0]), float(rl_norm[-1]), float(rl_norm.min()), float(rl_norm.max()))
else:
    print('No RL networth found')
if len(bah_norm) > 0:
    print('B&H networth start/last/min/max:', float(bah_norm[0]), float(bah_norm[-1]), float(bah_norm.min()), float(bah_norm.max()))

if len(rl_norm) > 0:
    print('\nFirst 10 RL normalized networth:', rl_norm[:10])
    print('Last 10 RL normalized networth:', rl_norm[-10:])
if len(bah_norm) > 0:
    print('\nFirst 10 B&H normalized networth:', bah_norm[:10])
    print('Last 10 B&H normalized networth:', bah_norm[-10:])

print('\nDone.')
