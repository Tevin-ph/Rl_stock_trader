from tensorflow.keras.models import load_model
import numpy as np
import os
import glob
import argparse
from trading_env import TradingEnv

here = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description='Diagnostic: inspect saved DQN checkpoints for action distributions')
parser.add_argument('--model', type=str, help='Exact model path to load')
parser.add_argument('--epoch', type=int, help='Load checkpoint by epoch number (e.g. --epoch 100 loads dqn_model_epoch_100.keras)')
parser.add_argument('--model_name', type=str, default='dqn_model', help='Base model name used for checkpoints')
parser.add_argument('--scan', action='store_true', help='Scan all *_epoch_*.keras checkpoints in the folder')
args = parser.parse_args()

def find_model_path(args):
    if args.model:
        if os.path.exists(args.model):
            return args.model
        # try relative to here
        candidate = os.path.join(here, args.model)
        if os.path.exists(candidate):
            return candidate
        return None
    if args.epoch is not None:
        candidate = os.path.join(here, f"{args.model_name}_epoch_{args.epoch}.keras")
        if os.path.exists(candidate):
            return candidate
        # fallback to base model
        base = os.path.join(here, f"{args.model_name}.keras")
        if os.path.exists(base):
            return base
        return None
    return None

test_csv = os.path.join(here, 'aapl_test.csv')
if not os.path.exists(test_csv):
    print('Test CSV not found:', test_csv)
    raise SystemExit(1)

env = TradingEnv(test_csv, window_size=30)

def predict_q_for_model(model, env, sample_steps=5):
    state = env.reset()
    state_size = state.size
    def predict_q(s):
        s = np.reshape(s, [1, state_size])
        q = model.predict(s, verbose=0)[0]
        return q

    # Sample first few Q-values
    q_samples = []
    s = state
    for i in range(sample_steps):
        q = predict_q(s)
        q_samples.append((np.round(q, 4), int(np.argmax(q))))
        # advance with a neutral action so we observe predictions on natural states
        s, _, done, _ = env.step(0)
        if done:
            break

    # Full episode action distribution (predict-only)
    counts = {0:0, 1:0, 2:0}
    s = env.reset()
    done = False
    while not done:
        q = predict_q(s)
        a = int(np.argmax(q))
        counts[a] += 1
        s, _, done, _ = env.step(0)

    return q_samples, counts

def scan_and_report():
    pattern = os.path.join(here, f"{args.model_name}_epoch_*.keras")
    files = sorted(glob.glob(pattern))
    if not files:
        print('No checkpoint files found matching', pattern)
        return
    print(f'Found {len(files)} checkpoints. Scanning...')
    for f in files:
        try:
            model = load_model(f)
        except Exception as e:
            print(f'Failed to load {f}: {e}')
            continue
        q_samples, counts = predict_q_for_model(model, env, sample_steps=3)
        print('\nCheckpoint:', os.path.basename(f))
        print('  Sample Q-values (first states):')
        for i, (q, argmax) in enumerate(q_samples):
            print(f'    Step {i}: Q={q} argmax={argmax}')
        total = sum(counts.values())
        if total == 0:
            print('  No steps evaluated (empty env).')
            continue
        print('  Action distribution:', counts, f'(total steps: {total})')

if args.scan:
    scan_and_report()
else:
    path = find_model_path(args)
    if path is None:
        print('Model not found. Use --scan to list checkpoints or pass --epoch/--model')
        raise SystemExit(1)
    print('Loading model:', path)
    model = load_model(path)
    q_samples, counts = predict_q_for_model(model, env, sample_steps=5)
    print('\nSample Q-values for first states:')
    for i, (q, argmax) in enumerate(q_samples):
        print(f'Step {i}: Q =', q, 'argmax=', argmax)
    print('\nFull action distribution across the test episode:')
    print(counts)
    print('\nIf the model always picks the same action (e.g. 2), the agent will likely have flat net worth because actions do nothing meaningful when starting with cash or no shares.')
