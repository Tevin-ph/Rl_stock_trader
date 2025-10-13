from trading_env import TradingEnv
from dqn_agent import DQNAgent
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import tensorflow as tf
import argparse
from tensorflow.keras.models import load_model
import os  # For file checks

# Deterministic seeding for reproducibility
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# CLI for flexibility
parser = argparse.ArgumentParser(description='Train DQN Trading Agent')
parser.add_argument('--episodes', type=int, default=100, help='Number of training episodes')
parser.add_argument('--model_name', type=str, default='dqn_model', help='Base name for saved models')
parser.add_argument('--batch_size', type=int, default=32, help='Replay batch size')
parser.add_argument('--resume_epoch', type=int, default=0, help='Resume from this epoch (0 = fresh start)')
parser.add_argument('--target_update', type=int, default=5, help='Episodes between target network updates')
args = parser.parse_args()

import os
here = os.path.dirname(os.path.abspath(__file__))

# Setup env and agent (use script-relative path for CSV)
train_csv = os.path.join(here, 'aapl_train.csv')
env = TradingEnv(train_csv, window_size=30)
if hasattr(env, 'seed'):  # Gymnasium seed if available
    env.seed(42)
state_size = env.observation_space.shape[0] * env.observation_space.shape[1]  # 30 * 12 = 360 (flattened state)
agent = DQNAgent(state_size, 3)  # 3 actions: 0=hold, 1=buy, 2=sell

# Resume from checkpoint if specified
start_episode = 0
if args.resume_epoch > 0:
    # Use script-relative path for resume so running from workspace root still finds the checkpoint
    resume_path = os.path.join(here, f"{args.model_name}_epoch_{args.resume_epoch}.keras")
    if os.path.exists(resume_path):
        agent.model = load_model(resume_path)
        start_episode = args.resume_epoch
        print(f"Resumed from epoch {args.resume_epoch} using {resume_path}")
    else:
        print(f"Warning: Checkpoint {resume_path} not found. Starting fresh.")
else:
    print("Starting fresh training.")

def train_agent(agent, env, episodes, batch_size, start_episode):
    scores = []  # Track total rewards per episode
    log_data = []  # For CSV log
    
    # Load existing log if resuming
    log_path = os.path.join(here, 'training_log.csv')
    if start_episode > 0 and os.path.exists(log_path):
        existing_log = pd.read_csv(log_path).to_dict('records')
        log_data = existing_log  # Append to it
        # Load past scores for plot
        past_scores = [row['score'] for row in existing_log]
        scores = past_scores  # Start with past for full plot
        print(f"Loaded {len(existing_log)} past episodes from log.")
    
    for e in range(start_episode, episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])  # Flatten for the neural net input
        total_reward = 0
        done = False
        while not done:
            action = agent.act(state)  # Agent picks action (random early, smart later)
            next_state, reward, done, _ = env.step(action)  # Env responds with new state/reward
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)  # Store for replay
            state = next_state
            total_reward += reward
            if done:
                print(f"Episode {e+1}/{episodes}, Score: {total_reward:.4f}, Epsilon: {agent.epsilon:.3f}")
                scores.append(total_reward)
                log_data.append({'episode': e+1, 'score': total_reward, 'epsilon': agent.epsilon})
                break
        # Learn: Replay experiences to update neural net
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        # Periodically copy weights to the target network for stability
        if (e + 1) % args.target_update == 0:
            agent.update_target_model()
        
        # Checkpoint every 10 episodes
        if (e + 1) % 10 == 0:
            checkpoint_name = os.path.join(here, f"{args.model_name}_epoch_{e+1}.keras")
            agent.model.save(checkpoint_name)
            print(f"Checkpoint saved: {checkpoint_name}")
    
    # Save updated training log to CSV
    log_df = pd.DataFrame(log_data)
    log_df.to_csv(log_path, index=False)
    print(f"Training log updated/saved to '{log_path}'")
    
    return scores

# Kick off training
print("Starting DQN training on AAPL data (seeded for reproducibility)...")
scores = train_agent(agent, env, args.episodes, args.batch_size, start_episode)

# Plot the learning curve (full history if resumed)
plt.figure(figsize=(10, 5))
plt.plot(scores)
plt.title('Training Progress: Total Reward per Episode')
plt.xlabel('Episode')
plt.ylabel('Cumulative Reward')
plt.grid(True)
plot_path = os.path.join(here, 'training_plot.png')
plt.savefig(plot_path)  # Save for report
plt.show()

# Save final model
final_model_name = os.path.join(here, f"{args.model_name}.keras")
agent.model.save(final_model_name)
log_path = os.path.join(here, 'training_log.csv')
print(f"Training complete! Final model saved as '{final_model_name}'. Checkpoints: *_epoch_*.keras. Log: {log_path}")