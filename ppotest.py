import streamlit as st
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import time
import pandas as pd

# --- 1. DEFINE THE ENVIRONMENT ---
class CollisionEnv(gym.Env):
    def __init__(self, mode="Avoid Collision"):
        super(CollisionEnv, self).__init__()
        self.mode = mode
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=100.0, shape=(1,), dtype=np.float32)
        self.state = np.array([50.0], dtype=np.float32)
        self.steps_taken = 0
        
    def step(self, action):
        self.steps_taken += 1
        self.state[0] += action[0] * 5.0 
        self.state[0] = np.clip(self.state[0], 0.0, 100.0)
        
        collision = bool(self.state[0] <= 0.0)
        
        if self.mode == "Avoid Collision":
            if collision:
                reward = -100.0
                terminated = True
            else:
                reward = 1.0
                terminated = False
        else:  # "Seek Collision"
            if collision:
                reward = 100.0
                terminated = True
            else:
                reward = -1.0
                terminated = False
                
        truncated = self.steps_taken >= 100
        info = {"collision": collision}
        return self.state, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([np.random.uniform(30.0, 70.0)], dtype=np.float32)
        self.steps_taken = 0
        return self.state, {}

# --- 2. LIVE TRAINING PROGRESS CALLBACK ---
class DynamicProgressCallback(BaseCallback):
    def __init__(self, mode, chart_placeholder, status_placeholder, total_timesteps):
        super().__init__(verbose=0)
        self.mode = mode
        self.chart_placeholder = chart_placeholder
        self.status_placeholder = status_placeholder
        self.total_timesteps = total_timesteps
        
    def _on_step(self) -> bool:
        # Detect when an episode finishes
        if self.locals.get("dones")[0]:
            info = self.locals.get("infos")[0]
            collision = info.get("collision", False)
            
            # Evaluate if the episode was a success based on the current mode
            if self.mode == "Avoid Collision":
                success = 1 if not collision else 0
            else:
                success = 1 if collision else 0
                
            st.session_state.episode_outcomes.append(success)
            st.session_state.total_episodes_completed += 1
            
            # Calculate rolling accuracy over the last 50 episodes for smooth tracking
            window = st.session_state.episode_outcomes[-50:]
            current_accuracy = (sum(window) / len(window)) * 100
            
            st.session_state.accuracy_history.append({
                "Episode": st.session_state.total_episodes_completed,
                "Rolling Accuracy (%)": current_accuracy
            })
            
            # Dynamically update line chart
            df = pd.DataFrame(st.session_state.accuracy_history)
            self.chart_placeholder.line_chart(df.set_index("Episode"))
            
        # Update text progression status
        current_steps = self.num_timesteps
        percent = min(100, int((current_steps / self.total_timesteps) * 100))
        self.status_placeholder.text(
            f"Training session progress: {percent}% ({current_steps}/{self.total_timesteps} steps) | "
            f"Total Episodes Run: {st.session_state.total_episodes_completed}"
        )
        return True

# --- 3. 100-EPISODE ACCURACY EVALUATOR ---
def evaluate_model_100_times(model, mode):
    eval_env = CollisionEnv(mode=mode)
    successes = 0
    for _ in range(100):
        obs, _ = eval_env.reset()
        done = False
        steps = 0
        while not done and steps < 100:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            steps += 1
        
        collision = info.get("collision", False)
        if mode == "Avoid Collision" and not collision:
            successes += 1
        elif mode == "Seek Collision" and collision:
            successes += 1
            
    return successes  # Out of 100, this is directly the percentage

# --- 4. INITIALIZE STREAMLIT SESSION STATE ---
if "model" not in st.session_state:
    st.session_state.model = None
if "model_mode" not in st.session_state:
    st.session_state.model_mode = None
if "accuracy_history" not in st.session_state:
    st.session_state.accuracy_history = []
if "episode_outcomes" not in st.session_state:
    st.session_state.episode_outcomes = []
if "total_episodes_completed" not in st.session_state:
    st.session_state.total_episodes_completed = 0
if "sessions_trained" not in st.session_state:
    st.session_state.sessions_trained = 0
if "last_eval_score" not in st.session_state:
    st.session_state.last_eval_score = None

# --- 5. STREAMLIT UI DESIGN ---
st.title("🏃‍♂️ Dynamic PPO Training & Accuracy Tracker")
st.write("Train your model continuously, change learning rates on the fly, and watch performance metrics adjust in real-time.")

# Sidebar Controls
st.sidebar.header("🔧 Hyperparameters & Settings")
goal_mode = st.sidebar.selectbox("Select Agent Goal", ("Avoid Collision", "Seek Collision"))
learning_rate = st.sidebar.slider("Learning Rate", min_value=0.0001, max_value=0.0100, value=0.0030, step=0.0002, format="%.4f")
timesteps = st.sidebar.slider("Steps per Training Session", min_value=1000, max_value=20000, value=5000, step=1000)

# Reset Button Layout
st.sidebar.markdown("---")
if st.sidebar.button("🧹 Reset Model & History", type="primary"):
    st.session_state.model = None
    st.session_state.model_mode = None
    st.session_state.accuracy_history = []
    st.session_state.episode_outcomes = []
    st.session_state.total_episodes_completed = 0
    st.session_state.sessions_trained = 0
    st.session_state.last_eval_score = None
    st.success("Environment and training logs reset successfully!")
    st.rerun()

# Display current session stats
if st.session_state.sessions_trained > 0:
    c1, c2, c3 = st.columns(3)
    c1.metric("Sessions Completed", st.session_state.sessions_trained)
    c2.metric("Total Episodes Logged", st.session_state.total_episodes_completed)
    if st.session_state.last_eval_score is not None:
        c3.metric("Post-Session Accuracy", f"{st.session_state.last_eval_score}%")

# Main action interfaces
col1, col2 = st.columns(2)

with col1:
    if st.button("🏋️‍♂️ Train Model"):
        # Auto-reset if user switches modes mid-training session
        if st.session_state.model_mode != goal_mode and st.session_state.model is not None:
            st.warning("Goal mode changed! Resetting model network for the new objective.")
            st.session_state.model = None
            st.session_state.accuracy_history = []
            st.session_state.episode_outcomes = []
            st.session_state.total_episodes_completed = 0
            
        env = CollisionEnv(mode=goal_mode)
        
        # Instantiate new model or update existing model parameters to allow continuous training
        if st.session_state.model is None:
            st.session_state.model = PPO("MlpPolicy", env, verbose=0, learning_rate=learning_rate)
            st.session_state.model_mode = goal_mode
        else:
            # Update the environment reference and manually adjust learning rate in the active optimizer
            st.session_state.model.set_env(env)
            st.session_state.model.learning_rate = learning_rate
            for param_group in st.session_state.model.policy.optimizer.param_groups:
                param_group['lr'] = learning_rate
                
        st.write("### 📈 Live Performance Graph")
        chart_space = st.empty()
        status_space = st.empty()
        
        # Pre-populate chart if historical points exist
        if st.session_state.accuracy_history:
            df_init = pd.DataFrame(st.session_state.accuracy_history)
            chart_space.line_chart(df_init.set_index("Episode"))
            
        # Execute training with callback hooks for dynamic streaming data
        callback = DynamicProgressCallback(goal_mode, chart_space, status_space, timesteps)
        st.session_state.model.learn(total_timesteps=timesteps, callback=callback)
        st.session_state.sessions_trained += 1
        
        # Post-training evaluation sequence
        status_space.text("Evaluating model stability across 100 independent runs...")
        eval_score = evaluate_model_100_times(st.session_state.model, goal_mode)
        st.session_state.last_eval_score = eval_score
        st.rerun()

with col2:
    sim_disabled = st.session_state.model is None or st.session_state.model_mode != goal_mode
    if st.button("🎮 Watch Agent Run", disabled=sim_disabled):
        st.subheader("Live Simulation Visualizer")
        eval_env = CollisionEnv(mode=goal_mode)
        obs, _ = eval_env.reset()
        
        distance_metric = st.metric(label="Distance to Danger", value=f"{obs[0]:.2f} m")
        progress_visual = st.progress(int(obs[0]))
        
        done = False
        step = 0
        while not done and step < 100:
            action, _ = st.session_state.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            
            distance_metric.metric(label="Distance to Danger", value=f"{obs[0]:.2f} m")
            progress_visual.progress(int(obs[0]))
            step += 1
            time.sleep(0.08)
            
        if info.get("collision", False):
            if goal_mode == "Avoid Collision":
                st.error("💥 Crash! The agent hit the obstacle.")
            else:
                st.success("🎯 Target Hit! Successfully caused a collision.")
        else:
            if goal_mode == "Avoid Collision":
                st.success("🎉 Safe! The agent completely avoided collision.")
            else:
                st.error("⌛ Time-out! The agent missed the target.")

# Static Fallback Warning UI
if st.session_state.model is None:
    st.info("The neural network is uninitialized. Select your target configuration and click **Train Model** to generate and chart tracking curves.")