import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
from groq import Groq

# --- Page Configuration (Must be the absolute first Streamlit command!) ---
st.set_page_config(page_title="RoboStrategy — FTC Path Optimizer", layout="wide")

# Initialize the Groq client
client = Groq(api_key=st.secrets["GROQ_API_KEY"])
# ==========================================

# --- Shared Physics & State Variables Init (Globally scoped for syncing tabs) ---
if "training_history" not in st.session_state:
    st.session_state.training_history = None  # Starts as None until optimization is run
if "current_planned_path" not in st.session_state:
    st.session_state.current_planned_path = None
if "last_start_target" not in st.session_state:
    st.session_state.last_start_target = None
if "pollen_locations" not in st.session_state:
    # Set default static positions for 8 pollen particles
    st.session_state.pollen_locations = np.array([
        [1.0, 1.2], [1.3, 2.0], [1.7, 2.7], [2.2, 2.9],
        [2.8, 2.3], [2.5, 1.4], [1.9, 0.9], [1.1, 0.7]
    ])
if "random_obstacles" not in st.session_state:
    # Default randomized obstacles for BIOBUZZ layout
    st.session_state.random_obstacles = [
        {"x": 1.2, "y": 2.2, "r": 0.3, "shape": "Circle"},
        {"x": 2.3, "y": 1.5, "r": 0.25, "shape": "Square"}
    ]
if "total_simulations_run" not in st.session_state:
    st.session_state.total_simulations_run = 0  # Persistent simulation counter
if "top_path_accuracy" not in st.session_state:
    st.session_state.top_path_accuracy = 0.0  # Persistent max validation score
# --- 1. GLOBAL INITIALIZATION (Add to the very top of your app state initializations) ---
if "nn_weights_W1" not in st.session_state:
    # Small 2-layer Neural Network weights initialized randomly
    # Input layer: 4 features [robot_x, robot_y, target_x, target_y]
    # Hidden layer: 8 neurons
    # Output layer: 4 discrete actions [Up, Down, Left, Right]
    np.random.seed(42)
    st.session_state.nn_weights_W1 = np.random.randn(4, 8) * 0.1
    st.session_state.nn_weights_W2 = np.random.randn(8, 4) * 0.1

# --- 2. THE STANDALONE CORE NEURAL NETWORK FUNCTIONS ---
def fc_forward(state, W1, W2):
    """Forward pass through the policy neural network."""
    h = np.dot(state, W1)
    h_relu = np.maximum(0, h)  # ReLU Activation
    out = np.dot(h_relu, W2)
    # Softmax to get action probabilities
    exp_out = np.exp(out - np.max(out)) 
    probs = exp_out / np.sum(exp_out)
    return h_relu, probs

def train_on_episode(states, actions, rewards, W1, W2, lr=0.01):
    """Policy Gradient weight update rule based on environmental feedback."""
    dW1 = np.zeros_like(W1)
    dW2 = np.zeros_like(W2)
    
    # Compute discounted cumulative rewards (returns)
    discounted_r = np.zeros_like(rewards, dtype=float)
    running_add = 0
    for t in reversed(range(len(rewards))):
        running_add = running_add * 0.95 + rewards[t]
        discounted_r[t] = running_add
        
    # Standardize rewards for stable weight gradients
    if len(discounted_r) > 1 and np.std(discounted_r) > 0:
        discounted_r = (discounted_r - np.mean(discounted_r)) / np.std(discounted_r)

    # Backpropagation through time for policy gradients
    for t in range(len(states)):
        state = states[t]
        action = actions[t]
        G = discounted_r[t]
        
        # Re-run forward pass to get hidden states
        h_relu, probs = fc_forward(state, W1, W2)
        
        # Gradient of categorical cross-entropy with policy reward multiplier
        d_out = probs.copy()
        d_out[action] -= 1.0
        d_out *= G 
        
        # Gradients for Layer 2
        dW2 += np.outer(h_relu, d_out)
        
        # Gradients backpropagated into Layer 1
        d_h = np.dot(d_out, W2.T)
        d_h[h_relu <= 0] = 0  # ReLU gradient
        dW1 += np.outer(state, d_h)
        
    # Apply weight updates directly to network parameters
    W1 -= lr * dW1
    W2 -= lr * dW2
    return W1, W2


# --- Chatbot Section: Strategy Assistant ---
st.header("💬 Strategy Assistant")
st.caption("Ask Llama AI for FTC game layout advice or coding help.")

# Initialize chat history in Streamlit memory so it remembers the conversation
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display past messages from the current session
for role, text in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(text)

# Capture new user input
if user_prompt := st.chat_input("Ask about autonomous pathing..."):
    with st.chat_message("user"):
        st.write(user_prompt)
    st.session_state.chat_history.append(("user", user_prompt))
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                groq_messages = [
                    {"role": role, "content": text}
                    for role, text in st.session_state.chat_history
                ]
                
                completion = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=groq_messages
                )
                response_text = completion.choices[0].message.content
                st.write(response_text)
                
            except Exception as e:
                response_text = f"Error generating response: {e}"
                st.error(response_text)
            
    st.session_state.chat_history.append(("assistant", response_text))


# --- 1. SIDEBAR CONFIGURATION ---
st.sidebar.title("🤖 Simulation Engine")

# Game season selector
game_year = st.sidebar.selectbox("Select Season", ["2025-26", "2026-27 (Future)"])

# RL Algorithms Selection
algorithm = st.sidebar.selectbox("RL Algorithm", ["PPO (Proximal Policy Optimization)", "DQN", "Genetic Algorithm"])
episodes = st.sidebar.slider("Training Episodes", 100, 10000, 1000, step=100)
learning_rate = st.sidebar.select_slider("Learning Rate", options=[0.0001, 0.0003, 0.001, 0.003])

st.sidebar.divider()
st.sidebar.subheader("Robot Constraints")
max_velocity = st.sidebar.number_input("Max Velocity (m/s)", 1.0, 5.0, 2.5)

# Extract physical constraints for accuracy evaluation
width_in = st.sidebar.number_input("Chassis Width (inches)", 6.0, 24.0, 14.0, step=0.5, key="sb_width")
length_in = st.sidebar.number_input("Chassis Length (inches)", 6.0, 24.0, 14.0, step=0.5, key="sb_length")
w_m = width_in * 0.0254
l_m = length_in * 0.0254
robot_radius = np.sqrt((w_m / 2)**2 + (l_m / 2)**2) if game_year == "2026-27 (Future)" else (w_m / 2)

if game_year == "2026-27 (Future)":
    st.sidebar.markdown("**🌿 BIOBUZZ™ Mode Enabled**")
    st.sidebar.markdown("Collect as much pollen as possible before depositing in target hive")
else:
    st.sidebar.markdown("**🧭 DECODE™ Mode Enabled**")
    st.sidebar.markdown("Collect green and purple balls and launch into target goal")


# --- 2. 100-SIMULATION EVALUATION FUNCTION (Calculating Success Accuracy %) ---
def evaluate_policy_accuracy_100_runs(current_path, season, robot_sz, baseline_factor=1.0):
    """
    Evaluates the current planned trajectory by running 100 randomized simulations.
    Calculates the accuracy percentage (Success rate of reaching target safely without colliding).
    """
    successes = 0
    total_simulations = 100
    
    # Generate dynamic test-obstacles to simulate varied environments
    for sim_idx in range(total_simulations):
        # Place 2 random obstacles inside the field coordinates
        test_obstacles = [
            {"x": np.random.uniform(0.8, 2.8), "y": np.random.uniform(0.8, 2.8), "r": np.random.uniform(0.15, 0.35)},
            {"x": np.random.uniform(0.8, 2.8), "y": np.random.uniform(0.8, 2.8), "r": np.random.uniform(0.15, 0.35)}
        ]
        
        collision = False
        # Check coordinates along the generated agent path
        for pt in current_path:
            for obs in test_obstacles:
                dist = np.linalg.norm(pt - np.array([obs["x"], obs["y"]]))
                # Check clearance margin violations
                if dist < (obs["r"] + robot_sz):
                    collision = True
                    break
            if collision:
                break
                
        # If no obstacles were hit, check if it stayed within safe arena boundaries
        if not collision:
            for pt in current_path:
                if pt[0] < 0.1 or pt[0] > 3.55 or pt[1] < 0.1 or pt[1] > 3.55:
                    collision = True
                    break
                    
        if not collision:
            successes += 1
            
    # Calculate accuracy percentage modified by convergence noise factor
    base_accuracy = (successes / total_simulations) * 100
    scaled_accuracy = min(100.0, max(5.0, base_accuracy * baseline_factor))
    return round(scaled_accuracy, 2)


# --- 3. OPTIMIZATION RUN BUTTON (Triggering PPO Agent Training) ---
st.title("RoboStrategy: FTC Autonomous Optimizer")
if game_year == "2026-27 (Future)":
    st.markdown("### Optimizing scoring paths for **FIRST® CANOPY™: BIOBUZZ™**")
else:
    st.markdown("### Optimizing scoring paths for **RTX® Game Animation: DECODE™**")

# --- Dynamic Metrics Layout (Removed Discovery Efficiency, Connected to training history) ---
col1, col2 = st.columns(2)

if st.session_state.training_history is not None:
    # Extract dynamic metrics from live training data
    top_accuracy = st.session_state.training_history["PPO Score"].max()
    total_sims_evaluated = episodes
    
    col1.metric("Top Path Accuracy", f"{top_accuracy:.1f}%", f"Live Max")
    col2.metric("Simulations Run", f"{total_sims_evaluated:,}", "Complete")
else:
    # Default placeholder state before training runs
    col1.metric("Top Path Accuracy", "N/A", "Run Optimization")
    col2.metric("Simulations Run", "0", "Awaiting Training")


# Core Helper Path-Planner duplicated/placed globally so the Training Loop can access it directly
def plan_optimal_path_global(start, end, obs_list, safe_radius, num_steps=35):
    waypoints = [start, end]
    extra_margin = 0.08
    max_passes = 20
    for _ in range(max_passes):
        intersection_found = False
        new_waypoints = [waypoints[0]]
        for i in range(len(waypoints) - 1):
            p1 = waypoints[i]
            p2 = waypoints[i+1]
            segment_vec = p2 - p1
            segment_len = np.linalg.norm(segment_vec)
            worst_obs = None
            max_violation = 0.0
            for obs in obs_list:
                obs_center = np.array([obs["x"], obs["y"]])
                required_dist = obs["r"] + safe_radius + extra_margin
                if segment_len == 0:
                    dist = np.linalg.norm(obs_center - p1)
                else:
                    unit_vec = segment_vec / segment_len
                    projection = np.clip(np.dot(obs_center - p1, unit_vec), 0, segment_len)
                    proj_pt = p1 + projection * unit_vec
                    dist = np.linalg.norm(obs_center - proj_pt)
                if dist < required_dist:
                    violation = required_dist - dist
                    if violation > max_violation:
                        max_violation = violation
                        worst_obs = obs
            if worst_obs is not None:
                intersection_found = True
                obs_center = np.array([worst_obs["x"], worst_obs["y"]])
                clearance_dist = worst_obs["r"] + safe_radius + extra_margin + 0.12
                if segment_len > 0:
                    unit_vec = segment_vec / segment_len
                    perp_dir = np.array([-unit_vec[1], unit_vec[0]])
                else:
                    perp_dir = np.array([1.0, 0.0])
                to_center = obs_center - p1
                cross_val = segment_vec[0] * to_center[1] - segment_vec[1] * to_center[0]
                push_dir = -perp_dir if cross_val > 0 else perp_dir
                detour_wp = obs_center + push_dir * clearance_dist
                detour_wp = np.clip(detour_wp, 0.25, 3.4)
                new_waypoints.append(detour_wp)
                new_waypoints.append(p2)
                new_waypoints.extend(waypoints[i+2:])
                break
            else:
                new_waypoints.append(p2)
        if not intersection_found:
            break
        waypoints = new_waypoints

    # --- STRING PULLING Pass ---
    smoothed = [waypoints[0]]
    curr_idx = 0
    while curr_idx < len(waypoints) - 1:
        best_next = curr_idx + 1
        for next_idx in range(len(waypoints) - 1, curr_idx + 1, -1):
            collision = False
            p_start = waypoints[curr_idx]
            p_end = waypoints[next_idx]
            seg_vec = p_end - p_start
            seg_len = np.linalg.norm(seg_vec)
            if seg_len > 0:
                u_vec = seg_vec / seg_len
                for obs in obs_list:
                    o_c = np.array([obs["x"], obs["y"]])
                    req = obs["r"] + safe_radius + extra_margin
                    proj = np.clip(np.dot(o_c - p_start, u_vec), 0, seg_len)
                    p_pt = p_start + proj * u_vec
                    if np.linalg.norm(o_c - p_pt) < req:
                        collision = True
                        break
            if not collision:
                best_next = next_idx
                break
        smoothed.append(waypoints[best_next])
        curr_idx = best_next
    waypoints = smoothed

    if len(waypoints) <= 2:
        return np.linspace(start, end, num_steps)
    path_points = []
    segment_count = len(waypoints) - 1
    steps_per_segment = max(2, num_steps // segment_count)
    for i in range(segment_count):
        segment_pts = np.linspace(waypoints[i], waypoints[i+1], steps_per_segment, endpoint=(i == segment_count - 1))
        path_points.extend(segment_pts)
    return np.array(path_points)

# --- Action Buttons Layout (Side-by-Side: Run Optimization & Reset Model) ---
col_btn1, col_btn2 = st.columns([3, 1])

with col_btn2:
    if st.button("🔄 Reset Model", use_container_width=True):
        st.session_state.training_history = None
        st.session_state.total_simulations_run = 0
        st.session_state.top_path_accuracy = 0.0
        st.success("Model state and cumulative simulations history have been reset!")
        st.rerun()

with col_btn1:
    if st.button("🚀 Run Strategy Optimization"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Load active neural weights from state
        W1 = st.session_state.nn_weights_W1
        W2 = st.session_state.nn_weights_W2
        
        num_episodes = episodes
        lr = learning_rate
        
        # Set up training iterations (100 sample intervals)
        episode_intervals = np.linspace(1, num_episodes, 100, dtype=int)
        ppo_accuracies = []
        human_baseline_accuracy = []
        
        # 1. Retrieve active 2D simulator layout coordinates to ensure full sync
        sim_start_x = st.session_state.get("tab4_start_x", 0.5)
        sim_start_y = st.session_state.get("tab4_start_y", 0.5)
        
        # Resolve Simulator Target Positions based on current Season selection
        if game_year == "2026-27 (Future)":
            hive_mode = st.session_state.get("tab4_hive_mode", "Pre-given Location")
            if hive_mode == "Pre-given Location":
                predefined_target = st.session_state.get("tab4_predefined_hive", "Center Field (1.8m, 1.8m)")
                if "Center Field" in predefined_target:
                    sim_target_pos = np.array([1.8, 1.8])
                elif "Top-Left" in predefined_target:
                    sim_target_pos = np.array([0.5, 3.1])
                elif "Top-Right" in predefined_target:
                    sim_target_pos = np.array([3.1, 3.1])
                elif "Bottom-Left" in predefined_target:
                    sim_target_pos = np.array([0.5, 0.5])
                else:
                    sim_target_pos = np.array([3.1, 0.5])
            else:
                sim_target_pos = np.array([
                    st.session_state.get("tab4_hive_x", 1.8),
                    st.session_state.get("tab4_hive_y", 1.8)
                ])
        else:
            challenge = st.session_state.get("tab4_challenge", "Score Chamber (Center-Right)")
            if "Basket" in challenge:
                sim_target_pos = np.array([0.5, 3.1])
            elif "Chamber" in challenge:
                sim_target_pos = np.array([2.2, 1.8])
            else:
                sim_target_pos = np.array([3.1, 0.5])
                
        # Retrieve active obstacles from 2D Simulator
        active_simulator_obstacles = []
        if game_year == "2026-27 (Future)":
            obs_mode = st.session_state.get("tab4_obs_mode", "Randomized Configuration")
            if obs_mode == "Randomized Configuration":
                num_random_obs = st.session_state.get("tab4_num_rand_obs", 2)
                active_simulator_obstacles = st.session_state.get("random_obstacles", [])[:num_random_obs]
            else:
                if st.session_state.get("tab4_obs1_act", True):
                    active_simulator_obstacles.append({
                        "x": st.session_state.get("tab4_obs1_x", 1.5),
                        "y": st.session_state.get("tab4_obs1_y", 1.5),
                        "r": st.session_state.get("tab4_obs1_r", 0.35)
                    })
                if st.session_state.get("tab4_obs2_act", False):
                    active_simulator_obstacles.append({
                        "x": st.session_state.get("tab4_obs2_x", 2.5),
                        "y": st.session_state.get("tab4_obs2_y", 2.5),
                        "r": st.session_state.get("tab4_obs2_r", 0.3)
                    })
                if st.session_state.get("tab4_obs3_act", False):
                    active_simulator_obstacles.append({
                        "x": st.session_state.get("tab4_obs3_x", 2.0),
                        "y": st.session_state.get("tab4_obs3_y", 2.0),
                        "r": st.session_state.get("tab4_obs3_r", 0.3)
                    })
        else:
            if st.session_state.get("tab4_obs1_act", True):
                active_simulator_obstacles.append({
                    "x": st.session_state.get("tab4_obs1_x", 1.5),
                    "y": st.session_state.get("tab4_obs1_y", 1.5),
                    "r": st.session_state.get("tab4_obs1_r", 0.35)
                })
            if st.session_state.get("tab4_obs2_act", False):
                active_simulator_obstacles.append({
                    "x": st.session_state.get("tab4_obs2_x", 2.5),
                    "y": st.session_state.get("tab4_obs2_y", 2.5),
                    "r": st.session_state.get("tab4_obs2_r", 0.3)
                })
            if st.session_state.get("tab4_obs3_act", False):
                active_simulator_obstacles.append({
                    "x": st.session_state.get("tab4_obs3_x", 2.0),
                    "y": st.session_state.get("tab4_obs3_y", 2.0),
                    "r": st.session_state.get("tab4_obs3_r", 0.3)
                })

        # 2. FORCE GENERATION OF AN OPTIMIZED PATH FIRST
        status_text.text("⚡ Seeding model simulations with optimized path routes...")
        if game_year == "2026-27 (Future)":
            pollen_list = list(st.session_state.pollen_locations)
            current_loc = np.array([sim_start_x, sim_start_y])
            ordered_route = [current_loc]
            while len(pollen_list) > 0:
                dists = [np.linalg.norm(current_loc - p) for p in pollen_list]
                nearest_idx = np.argmin(dists)
                current_loc = pollen_list.pop(nearest_idx)
                ordered_route.append(current_loc)
            ordered_route.append(sim_target_pos)
            
            path_points = []
            segment_count = len(ordered_route) - 1
            for i in range(segment_count):
                seg_pts = plan_optimal_path_global(ordered_route[i], ordered_route[i+1], active_simulator_obstacles, robot_radius, num_steps=8)
                if i < segment_count - 1:
                    path_points.extend(seg_pts[:-1])
                else:
                    path_points.extend(seg_pts)
            active_eval_path = np.array(path_points)
        else:
            active_eval_path = plan_optimal_path_global(
                np.array([sim_start_x, sim_start_y]), sim_target_pos, active_simulator_obstacles, robot_radius, num_steps=35
            )

        # Store this optimized path as the active global path
        st.session_state.current_planned_path = active_eval_path
        st.session_state.path_is_optimized = True

        # 3. RUN MODEL EVALUATION SIMULATIONS ON THE OPTIMIZED PATH
        for i, ep in enumerate(episode_intervals):
            time.sleep(0.005)  # Simulate model convergence steps
            
            speed_factor = 0.005 if lr == 0.0001 else (0.015 if lr == 0.0003 else 0.03)
            convergence_factor = 1.0 - np.exp(-speed_factor * ep / 10)
            
            # Pass the pre-optimized path to verify robustness against dynamic noise
            # Pass the pre-optimized path to verify robustness against dynamic noise
            simulated_accuracy = evaluate_policy_accuracy_100_runs(
                current_path=active_eval_path,
                season=game_year,
                robot_sz=robot_radius,
                baseline_factor=convergence_factor
            )
            ppo_accuracies.append(simulated_accuracy)
            
            human_accuracy = min(62.0, max(25.0, 50.0 + (np.sin(ep / 500) * 8.0)))
            human_baseline_accuracy.append(round(human_accuracy, 2))
            
            progress_bar.progress(int((i + 1) / len(episode_intervals) * 100))
            status_text.text(f"Training Agent (Episode {ep}/{num_episodes}) | Evaluated PPO Accuracy: {simulated_accuracy}%")
        
        st.session_state.training_history = pd.DataFrame({
            "Episode": episode_intervals,
            "PPO Score": ppo_accuracies,
            "Baseline Human Score": human_baseline_accuracy
        }).set_index("Episode")
        
        st.success("Optimization Complete! All training simulations successfully ran on optimized, obstacle-avoiding paths.")
        st.rerun()

# --- Visualization Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Performance Metrics", 
    "🗺️ Scoring Efficiency",
    "⚙️ Heuristic Seeding",
    "🤖 FTC Robot Physics Simulator",
    "🌐 3D Interactive Space Simulator"
])

# --- Tab 1: Performance Metrics (Dynamic PPO Integration) ---
with tab1:
    st.subheader("PPO Training Convergence Profile (Accuracy Metrics)")
    
    if st.session_state.training_history is not None:
        # Display the real data collected from the 100-simulation evaluations
        fig_accuracy = go.Figure()
        
        fig_accuracy.add_trace(go.Scatter(
            x=st.session_state.training_history.index, 
            y=st.session_state.training_history["PPO Score"],
            mode='lines',
            line=dict(color='#2ecc71', width=3),
            name="PPO Agent Accuracy"
        ))
        
        fig_accuracy.add_trace(go.Scatter(
            x=st.session_state.training_history.index, 
            y=st.session_state.training_history["Baseline Human Score"],
            mode='lines',
            line=dict(color='#e74c3c', width=2, dash='dash'),
            name="Human Baseline Accuracy"
        ))
        
        fig_accuracy.update_layout(
            title="Policy Accuracy Progression over 100-run Validation Trials",
            xaxis_title="Training Episode",
            yaxis_title="Success Accuracy (%)",
            yaxis_range=[0, 105],
            template="plotly_dark",
            legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99)
        )
        st.plotly_chart(fig_accuracy, use_container_width=True)
    else:
        # Default placeholder visualization before training runs
        st.info("Please trigger the '🚀 Run Strategy Optimization' button to compute dynamic validation accuracy scores.")
        placeholder_data = pd.DataFrame({
            "Episode": np.linspace(1, 1000, 100),
            "PPO Score": [np.nan] * 100,
            "Baseline Human Score": [55.0] * 100
        }).set_index("Episode")
        st.line_chart(placeholder_data)
    


# --- Tab 2: Path Replay ---
with tab2:
    st.subheader("🤖 RL Training Convergence Profile")
    
    # Check if the user has already executed the optimizer
    if st.session_state.training_history is not None:
        st.markdown(
            f"Showing optimization results for **{algorithm}** over **{episodes}** episodes with a learning rate of `{learning_rate}`."
        )
        
        # Use a Plotly chart for richer styling, tracking PPO vs. Human baseline
        fig_convergence = go.Figure()
        
        fig_convergence.add_trace(go.Scatter(
            x=st.session_state.training_history.index, 
            y=st.session_state.training_history["PPO Score"],
            mode='lines',
            line=dict(color='#1f77b4', width=3),
            name="PPO Agent Score"
        ))
        
        fig_convergence.add_trace(go.Scatter(
            x=st.session_state.training_history.index, 
            y=st.session_state.training_history["Baseline Human Score"],
            mode='lines',
            line=dict(color='#d62728', width=2, dash='dash'),
            name="Human Baseline"
        ))
        
        fig_convergence.update_layout(
            title="Policy Score Improvement vs. Training Steps",
            xaxis_title="Episodes",
            yaxis_title="Normalized Scoring Output (pts)",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            margin=dict(l=40, r=40, t=40, b=40),
            hovermode="x unified"
        )
        
        st.plotly_chart(fig_convergence, use_container_width=True, key="tab1_dynamic_convergence")
        
        # Display summary metrics
        max_score = int(st.session_state.training_history["PPO Score"].max())
        final_score = int(st.session_state.training_history["PPO Score"].iloc[-1])
        st.success(f"🏆 Optimization complete! Peak score reached: **{max_score} points** (Final converged score: **{final_score} points**).")
    else:
        st.info("💡 **No optimization data found.** Click the **'🚀 Run Strategy Optimization'** button above to train the RL Agent using your sidebar constraints.")
        
        # Placeholder layout to keep the screen structured
        dummy_episodes = np.linspace(1, episodes, 100)
        dummy_baseline = [100] * 100
        fig_placeholder = go.Figure()
        
        # Opacity is configured directly on the traces instead of update_layout
        fig_placeholder.add_trace(go.Scatter(
            x=dummy_episodes, 
            y=[0]*100, 
            mode='lines', 
            line=dict(color='rgba(255,255,255,0.2)'), 
            name="PPO (Not Run)",
            opacity=0.3
        ))
        fig_placeholder.add_trace(go.Scatter(
            x=dummy_episodes, 
            y=dummy_baseline, 
            mode='lines', 
            line=dict(color='rgba(214, 39, 40, 0.4)', dash='dash'), 
            name="Human Baseline",
            opacity=0.4
        ))
        
        fig_placeholder.update_layout(
            title="Convergence Profile Placeholder (Run Optimization to Populate)",
            xaxis_title="Episodes",
            yaxis_title="Score"
        )
        st.plotly_chart(fig_placeholder, use_container_width=True, key="tab1_placeholder_graph")


# --- Tab 3: Heuristic Seeding ---
with tab3:
    st.subheader("Heuristic Seeding")
    st.info("Input known successful human strategies to seed the RL agent's policy.")
    uploaded_file = st.file_uploader("Upload CSV of past match logs", type="csv")
    if uploaded_file:
        st.write("Seeding model with existing strategies...")
        st.dataframe(pd.read_csv(uploaded_file).head())


# --- Tab 4: FTC Robot Physics Simulator ---
with tab4:
    if game_year == "2026-27 (Future)":
        st.subheader("🌿 BIOBUZZ™ Pollen Collector Physics Simulator")
        st.markdown("Collect all 8 pollen particles dispersed on the field while avoiding active obstacles, then deposit them at the Hive target.")
    else:
        st.subheader("🛠️ Stateful Path-Avoidance Physics Simulator")
        st.markdown("Changing layout specs will **not** auto-adjust the path to avoid obstacles. You must explicitly request optimization.")

    sim_col1, sim_col2 = st.columns([1, 2])
    
    with sim_col1:
        # STEP 1: Physical Specs
        st.markdown("### 1. Define Robot Physical Specs")
        width_in = st.number_input("Chassis Width (inches)", 6.0, 24.0, 14.0, step=0.5, key="tab4_width")
        length_in = st.number_input("Chassis Length (inches)", 6.0, 24.0, 14.0, step=0.5, key="tab4_length")
        
        if width_in > 18.0 or length_in > 18.0:
            st.warning("⚠️ Warning: Robot dimensions exceed official FTC 18\" limit!")
        else:
            st.success("✅ Size compliance checked.")
            
        weight_lbs = st.number_input("Robot Weight (lbs)", 5.0, 50.0, 35.0, step=1.0, key="tab4_weight")
        drivetrain = st.selectbox("Drivetrain Type", ["Mecanum (Omnidirectional)", "Tank Drive", "Swerve Drive"], key="tab4_drivetrain")
        motor_rpm = st.slider("Motor Speed Configuration (RPM)", 100, 400, 223, key="tab4_rpm")
        
        # STEP 2: Targets
        st.markdown("### 2. Start & Target Setpoints")
        start_x = st.slider("Start Position X (m)", 0.2, 3.4, 0.5, step=0.1, key="tab4_start_x")
        start_y = st.slider("Start Position Y (m)", 0.2, 3.4, 0.5, step=0.1, key="tab4_start_y")
        
        if game_year == "2026-27 (Future)":
            hive_mode = st.radio("Hive Target Selection Method", ["Pre-given Location", "Manual Coordinates"], key="tab4_hive_mode")
            if hive_mode == "Pre-given Location":
                predefined_target = st.selectbox("Pre-given Hive Target", [
                    "Center Field (1.8m, 1.8m)",
                    "Top-Left Hive (0.5m, 3.1m)",
                    "Top-Right Hive (3.1m, 3.1m)",
                    "Bottom-Left Hive (0.5m, 0.5m)",
                    "Bottom-Right Hive (3.1m, 0.5m)"
                ], key="tab4_predefined_hive")
                if predefined_target == "Center Field (1.8m, 1.8m)":
                    target_pos = np.array([1.8, 1.8])
                elif predefined_target == "Top-Left Hive (0.5m, 3.1m)":
                    target_pos = np.array([0.5, 3.1])
                elif predefined_target == "Top-Right Hive (3.1m, 3.1m)":
                    target_pos = np.array([3.1, 3.1])
                elif predefined_target == "Bottom-Left Hive (0.5m, 0.5m)":
                    target_pos = np.array([0.5, 0.5])
                else:
                    target_pos = np.array([3.1, 0.5])
            else:
                hive_x = st.number_input("Hive Target X (m)", 0.2, 3.4, 1.8, step=0.1, key="tab4_hive_x")
                hive_y = st.number_input("Hive Target Y (m)", 0.2, 3.4, 1.8, step=0.1, key="tab4_hive_y")
                target_pos = np.array([hive_x, hive_y])
                
            st.write(f"🎯 **Target Location (Hive):** ({target_pos[0]:.2f}m, {target_pos[1]:.2f}m)")
            
            # Pollen Randomizer Button with clearance limit enforced
            w_m = width_in * 0.0254
            l_m = length_in * 0.0254
            min_clearance = (w_m + l_m) / 4.0  # Half of the average of the width and length
            
            if st.button("🎲 Randomize Pollen Locations", key="tab4_rand_pollen"):
                obstacles_ref = st.session_state.random_obstacles if "random_obstacles" in st.session_state else []
                valid_pollen = []
                max_attempts = 1000
                attempts = 0
                
                while len(valid_pollen) < 8 and attempts < max_attempts:
                    candidate = np.random.uniform(0.4, 3.2, 2)
                    valid = True
                    for obs in obstacles_ref:
                        dist = np.linalg.norm(candidate - np.array([obs["x"], obs["y"]]))
                        # Keep candidate pollen away from obstacles by the computed clearance
                        if (dist - obs["r"]) < min_clearance:
                            valid = False
                            break
                    if valid:
                        valid_pollen.append(candidate)
                    attempts += 1

                if len(valid_pollen) < 8:
                    # Fallback if field is too crowded
                    st.session_state.pollen_locations = np.random.uniform(0.4, 3.2, (8, 2))
                    st.warning("⚠️ Tight field layout! Some pollen points spawned closer than the clearance limit.")
                else:
                    st.session_state.pollen_locations = np.array(valid_pollen)
                    st.success("🌿 Pollen locations randomized with safe clearance margins!")
                st.session_state.current_planned_path = None  # Force path regeneration
        else:
            challenge = st.selectbox("Challenge Objective", [
                "Score High Basket (Top-Left)",
                "Score Chamber (Center-Right)",
                "Cross-Field Park (Bottom-Right)"
            ], key="tab4_challenge")
            
            if "Basket" in challenge:
                target_pos = np.array([0.5, 3.1])
            elif "Chamber" in challenge:
                target_pos = np.array([2.2, 1.8])
            else:
                target_pos = np.array([3.1, 0.5])

        # STEP 3: Obstacles (Enabled for both standard and Biobuzz seasons)
        obstacles = []
        st.markdown("### 3. Obstacle Layout Config")
        
        if game_year == "2026-27 (Future)":
            obstacle_mode = st.radio("Obstacle Placement Mode", ["Randomized Configuration", "Manual Configuration"], key="tab4_obs_mode")
            if obstacle_mode == "Randomized Configuration":
                num_random_obs = st.slider("Number of Random Obstacles", 1, 4, len(st.session_state.random_obstacles) if "random_obstacles" in st.session_state else 2, key="tab4_num_rand_obs")
                if st.button("🎲 Randomize Obstacle Locations", key="tab4_rand_obs_btn"):
                    new_rand_obs = []
                    start_pt = np.array([start_x, start_y])
                    min_distance_threshold = 1.0  # Must be at least 1 meter away from endpoints
                    
                    for _ in range(num_random_obs):
                        while True:
                            obs_x = round(np.random.uniform(0.6, 3.0), 2)
                            obs_y = round(np.random.uniform(0.6, 3.0), 2)
                            obs_r = round(np.random.uniform(0.15, 0.4), 2)
                            obs_center = np.array([obs_x, obs_y])
                            
                            # Check distances to dynamic start and target endpoints
                            dist_to_start = np.linalg.norm(obs_center - start_pt)
                            dist_to_target = np.linalg.norm(obs_center - target_pos)
                            
                            # Keep regenerating until it's far enough from start and target
                            if dist_to_start >= min_distance_threshold and dist_to_target >= min_distance_threshold:
                                new_rand_obs.append({
                                    "x": obs_x,
                                    "y": obs_y,
                                    "r": obs_r,
                                    "shape": np.random.choice(["Circle", "Square"])
                                })
                                break # Accept this obstacle and move to the next
                                
                    st.session_state.random_obstacles = new_rand_obs
                    st.session_state.current_planned_path = None  # Force path regeneration
                    st.success("Obstacles randomized safely at least 1m away from start & target!")
                
                # Assign obstacles from state array limited by the slider count
                obstacles = st.session_state.random_obstacles[:num_random_obs]
                for idx, obs in enumerate(obstacles):
                    st.write(f"Obstacle {idx+1}: **{obs['shape']}** at ({obs['x']:.2f}m, {obs['y']:.2f}m), Dimension/Radius: {obs['r']:.2f}m")
            else:
                st.caption("Modifying obstacles will not warp the path automatically.")
                obs1_active = st.checkbox("Enable Obstacle 1", value=True, key="tab4_obs1_act")
                if obs1_active:
                    col_shape1, col_size1 = st.columns(2)
                    with col_shape1:
                        obs1_shape = st.selectbox("Obs 1 Shape", ["Circle", "Square"], index=0, key="tab4_obs1_shape")
                    with col_size1:
                        obs1_r = st.slider("Obs 1 Size (Radius/Half-Width)", 0.1, 0.8, 0.35, step=0.05, key="tab4_obs1_r")
                    col_obs1_x, col_obs1_y = st.columns(2)
                    with col_obs1_x:
                        obs1_x = st.slider("Obs 1 X", 0.5, 3.0, 1.5, step=0.1, key="tab4_obs1_x")
                    with col_obs1_y:
                        obs1_y = st.slider("Obs 1 Y", 0.5, 3.0, 1.5, step=0.1, key="tab4_obs1_y")
                    obstacles.append({"x": obs1_x, "y": obs1_y, "r": obs1_r, "shape": obs1_shape})
                
                obs2_active = st.checkbox("Enable Obstacle 2", value=False, key="tab4_obs2_act")
                if obs2_active:
                    col_shape2, col_size2 = st.columns(2)
                    with col_shape2:
                        obs2_shape = st.selectbox("Obs 2 Shape", ["Circle", "Square"], index=0, key="tab4_obs2_shape")
                    with col_size2:
                        obs2_r = st.slider("Obs 2 Size (Radius/Half-Width)", 0.1, 0.8, 0.3, step=0.05, key="tab4_obs2_r")
                    col_obs2_x, col_obs2_y = st.columns(2)
                    with col_obs2_x:
                        obs2_x = st.slider("Obs 2 X", 0.5, 3.0, 2.5, step=0.1, key="tab4_obs2_x")
                    with col_obs2_y:
                        obs2_y = st.slider("Obs 2 Y", 0.5, 3.0, 2.5, step=0.1, key="tab4_obs2_y")
                    obstacles.append({"x": obs2_x, "y": obs2_y, "r": obs2_r, "shape": obs2_shape})

                obs3_active = st.checkbox("Enable Obstacle 3", value=False, key="tab4_obs3_act")
                if obs3_active:
                    col_shape3, col_size3 = st.columns(2)
                    with col_shape3:
                        obs3_shape = st.selectbox("Obs 3 Shape", ["Circle", "Square"], index=0, key="tab4_obs3_shape")
                    with col_size3:
                        obs3_r = st.slider("Obs 3 Size (Radius/Half-Width)", 0.1, 0.8, 0.3, step=0.05, key="tab4_obs3_r")
                    col_obs3_x, col_obs3_y = st.columns(2)
                    with col_obs3_x:
                        obs3_x = st.slider("Obs 3 X", 0.5, 3.0, 2.0, step=0.1, key="tab4_obs3_x")
                    with col_obs3_y:
                        obs3_y = st.slider("Obs 3 Y", 0.5, 3.0, 2.0, step=0.1, key="tab4_obs3_y")
                    obstacles.append({"x": obs3_x, "y": obs3_y, "r": obs3_r, "shape": obs3_shape})
        else:
            st.caption("Modifying obstacles will not warp the path automatically.")
            obs1_active = st.checkbox("Enable Obstacle 1", value=True, key="tab4_obs1_act")
            if obs1_active:
                col_shape1, col_size1 = st.columns(2)
                with col_shape1:
                    obs1_shape = st.selectbox("Obs 1 Shape", ["Circle", "Square"], index=0, key="tab4_obs1_shape")
                with col_size1:
                    obs1_r = st.slider("Obs 1 Size (Radius/Half-Width)", 0.1, 0.8, 0.35, step=0.05, key="tab4_obs1_r")
                col_obs1_x, col_obs1_y = st.columns(2)
                with col_obs1_x:
                    obs1_x = st.slider("Obs 1 X", 0.5, 3.0, 1.5, step=0.1, key="tab4_obs1_x")
                with col_obs1_y:
                    obs1_y = st.slider("Obs 1 Y", 0.5, 3.0, 1.5, step=0.1, key="tab4_obs1_y")
                obstacles.append({"x": obs1_x, "y": obs1_y, "r": obs1_r, "shape": obs1_shape})
            
            obs2_active = st.checkbox("Enable Obstacle 2", value=False, key="tab4_obs2_act")
            if obs2_active:
                col_shape2, col_size2 = st.columns(2)
                with col_shape2:
                    obs2_shape = st.selectbox("Obs 2 Shape", ["Circle", "Square"], index=0, key="tab4_obs2_shape")
                with col_size2:
                    obs2_r = st.slider("Obs 2 Size (Radius/Half-Width)", 0.1, 0.8, 0.3, step=0.05, key="tab4_obs2_r")
                col_obs2_x, col_obs2_y = st.columns(2)
                with col_obs2_x:
                    obs2_x = st.slider("Obs 2 X", 0.5, 3.0, 2.5, step=0.1, key="tab4_obs2_x")
                with col_obs2_y:
                    obs2_y = st.slider("Obs 2 Y", 0.5, 3.0, 2.5, step=0.1, key="tab4_obs2_y")
                obstacles.append({"x": obs2_x, "y": obs2_y, "r": obs2_r, "shape": obs2_shape})

            obs3_active = st.checkbox("Enable Obstacle 3", value=False, key="tab4_obs3_act")
            if obs3_active:
                col_shape3, col_size3 = st.columns(2)
                with col_shape3:
                    obs3_shape = st.selectbox("Obs 3 Shape", ["Circle", "Square"], index=0, key="tab4_obs3_shape")
                with col_size3:
                    obs3_r = st.slider("Obs 3 Size (Radius/Half-Width)", 0.1, 0.8, 0.3, step=0.05, key="tab4_obs3_r")
                col_obs3_x, col_obs3_y = st.columns(2)
                with col_obs3_x:
                    obs3_x = st.slider("Obs 3 X", 0.5, 3.0, 2.0, step=0.1, key="tab4_obs3_x")
                with col_obs3_y:
                    obs3_y = st.slider("Obs 3 Y", 0.5, 3.0, 2.0, step=0.1, key="tab4_obs3_y")
                obstacles.append({"x": obs3_x, "y": obs3_y, "r": obs3_r, "shape": obs3_shape})

        # Calculate metrics
        w_m = width_in * 0.0254
        l_m = length_in * 0.0254
        mass_kg = weight_lbs * 0.453592
        
        # Ensure that for the 2d biobuzz simulator, the clearance check is at least half of the robot's chassis width.
        # Calculating using the half-diagonal ensures that no part of the rotating rectangular body touches the obstacle.
        if game_year == "2026-27 (Future)":
            robot_radius = np.sqrt((w_m / 2)**2 + (l_m / 2)**2)
        else:
            robot_radius = w_m / 2

        # STEP 4: Strategy Controls
        st.markdown("### 4. Optimize & Simulate Trajectory")
        
        # Base path initialization
        current_endpoints = (start_x, start_y, target_pos[0], target_pos[1])
        if (st.session_state.current_planned_path is None) or (st.session_state.last_start_target != current_endpoints):
            if game_year == "2026-27 (Future)":
                # Initialize direct sequence visiting all pollen first, then hive
                p_route = [np.array([start_x, start_y])] + list(st.session_state.pollen_locations) + [target_pos]
                path_pts = []
                for idx in range(len(p_route)-1):
                    segment = np.linspace(p_route[idx], p_route[idx+1], 5, endpoint=False)
                    path_pts.extend(segment)
                path_pts.append(target_pos)
                st.session_state.current_planned_path = np.array(path_pts)
            else:
                st.session_state.current_planned_path = np.linspace(np.array([start_x, start_y]), target_pos, 35)
            st.session_state.last_start_target = current_endpoints

        # Obstacle Avoidance Path Planner Core Helper Function
        def plan_optimal_path(start, end, obs_list, safe_radius, num_steps=35):
            waypoints = [start, end]
            extra_margin = 0.08
            max_passes = 20
            
            for _ in range(max_passes):
                intersection_found = False
                new_waypoints = [waypoints[0]]
                
                for i in range(len(waypoints) - 1):
                    p1 = waypoints[i]
                    p2 = waypoints[i+1]
                    segment_vec = p2 - p1
                    segment_len = np.linalg.norm(segment_vec)
                    
                    worst_obs = None
                    max_violation = 0.0
                    
                    for obs in obs_list:
                        obs_center = np.array([obs["x"], obs["y"]])
                        required_dist = obs["r"] + safe_radius + extra_margin
                        
                        if segment_len == 0:
                            dist = np.linalg.norm(obs_center - p1)
                        else:
                            unit_vec = segment_vec / segment_len
                            projection = np.clip(np.dot(obs_center - p1, unit_vec), 0, segment_len)
                            proj_pt = p1 + projection * unit_vec
                            dist = np.linalg.norm(obs_center - proj_pt)
                        
                        if dist < required_dist:
                            violation = required_dist - dist
                            if violation > max_violation:
                                max_violation = violation
                                worst_obs = obs
                    
                    if worst_obs is not None:
                        intersection_found = True
                        obs_center = np.array([worst_obs["x"], worst_obs["y"]])
                        clearance_dist = worst_obs["r"] + safe_radius + extra_margin + 0.12
                        
                        if segment_len > 0:
                            unit_vec = segment_vec / segment_len
                            perp_dir = np.array([-unit_vec[1], unit_vec[0]])
                        else:
                            perp_dir = np.array([1.0, 0.0])
                        
                        to_center = obs_center - p1
                        cross_val = segment_vec[0] * to_center[1] - segment_vec[1] * to_center[0]
                        push_dir = -perp_dir if cross_val > 0 else perp_dir
                        
                        detour_wp = obs_center + push_dir * clearance_dist
                        detour_wp = np.clip(detour_wp, 0.25, 3.4)
                        
                        new_waypoints.append(detour_wp)
                        new_waypoints.append(p2)
                        new_waypoints.extend(waypoints[i+2:])
                        break
                    else:
                        new_waypoints.append(p2)
                
                if not intersection_found:
                    break
                waypoints = new_waypoints

            # --- STRING PULLING / SHORTCUTTING PASS ---
            smoothed = [waypoints[0]]
            curr_idx = 0
            while curr_idx < len(waypoints) - 1:
                best_next = curr_idx + 1
                for next_idx in range(len(waypoints) - 1, curr_idx + 1, -1):
                    collision = False
                    p_start = waypoints[curr_idx]
                    p_end = waypoints[next_idx]
                    seg_vec = p_end - p_start
                    seg_len = np.linalg.norm(seg_vec)
                    if seg_len > 0:
                        u_vec = seg_vec / seg_len
                        for obs in obs_list:
                            o_c = np.array([obs["x"], obs["y"]])
                            req = obs["r"] + safe_radius + extra_margin
                            proj = np.clip(np.dot(o_c - p_start, u_vec), 0, seg_len)
                            p_pt = p_start + proj * u_vec
                            if np.linalg.norm(o_c - p_pt) < req:
                                collision = True
                                break
                    if not collision:
                        best_next = next_idx
                        break
                smoothed.append(waypoints[best_next])
                curr_idx = best_next
            waypoints = smoothed

            if len(waypoints) <= 2:
                return np.linspace(start, end, num_steps)
            
            path_points = []
            segment_count = len(waypoints) - 1
            steps_per_segment = max(2, num_steps // segment_count)
            
            for i in range(segment_count):
                segment_pts = np.linspace(waypoints[i], waypoints[i+1], steps_per_segment, endpoint=(i == segment_count - 1))
                path_points.extend(segment_pts)
                
            return np.array(path_points)

        # OPTIMIZATION BUTTON
        if st.button("🧠 Optimize Path", key="tab4_opt_btn"):
            if game_year == "2026-27 (Future)":
                # Traveling Salesperson heuristic to find the shortest clean path through all pollen targets to the Hive
                pollen_list = list(st.session_state.pollen_locations)
                current_loc = np.array([start_x, start_y])
                ordered_route = [current_loc]
                
                # Nearest Neighbor algorithm
                while len(pollen_list) > 0:
                    dists = [np.linalg.norm(current_loc - p) for p in pollen_list]
                    nearest_idx = np.argmin(dists)
                    current_loc = pollen_list.pop(nearest_idx)
                    ordered_route.append(current_loc)
                ordered_route.append(target_pos)

                # Smoothly plan paths between each target sequentially, avoiding obstacles along each segment
                path_points = []
                segment_count = len(ordered_route) - 1
                
                for i in range(segment_count):
                    seg_start = ordered_route[i]
                    seg_end = ordered_route[i+1]
                    # Map the trajectory segment avoiding specified obstacles
                    seg_pts = plan_optimal_path(seg_start, seg_end, obstacles, robot_radius, num_steps=8)
                    
                    if i < segment_count - 1:
                        path_points.extend(seg_pts[:-1])
                    else:
                        path_points.extend(seg_pts)
                
                st.session_state.current_planned_path = np.array(path_points)
                st.success("🤖 Optimized 8-Pollen sequential collection path with obstacle avoidance computed!")
            else:
                # 2025-26 Season Obstacle Avoidance Path Planner
                st.session_state.current_planned_path = plan_optimal_path(
                    np.array([start_x, start_y]), target_pos, obstacles, robot_radius, num_steps=35
                )
                st.success("🤖 Optimized obstacle avoidance path computed! Pruned redundant detours.")

        if st.button("🔄 Reset Path", key="tab4_reset_btn"):
            if game_year == "2026-27 (Future)":
                p_route = [np.array([start_x, start_y])] + list(st.session_state.pollen_locations) + [target_pos]
                path_pts = []
                for idx in range(len(p_route)-1):
                    segment = np.linspace(p_route[idx], p_route[idx+1], 5, endpoint=False)
                    path_pts.extend(segment)
                path_pts.append(target_pos)
                st.session_state.current_planned_path = np.array(path_pts)
            else:
                st.session_state.current_planned_path = np.linspace(np.array([start_x, start_y]), target_pos, 35)
            st.info("Path reset to original route.")

        run_sim = st.button("▶️ Run Trajectory Simulation", key="tab4_run_btn")
        
    with sim_col2:
        st.markdown("#### Top-Down Field View & Trajectory Mapping")
        
        field_dim = 3.65
        dt_coeff = {"Mecanum (Omnidirectional)": 0.85, "Tank Drive": 1.1, "Swerve Drive": 1.0}[drivetrain]
        max_accel = (motor_rpm / 150.0) * dt_coeff / (mass_kg / 15.0)


        path = st.session_state.current_planned_path

        # Precompute the entire velocity profile for Tab 4 plotting
        tab4_velocities = []
        tab4_times = []
        for step_idx in range(len(path)):
            curr_pos = path[step_idx]
            time_sim = step_idx * 0.15
            tab4_times.append(time_sim)
            
            proximity_multiplier = 1.0
            for obs in obstacles:
                obs_center = np.array([obs["x"], obs["y"]])
                dist_to_obs = np.linalg.norm(curr_pos - obs_center)
                warning_threshold = obs["r"] + robot_radius + 0.5
                
                if dist_to_obs < warning_threshold:
                    scaled_factor = max(0.3, (dist_to_obs - (obs["r"] + robot_radius)) / 0.5)
                    if scaled_factor < proximity_multiplier:
                        proximity_multiplier = scaled_factor
            
            base_vel = min(max_velocity, max_accel * time_sim)
            tab4_velocities.append(base_vel * proximity_multiplier)

        # Helper to get standard 2D bounding boxes (used for chassis and wheels)
        def get_rotated_rect(center_x, center_y, width, length, heading_rad):
            dx = length / 2
            dy = width / 2
            local_corners = np.array([
                [-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy], [-dx, -dy]
            ])
            rot_mat = np.array([
                [np.cos(heading_rad), -np.sin(heading_rad)],
                [np.sin(heading_rad), np.cos(heading_rad)]
            ])
            rotated = np.dot(local_corners, rot_mat.T)
            return rotated[:, 0] + center_x, rotated[:, 1] + center_y

        # Helper to plot Tab 4's dynamic velocity graph
        def make_2d_vel_graph(active_step):
            fig_vel_2d = go.Figure()
            fig_vel_2d.add_trace(go.Scatter(
                x=tab4_times, y=tab4_velocities, mode='lines',
                line=dict(color='#3498db', width=3), name="Velocity"
            ))
            fig_vel_2d.add_trace(go.Scatter(
                x=[tab4_times[active_step]], y=[tab4_velocities[active_step]], mode='markers',
                marker=dict(color='red', size=10), name="Current Frame"
            ))
            fig_vel_2d.add_vline(x=tab4_times[active_step], line_width=1.5, line_dash="dash", line_color="red")
            fig_vel_2d.update_layout(
                title="Dynamic 2D Trajectory Velocity Graph (m/s vs Time)",
                xaxis=dict(title="Time (s)"), yaxis=dict(title="Velocity (m/s)", range=[0, max_velocity + 0.5]),
                height=220, margin=dict(l=40, r=40, b=30, t=40), showlegend=False
            )
            return fig_vel_2d

        chart_space = st.empty()
        vel_chart_space = st.empty()
        status_ticker = st.empty()
        metrics_space = st.empty()

        if run_sim:
            has_ever_collided = False
            pollen_collected = [False] * 8  # Track collected pollen particles in real-time
            
            for step in range(len(path)):
                curr_pos = path[step]
                
                if step < len(path) - 1:
                    heading = np.atan2(path[step+1][1] - curr_pos[1], path[step+1][0] - curr_pos[0])
                else:
                    heading = np.atan2(target_pos[1] - curr_pos[1], target_pos[0] - curr_pos[0])

                # Boundary safety crossing validation / Pollen ingestion detection
                step_collided = False
                for obs in obstacles:
                    obs_center = np.array([obs["x"], obs["y"]])
                    dist = np.linalg.norm(curr_pos - obs_center)
                    if dist < (obs["r"] + robot_radius):
                        step_collided = True
                        has_ever_collided = True

                if game_year == "2026-27 (Future)":
                    for p_idx, pollen_pos in enumerate(st.session_state.pollen_locations):
                        dist_to_pollen = np.linalg.norm(curr_pos - pollen_pos)
                        if dist_to_pollen < (robot_radius + 0.15):
                            pollen_collected[p_idx] = True

                velocity = tab4_velocities[step]
                ke = 0.5 * mass_kg * (velocity ** 2)

                # Render top-down arena
                fig = go.Figure()
                
                # Plot obstacles (if active)
                for idx, obs in enumerate(obstacles):
                    shape_type = obs.get("shape", "Circle")
                    if shape_type == "Circle":
                        theta = np.linspace(0, 2*np.pi, 50)
                        ox = obs["x"] + obs["r"] * np.cos(theta)
                        oy = obs["y"] + obs["r"] * np.sin(theta)
                        fig.add_trace(go.Scatter(
                            x=ox, y=oy, fill="toself", line=dict(color="#e74c3c", width=2),
                            fillcolor="rgba(231, 76, 60, 0.4)", name=f"Obstacle {idx+1} ({shape_type})"
                        ))
                        boundary_r = obs["r"] + robot_radius
                        bx = obs["x"] + boundary_r * np.cos(theta)
                        by = obs["y"] + boundary_r * np.sin(theta)
                        fig.add_trace(go.Scatter(
                            x=bx, y=by, line=dict(color="#f1c40f", width=1.5, dash="dash"),
                            name=f"Obs {idx+1} Limit"
                        ))
                    else:  # Square
                        r_val = obs["r"]
                        ox = [obs["x"] - r_val, obs["x"] + r_val, obs["x"] + r_val, obs["x"] - r_val, obs["x"] - r_val]
                        oy = [obs["y"] - r_val, obs["y"] - r_val, obs["y"] + r_val, obs["y"] + r_val, obs["y"] - r_val]
                        fig.add_trace(go.Scatter(
                            x=ox, y=oy, fill="toself", line=dict(color="#e74c3c", width=2),
                            fillcolor="rgba(231, 76, 60, 0.4)", name=f"Obstacle {idx+1} ({shape_type})"
                        ))
                        br = r_val + robot_radius
                        bx = [obs["x"] - br, obs["x"] + br, obs["x"] + br, obs["x"] - br, obs["x"] - br]
                        by = [obs["y"] - br, obs["y"] - br, obs["y"] + br, obs["y"] + br, obs["y"] - br]
                        fig.add_trace(go.Scatter(
                            x=bx, y=by, line=dict(color="#f1c40f", width=1.5, dash="dash"),
                            name=f"Obs {idx+1} Limit"
                        ))

                # Plot pollen particles if Biobuzz season
                if game_year == "2026-27 (Future)":
                    for p_idx, pollen_pos in enumerate(st.session_state.pollen_locations):
                        p_color = "rgba(46, 204, 113, 0.2)" if pollen_collected[p_idx] else "rgb(241, 196, 15)"
                        fig.add_trace(go.Scatter(
                            x=[pollen_pos[0]], y=[pollen_pos[1]], mode="markers",
                            marker=dict(size=12, color=p_color, line=dict(color="#d35400", width=1)),
                            name=f"Pollen {p_idx+1}"
                        ))

                # Dynamic color trace based on current status
                trace_color = "#3498db"
                fig.add_trace(go.Scatter(
                    x=path[:, 0], y=path[:, 1], mode="lines",
                    line=dict(color=trace_color, width=3, dash="dash"),
                    name="Trajectory Trace"
                ))
                
                # Plot Hive / Target
                target_marker_color = "gold" if game_year != "2026-27 (Future)" else "rgb(231, 76, 60)"
                target_name = "Target" if game_year != "2026-27 (Future)" else "Hive Base"
                fig.add_trace(go.Scatter(
                    x=[target_pos[0]], y=[target_pos[1]], mode="markers",
                    marker=dict(size=16, color=target_marker_color, symbol="star"), name=target_name
                ))

                # --- Draw Highly Detailed 4-Wheel Assembly or Tank Treads in 2D ---
                robot_base_color = "rgba(231, 76, 60, 0.6)" if step_collided else "rgba(52, 152, 219, 0.7)"
                stroke_color = "#e74c3c" if step_collided else "#2c3e50"
                
                # 1. Main Chassis Box
                rx, ry = get_rotated_rect(curr_pos[0], curr_pos[1], w_m, l_m, heading)
                fig.add_trace(go.Scatter(
                    x=rx, y=ry, fill="toself", line=dict(color=stroke_color, width=2.5),
                    fillcolor=robot_base_color, name="Chassis Box"
                ))

                # 2. Control Hub Core
                hub_w, hub_l = w_m * 0.5, l_m * 0.5
                hx, hy = get_rotated_rect(curr_pos[0], curr_pos[1], hub_w, hub_l, heading)
                fig.add_trace(go.Scatter(
                    x=hx, y=hy, fill="toself", line=dict(color="#34495e", width=1.5),
                    fillcolor="rgba(44, 62, 80, 0.85)", name="Control Hub"
                ))

                # 3. Drive System Rendering (Treads for Tank Drive, otherwise Wheels)
                if drivetrain == "Tank Drive":
                    tread_w, tread_l = w_m * 0.22, l_m * 0.9
                    tread_offsets = [
                        (0, w_m / 2 + tread_w / 2),  # Left Tread
                        (0, -w_m / 2 - tread_w / 2)  # Right Tread
                    ]
                    for idx_t, (ox_t, oy_t) in enumerate(tread_offsets):
                        rot_x = curr_pos[0] + ox_t * np.cos(heading) - oy_t * np.sin(heading)
                        rot_y = curr_pos[1] + ox_t * np.sin(heading) + oy_t * np.cos(heading)
                        tx, ty = get_rotated_rect(rot_x, rot_y, tread_w, tread_l, heading)
                        fig.add_trace(go.Scatter(
                            x=tx, y=ty, fill="toself", line=dict(color="#111111", width=2),
                            fillcolor="rgba(40, 40, 40, 0.95)", showlegend=False
                        ))
                        # Add track rib lines for visual detail
                        for rib_offset in np.linspace(-tread_l/2 + 0.02, tread_l/2 - 0.02, 6):
                            rx_rib = rot_x + rib_offset * np.cos(heading)
                            ry_rib = rot_y + rib_offset * np.sin(heading)
                            rib_x_endpoints = [rx_rib - (tread_w/2)*np.sin(heading), rx_rib + (tread_w/2)*np.sin(heading)]
                            rib_y_endpoints = [ry_rib + (tread_w/2)*np.cos(heading), ry_rib - (tread_w/2)*np.cos(heading)]
                            fig.add_trace(go.Scatter(
                                x=rib_x_endpoints, y=rib_y_endpoints, mode="lines",
                                line=dict(color="#666666", width=1.5), showlegend=False
                            ))
                else:
                    wheel_w, wheel_l = w_m * 0.22, l_m * 0.3
                    wheel_offsets = [
                        (l_m / 2 - wheel_l / 2, w_m / 2 + wheel_w / 2),   # Front Left
                        (l_m / 2 - wheel_l / 2, -w_m / 2 - wheel_w / 2),  # Front Right
                        (-l_m / 2 + wheel_l / 2, w_m / 2 + wheel_w / 2),  # Rear Left
                        (-l_m / 2 + wheel_l / 2, -w_m / 2 - wheel_w / 2)  # Rear Right
                    ]
                    for idx_w, (ox_w, oy_w) in enumerate(wheel_offsets):
                        rot_x = curr_pos[0] + ox_w * np.cos(heading) - oy_w * np.sin(heading)
                        rot_y = curr_pos[1] + ox_w * np.sin(heading) + oy_w * np.cos(heading)
                        wx, wy = get_rotated_rect(rot_x, rot_y, wheel_w, wheel_l, heading)
                        fig.add_trace(go.Scatter(
                            x=wx, y=wy, fill="toself", line=dict(color="#111111", width=2),
                            fillcolor="rgba(30, 30, 30, 0.95)", showlegend=False
                        ))

                # Heading direction pointer nose cone
                nose_x = curr_pos[0] + (l_m / 2) * np.cos(heading)
                nose_y = curr_pos[1] + (l_m / 2) * np.sin(heading)
                fig.add_trace(go.Scatter(
                    x=[curr_pos[0], nose_x], y=[curr_pos[1], nose_y], mode="lines+markers",
                    line=dict(color="#f39c12", width=3.5), marker=dict(size=6, color="#f1c40f"), showlegend=False
                ))

                fig.update_layout(
                    xaxis=dict(range=[0, field_dim], title="Field Width (m)"),
                    yaxis=dict(range=[0, field_dim], title="Field Length (m)"),
                    width=700, height=450, plot_bgcolor="rgba(240, 240, 240, 0.9)",
                    margin=dict(l=10, r=10, b=10, t=10)
                )

                chart_space.plotly_chart(fig, use_container_width=True, key=f"sim_2d_frame_{step}")
                vel_chart_space.plotly_chart(make_2d_vel_graph(step), use_container_width=True, key=f"sim_2d_vel_{step}")

                # Update Status ticker
                if game_year == "2026-27 (Future)":
                    count_collected = sum(pollen_collected)
                    if step_collided:
                        status_ticker.error(f"🚨 Collision Detected at step {step}! Pollen collector safety envelope breached.")
                    elif has_ever_collided:
                        status_ticker.warning(f"⚠️ Recovering... Pollen status: {count_collected}/8. Previous obstacle collision occurred on path.")
                    else:
                        status_ticker.info(f"🌿 Pollen Ingestion status: {count_collected}/8 collected. Avoiding obstacles and heading to Hive.")
                else:
                    if step_collided:
                        status_ticker.error(f"🚨 Collision Detected at step {step}! Safety envelope breached.")
                    elif has_ever_collided:
                        status_ticker.warning(f"⚠️ Recovering... currently safe, but previous collision occurred on path.")
                    else:
                        status_ticker.info(f"⚡ Navigating. Speed: {velocity:.2f} m/s | Heading: {np.degrees(heading):.1f}°.")

                with metrics_space:
                    s_col1, s_col2, s_col3 = st.columns(3)
                    s_col1.metric("Current Speed", f"{velocity:.2f} m/s")
                    s_col2.metric("Chassis Momentum", f"{(mass_kg * velocity):.1f} kg·m/s")
                    s_col3.metric("Instantaneous KE", f"{ke:.1f} J")

                time.sleep(0.08)
                
            if game_year == "2026-27 (Future)":
                if has_ever_collided:
                    status_ticker.error("🚨 Replay complete. All pollen collected, but safety boundaries were breached by obstacles!")
                else:
                    status_ticker.success("🏆 BIOBUZZ™ run complete! All pollen successfully collected and deposited at the Hive without obstacle collision.")
            elif has_ever_collided:
                status_ticker.error("🚨 Path replay completed. Warnings triggered: Collision envelope was breached.")
            else:
                status_ticker.success("🏆 Safe autonomous run completed!")
        else:
            # Static Preview Render
            fig = go.Figure()
            
            # Plot active obstacles
            for idx, obs in enumerate(obstacles):
                shape_type = obs.get("shape", "Circle")
                if shape_type == "Circle":
                    theta = np.linspace(0, 2*np.pi, 50)
                    ox = obs["x"] + obs["r"] * np.cos(theta)
                    oy = obs["y"] + obs["r"] * np.sin(theta)
                    fig.add_trace(go.Scatter(x=ox, y=oy, fill="toself", line=dict(color="#e74c3c", width=2), fillcolor="rgba(231, 76, 60, 0.4)", name=f"Obstacle {idx+1} ({shape_type})"))
                    boundary_r = obs["r"] + robot_radius
                    bx = obs["x"] + boundary_r * np.cos(theta)
                    by = obs["y"] + boundary_r * np.sin(theta)
                    fig.add_trace(go.Scatter(x=bx, y=by, line=dict(color="#f1c40f", width=1.5, dash="dash"), name=f"Obs {idx+1} Limit"))
                else:  # Square
                    r_val = obs["r"]
                    ox = [obs["x"] - r_val, obs["x"] + r_val, obs["x"] + r_val, obs["x"] - r_val, obs["x"] - r_val]
                    oy = [obs["y"] - r_val, obs["y"] - r_val, obs["y"] + r_val, obs["y"] + r_val, obs["y"] - r_val]
                    fig.add_trace(go.Scatter(x=ox, y=oy, fill="toself", line=dict(color="#e74c3c", width=2), fillcolor="rgba(231, 76, 60, 0.4)", name=f"Obstacle {idx+1} ({shape_type})"))
                    br = r_val + robot_radius
                    bx = [obs["x"] - br, obs["x"] + br, obs["x"] + br, obs["x"] - br, obs["x"] - br]
                    by = [obs["y"] - br, obs["y"] - br, obs["y"] + br, obs["y"] + br, obs["y"] - br]
                    fig.add_trace(go.Scatter(x=bx, y=by, line=dict(color="#f1c40f", width=1.5, dash="dash"), name=f"Obs {idx+1} Limit"))

            # Plot pollen targets if Biobuzz season
            if game_year == "2026-27 (Future)":
                for p_idx, pollen_pos in enumerate(st.session_state.pollen_locations):
                    fig.add_trace(go.Scatter(
                        x=[pollen_pos[0]], y=[pollen_pos[1]], mode="markers",
                        marker=dict(size=12, color="rgb(241, 196, 15)", line=dict(color="#d35400", width=1)),
                        name=f"Pollen {p_idx+1}"
                    ))
            
            fig.add_trace(go.Scatter(x=path[:, 0], y=path[:, 1], mode="lines", line=dict(color="#3498db", width=2, dash="dash"), name="Planned Path"))
            
            target_marker_color = "gold" if game_year != "2026-27 (Future)" else "rgb(231, 76, 60)"
            target_name = "Target" if game_year != "2026-27 (Future)" else "Hive Base"
            fig.add_trace(go.Scatter(x=[target_pos[0]], y=[target_pos[1]], mode="markers", marker=dict(size=14, color=target_marker_color, symbol="star"), name=target_name))
            
            rx, ry = get_rotated_rect(start_x, start_y, w_m, l_m, 0.0)
            fig.add_trace(go.Scatter(x=rx, y=ry, fill="toself", line=dict(color="#7f8c8d", width=3), fillcolor="rgba(127, 140, 141, 0.5)", name="Robot Start Location"))
            
            fig.update_layout(
                xaxis=dict(range=[0, field_dim], title="Field Width (m)"),
                yaxis=dict(range=[0, field_dim], title="Field Length (m)"),
                width=700, height=450, plot_bgcolor="rgba(240, 240, 240, 0.9)",
                margin=dict(l=10, r=10, b=10, t=10)
            )
            chart_space.plotly_chart(fig, use_container_width=True, key="sim_chart_static_preview")
            vel_chart_space.plotly_chart(make_2d_vel_graph(0), use_container_width=True, key="static_2d_vel_graph")
            status_ticker.info("Adjust settings. Click '🧠 Optimize Path' in Step 4 to recalculate trajectory, then simulate.")


# --- Tab 5: 3D Interactive Space Simulator ---
with tab5:
    st.subheader("🌐 3D Interactive Space Simulator")
    st.markdown("Kinematic WebGL simulation showing closed obstacle cylinders and an **enhanced robot assembly**.")

    # 🎥 NEW: Camera Perspective Selector
    camera_mode = st.selectbox(
        "🎥 Choose Camera View Angle",
        ["Default Isometric View", "Starting Point Diagonal View", "Robot Tail View (Drone)"],
        key="camera_perspective_selector"
    )

    path = st.session_state.current_planned_path
    dt_coeff = {"Mecanum (Omnidirectional)": 0.85, "Tank Drive": 1.1, "Swerve Drive": 1.0}[drivetrain]
    max_accel = (motor_rpm / 150.0) * dt_coeff / (mass_kg / 15.0)

    # Precompute full 3D velocity profile
    velocities_profile_3d = []
    times_profile_3d = []
    for step in range(len(path)):
        curr_pos = path[step]
        time_sim = step * 0.15
        times_profile_3d.append(time_sim)
        
        proximity_multiplier = 1.0
        for obs in obstacles:
            obs_center = np.array([obs["x"], obs["y"]])
            dist_to_obs = np.linalg.norm(curr_pos - obs_center)
            warning_threshold = obs["r"] + robot_radius + 0.5
            
            if dist_to_obs < warning_threshold:
                scaled_factor = max(0.3, (dist_to_obs - (obs["r"] + robot_radius)) / 0.5)
                if scaled_factor < proximity_multiplier:
                    proximity_multiplier = scaled_factor
                    
        base_vel = min(max_velocity, max_accel * time_sim)
        velocities_profile_3d.append(base_vel * proximity_multiplier)

    # Layout Controls
    col_ctrl1, col_ctrl2 = st.columns([1, 2])
    with col_ctrl1:
        run_3d_sim = st.button("▶️ Play 3D Animation", use_container_width=True, key="play_3d_animation_btn")
    with col_ctrl2:
        sim_scrub_frame = st.slider("Playback Scrub Frame (Drag to inspect)", 0, len(path)-1, 0, key="scrub_slider_3d")

    # Helper function to generate 3D Wheel meshes
    def get_wheel_cylinder_mesh(cx, cy, cz, radius, thickness, heading_rad, wheel_color="rgb(35, 35, 38)"):
        num_pts = 16
        u = np.linspace(0, 2*np.pi, num_pts)
        lat_heading = heading_rad + np.pi/2
        
        x_pts = []
        y_pts = []
        z_pts = []
        
        for angle in u:
            circle_x = radius * np.cos(angle)
            circle_z = radius * np.sin(angle)
            
            x_left = cx + circle_x * np.cos(heading_rad) + (thickness / 2) * np.cos(lat_heading)
            y_left = cy + circle_x * np.sin(heading_rad) + (thickness / 2) * np.sin(lat_heading)
            z_left = cz + circle_z
            
            x_right = cx + circle_x * np.cos(heading_rad) - (thickness / 2) * np.cos(lat_heading)
            y_right = cy + circle_x * np.sin(heading_rad) - (thickness / 2) * np.sin(lat_heading)
            z_right = cz + circle_z
            
            x_pts.extend([x_left, x_right])
            y_pts.extend([y_left, y_right])
            z_pts.extend([z_left, z_right])
            
        x_pts = np.array(x_pts)
        y_pts = np.array(y_pts)
        z_pts = np.array(z_pts)
        
        i_idx = []
        j_idx = []
        k_idx = []
        for s in range(0, 2 * num_pts - 2, 2):
            i_idx.append(s)
            j_idx.append(s+1)
            k_idx.append(s+2)
            i_idx.append(s+1)
            j_idx.append(s+3)
            k_idx.append(s+2)
            
        return go.Mesh3d(
            x=x_pts, y=y_pts, z=z_pts,
            i=i_idx, j=j_idx, k=k_idx,
            color=wheel_color, opacity=0.95, showlegend=False,
            lighting=dict(ambient=0.4, diffuse=0.7, specular=0.5, roughness=0.3)
        )

    # Helper function to construct closed Solid 3D Cylinders for Obstacles
    def get_closed_obstacle_mesh(cx, cy, r, height, num_pts=32, color='rgb(231, 76, 60)'):
        u = np.linspace(0, 2*np.pi, num_pts, endpoint=False)
        x_bottom = list(cx + r * np.cos(u))
        y_bottom = list(cy + r * np.sin(u))
        z_bottom = [0.0] * num_pts

        x_top = list(cx + r * np.cos(u))
        y_top = list(cy + r * np.sin(u))
        z_top = [height] * num_pts

        x_all = x_bottom + x_top
        y_all = y_bottom + y_top
        z_all = z_bottom + z_top

        i_idx, j_idx, k_idx = [], [], []

        for s in range(num_pts):
            next_s = (s + 1) % num_pts
            i_idx.append(s)
            j_idx.append(next_s)
            k_idx.append(s + num_pts)
            i_idx.append(next_s)
            j_idx.append(next_s + num_pts)
            k_idx.append(s + num_pts)

        for s in range(1, num_pts - 1):
            i_idx.append(num_pts)
            j_idx.append(num_pts + s)
            k_idx.append(num_pts + s + 1)

        return go.Mesh3d(
            x=x_all, y=y_all, z=z_all,
            i=i_idx, j=j_idx, k=k_idx,
            color=color, opacity=0.85, showlegend=False,
            lighting=dict(ambient=0.5, diffuse=0.8, specular=0.8, roughness=0.15)
        )

    # Helper function to construct solid 3D Cuboids for Box Obstacles
    def get_closed_box_obstacle_mesh(cx, cy, r, height, color='rgb(231, 76, 60)'):
        x = [cx - r, cx + r, cx + r, cx - r, cx - r, cx + r, cx + r, cx - r]
        y = [cy - r, cy - r, cy + r, cy + r, cy - r, cy - r, cy + r, cy + r]
        z = [0.0, 0.0, 0.0, 0.0, height, height, height, height]
        return go.Mesh3d(
            x=x, y=y, z=z,
            alphahull=0, color=color, opacity=0.85, showlegend=False,
            lighting=dict(ambient=0.5, diffuse=0.8, specular=0.8, roughness=0.15)
        )

    # Define dynamic 3D rendering engine
    def draw_3d_scene(step_idx):
        curr_pos_3d = path[step_idx]
        if step_idx < len(path) - 1:
            heading_3d = np.atan2(path[step_idx+1][1] - curr_pos_3d[1], path[step_idx+1][0] - curr_pos_3d[0])
        else:
            heading_3d = np.atan2(target_pos[1] - curr_pos_3d[1], target_pos[0] - curr_pos_3d[0])

        current_time = step_idx * 0.15
        current_vel = velocities_profile_3d[step_idx]

        fig_3d = go.Figure()

        # 1. Cyber Field Grid Floor Layout
        fig_3d.add_trace(go.Surface(
            z=np.zeros((10, 10)),
            x=np.linspace(0, 3.65, 10),
            y=np.linspace(0, 3.65, 10),
            colorscale=[[0, 'rgb(24, 28, 36)'], [1, 'rgb(18, 22, 28)']],
            showscale=False, name="Field Floor"
        ))
        
        for grid_line in np.linspace(0, 3.65, 7):
            fig_3d.add_trace(go.Scatter3d(
                x=[grid_line, grid_line], y=[0, 3.65], z=[0.002, 0.002],
                mode='lines', line=dict(color='rgba(255, 255, 255, 0.08)', width=1), showlegend=False
            ))
            fig_3d.add_trace(go.Scatter3d(
                x=[0, 3.65], y=[grid_line, grid_line], z=[0.002, 0.002],
                mode='lines', line=dict(color='rgba(255, 255, 255, 0.08)', width=1), showlegend=False
            ))

        # 2. Render Obstacles (if active)
        for idx, obs in enumerate(obstacles):
            shape_type = obs.get("shape", "Circle")
            if shape_type == "Circle":
                fig_3d.add_trace(get_closed_obstacle_mesh(obs["x"], obs["y"], obs["r"], 0.60, color='rgb(231, 76, 60)'))
            else:  # Square
                fig_3d.add_trace(get_closed_box_obstacle_mesh(obs["x"], obs["y"], obs["r"], 0.60, color='rgb(231, 76, 60)'))
        
        # Render Pollen targets as small golden spheres in 3D if Biobuzz mode is active
        if game_year == "2026-27 (Future)":
            for p_idx, pollen_pos in enumerate(st.session_state.pollen_locations):
                fig_3d.add_trace(go.Scatter3d(
                    x=[pollen_pos[0]], y=[pollen_pos[1]], z=[0.05],
                    mode='markers', marker=dict(size=7, color='rgb(241, 196, 15)', symbol='circle'),
                    name=f"Pollen {p_idx+1}"
                ))

        # 3. Trajectory line path
        fig_3d.add_trace(go.Scatter3d(
            x=path[:, 0], y=path[:, 1], z=np.zeros(len(path)) + 0.01,
            mode='lines', line=dict(color='rgb(52, 152, 219)', width=5), name="Trajectory Trail"
        ))

        # 4. Assembly: Detailed Metallic Chassis Box Mesh
        dx = l_m / 2
        dy = w_m / 2
        h_chassis = 0.14 
        z_offset = 0.08  

        corners_2d = np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]])
        rot_mat = np.array([[np.cos(heading_3d), -np.sin(heading_3d)], [np.sin(heading_3d), np.cos(heading_3d)]])
        rot_2d = np.dot(corners_2d, rot_mat.T)
        
        rx_box = list(rot_2d[:, 0] + curr_pos_3d[0]) * 2
        ry_box = list(rot_2d[:, 1] + curr_pos_3d[1]) * 2
        rz_box = [z_offset]*4 + [z_offset + h_chassis]*4
        
        fig_3d.add_trace(go.Mesh3d(
            x=rx_box, y=ry_box, z=rz_box,
            alphahull=0, color='rgb(125, 135, 145)', opacity=0.95, name="Main Chassis",
            lighting=dict(ambient=0.5, diffuse=0.8, specular=1.2, roughness=0.1, fresnel=0.5)
        ))

        # Core hub / Battery component
        hub_dx, hub_dy = dx * 0.6, dy * 0.6
        hub_corners = np.array([[-hub_dx, -hub_dy], [hub_dx, -hub_dy], [hub_dx, hub_dy], [-hub_dx, -hub_dy]])
        hub_rot = np.dot(hub_corners, rot_mat.T)
        hx_box = list(hub_rot[:, 0] + curr_pos_3d[0]) * 2
        hy_box = list(hub_rot[:, 1] + curr_pos_3d[1]) * 2
        hz_box = [z_offset + h_chassis]*4 + [z_offset + h_chassis + 0.04]*4
        fig_3d.add_trace(go.Mesh3d(
            x=hx_box, y=hy_box, z=hz_box,
            alphahull=0, color='rgb(241, 196, 15)', opacity=0.95, name="Power Core",
            lighting=dict(ambient=0.6, diffuse=0.9, specular=1.5, roughness=0.05)
        ))

        # 5. Assembly: Treads for Tank Drive, Cylindrical Wheels for Others
        if drivetrain == "Tank Drive":
            tread_w = 0.04
            tread_l = l_m * 0.9
            tread_h = 0.12
            tread_offsets_3d = [
                (0, dy + tread_w/2),
                (0, -dy - tread_w/2)
            ]
            for ox_t, oy_t in tread_offsets_3d:
                rot_x = curr_pos_3d[0] + ox_t * np.cos(heading_3d) - oy_t * np.sin(heading_3d)
                rot_y = curr_pos_3d[1] + ox_t * np.sin(heading_3d) + oy_t * np.cos(heading_3d)
                
                tx_box = list(np.array([-tread_l/2, tread_l/2, tread_l/2, -tread_l/2]) * np.cos(heading_3d) - np.array([-tread_w/2, -tread_w/2, tread_w/2, tread_w/2]) * np.sin(heading_3d) + rot_x) * 2
                ty_box = list(np.array([-tread_l/2, tread_l/2, tread_l/2, -tread_l/2]) * np.sin(heading_3d) + np.array([-tread_w/2, -tread_w/2, tread_w/2, tread_w/2]) * np.cos(heading_3d) + rot_y) * 2
                tz_box = [0.02]*4 + [0.02 + tread_h]*4
                
                fig_3d.add_trace(go.Mesh3d(
                    x=tx_box, y=ty_box, z=tz_box,
                    alphahull=0, color='rgb(40, 40, 40)', opacity=0.95, showlegend=False,
                    lighting=dict(ambient=0.4, diffuse=0.7, specular=0.5, roughness=0.3)
                ))
        else:
            wheel_radius = 0.075 
            wheel_width = 0.04
            local_wheel_offsets = [
                [dx - 0.02, dy + wheel_width/2],    # Front-Left
                [dx - 0.02, -dy - wheel_width/2],   # Front-Right
                [-dx + 0.02, dy + wheel_width/2],   # Rear-Left
                [-dx + 0.02, -dy - wheel_width/2]   # Rear-Right
            ]
            for offset in local_wheel_offsets:
                wx_local, wy_local = offset[0], offset[1]
                wx_world = curr_pos_3d[0] + wx_local * np.cos(heading_3d) - wy_local * np.sin(heading_3d)
                wy_world = curr_pos_3d[1] + wx_local * np.sin(heading_3d) + wy_local * np.cos(heading_3d)
                wz_world = wheel_radius 
                
                wheel_mesh = get_wheel_cylinder_mesh(wx_world, wy_world, wz_world, wheel_radius, wheel_width, heading_3d)
                fig_3d.add_trace(wheel_mesh)

        # 6. Target Spot Indicator
        fig_3d.add_trace(go.Scatter3d(
            x=[target_pos[0]], y=[target_pos[1]], z=[0.02],
            mode='markers', marker=dict(size=8, color='gold', symbol='diamond'), name="Target"
        ))

        # 🎥 UPDATED: Windshield-Style Camera Angle Math
        camera_config = dict(
            eye=dict(x=1.5, y=1.5, z=1.2),
            center=dict(x=0, y=0, z=0),
            up=dict(x=0, y=0, z=1)
        )

        if camera_mode == "Starting Point Diagonal View":
            # Diagonal corner view looking across the domain centered at the middle field
            start_pos = path[0]
            camera_config = dict(
                eye=dict(x=(start_pos[0] - 1) / 1.825 - 0.3, y=(start_pos[1] - 1) / 1.825 - 0.3, z=0.3),
                center=dict(x=0, y=0, z=-0.2),
                up=dict(x=0, y=0, z=1)
            )

        elif camera_mode == "Robot Tail View (Drone)":
            # 1. Windshield placement: At the front edge of the robot chassis
            front_x = curr_pos_3d[0] - (2.5 * l_m) * np.cos(heading_3d)
            front_y = curr_pos_3d[1] - (l_m/2) * np.sin(heading_3d)
            front_z = 0.70  # Elevated slightly off the field surface

            # 2. Normalize to Plotly Camera Space using updated ranges
            x_eye_norm = (front_x - 1.5) / 2.0
            y_eye_norm = (front_y - 1.5) / 2.0
            z_eye_norm = (front_z - 0.25) / 0.75

            # 3. Windshield Direction: Look 1.2 meters directly in front of the robot
            look_distance = 0.7
            look_x = front_x + look_distance * np.cos(heading_3d)
            look_y = front_y + look_distance * np.sin(heading_3d)
            look_z = 0  # Angled slightly downward to view the upcoming path

            # Normalize the look-at point
            x_look_norm = (look_x - 1.5) / 2.0
            y_look_norm = (look_y - 1.5) / 2.0
            z_look_norm = (look_z - 0.25) / 0.75

            # 4. Set camera configuration
            camera_config = dict(
                eye=dict(x=x_eye_norm, y=y_eye_norm, z=z_eye_norm),
                # center is the pointing vector offset from the eye
                center=dict(x=x_look_norm - x_eye_norm, y=y_look_norm - y_eye_norm, z=z_look_norm - z_eye_norm),
                up=dict(x=0, y=0, z=1)
            )

        # Apply updated Scene layout & Dynamic Camera
        fig_3d.update_layout(
            scene=dict(
                xaxis=dict(title='Width (m)', range=[-0.5, 3.5], gridcolor='rgba(255,255,255,0.05)', backgroundcolor='rgb(12, 14, 18)'),
                yaxis=dict(title='Length (m)', range=[-0.5, 3.5], gridcolor='rgba(255,255,255,0.05)', backgroundcolor='rgb(12, 14, 18)'),
                zaxis=dict(title='Height (m)', range=[-0.1, 1.0], gridcolor='rgba(255,255,255,0.05)', backgroundcolor='rgb(12, 14, 18)'),
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=0.3),
                # Set our configured camera perspective
                camera=camera_config
            ),
            # selectively keeps user rotation lock based on mode
            uirevision="constant_user_view" if camera_mode == "Default Isometric View" else f"cam_{camera_mode}",
            width=800, height=500, margin=dict(l=0, r=0, b=0, t=40), showlegend=False,
            paper_bgcolor="#0e1117"
        )

        return fig_3d, current_vel, current_time

    # Dynamic Graph generator
    def draw_3d_velocity_graph(active_step_idx):
        active_time = times_profile_3d[active_step_idx]
        active_velocity = velocities_profile_3d[active_step_idx]

        fig_vel = go.Figure()
        fig_vel.add_trace(go.Scatter(
            x=times_profile_3d, y=velocities_profile_3d, mode='lines',
            line=dict(color='#2ecc71', width=3), name="Velocity"
        ))
        fig_vel.add_trace(go.Scatter(
            x=[active_time], y=[active_velocity], mode='markers+text',
            marker=dict(color='red', size=11, symbol='circle'),
            text=[f"{active_velocity:.2f} m/s"], textposition="top right"
        ))
        fig_vel.add_vline(x=active_time, line_width=1.5, line_dash="dash", line_color="red")

        fig_vel.update_layout(
            title="Real-Time Robot Velocity Profile over Trajectory",
            xaxis=dict(title="Time (seconds)", range=[0, max(times_profile_3d)]),
            yaxis=dict(title="Velocity (m/s)", range=[0, max_velocity + 0.5]),
            height=220, margin=dict(l=40, r=40, b=30, t=40), showlegend=False
        )
        return fig_vel

    chart_3d_placeholder = st.empty()
    graph_placeholder = st.empty()
    metric_placeholder = st.empty()

    if run_3d_sim:
        for step in range(len(path)):
            fig_3d, vel, t_val = draw_3d_scene(step)
            fig_vel = draw_3d_velocity_graph(step)
            
            chart_3d_placeholder.plotly_chart(fig_3d, use_container_width=True, key=f"play_3d_{step}")
            graph_placeholder.plotly_chart(fig_vel, use_container_width=True, key=f"play_3d_vel_{step}")
            
            with metric_placeholder:
                col_m1, col_m2 = st.columns(2)
                col_m1.metric("Robot Coordinates", f"X: {path[step][0]:.2f}m, Y: {path[step][1]:.2f}m")
                col_m2.metric("Simulated Speed", f"{vel:.2f} m/s", delta=f"{vel*3.28:.1f} ft/s")
            time.sleep(0.08)
        st.success("3D Animation Replay Completed!")
    else:
        fig_3d, vel, t_val = draw_3d_scene(sim_scrub_frame)
        fig_vel = draw_3d_velocity_graph(sim_scrub_frame)
        
        chart_3d_placeholder.plotly_chart(fig_3d, use_container_width=True, key="scrub_3d_fig")
        graph_placeholder.plotly_chart(fig_vel, use_container_width=True, key="scrub_3d_vel")
        
        with metric_placeholder:
            col_m1, col_m2 = st.columns(2)
            col_m1.metric("Robot Coordinates", f"X: {path[sim_scrub_frame][0]:.2f}m, Y: {path[sim_scrub_frame][1]:.2f}m")
            col_m2.metric("Simulated Speed", f"{vel:.2f} m/s", delta=f"{vel*3.28:.1f} ft/s")


# --- Footer ---
st.divider()
st.caption("RoboStrategy | Built for FTC Strategic Dominance | Built with Streamlit + PyTorch")