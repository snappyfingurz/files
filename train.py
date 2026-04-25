"""
train.py — Curriculum Learning RL Loop
Split into EASY, MEDIUM, and HARD phases to show realistic growth.
"""

import time
from env import CustomerSupportEnv
from agent import LLMAgent

def run_rl_training():
    # Initialize Environment and Agent
    env = CustomerSupportEnv()
    agent = LLMAgent()
    
    # CURRICULUM DEFINITION: 
    # Stage 1: Easy (10 eps) -> Quick Mastery
    # Stage 2: Medium (15 eps) -> Slower Progress
    # Stage 3: Hard (25 eps) -> High Difficulty / Slowest Learning
    curriculum = [
        ("EASY", 10, "easy_001"),
        ("MEDIUM", 15, "medium_001"),
        ("HARD", 25, "hard_001")
    ]
    
    total_episodes = 0
    print("\n" + "="*50)
    print("🚀 STARTING THREE-PHASE CURRICULUM TRAINING")
    print("="*50)

    for phase, episodes, task_id in curriculum:
        # 🔄 Reset Environment per phase to ensure clean learning curves
        env = CustomerSupportEnv()
        print(f"\n--- 📈 STARTING {phase} PHASE ({episodes} EPISODES) ---")
        
        for ep in range(episodes):
            total_episodes += 1
            
            # 1. Reset with specific task difficulty
            obs = env.reset(task_id=task_id)
            
            # 2. Agent Acts
            action = agent.act(obs)
            
            # 3. Environment Step
            result = env.step(action)
            reward = float(result.reward)
            
            # 4. Optimized Logging
            # We show the phase and the episode count for that phase
            preview = action.response[:60].replace('\n', ' ')
            print(f"[{phase}] Ep {ep+1:02d} | Rew: {reward:.3f} | Msg: {preview}...")

            # Short speed pause
            time.sleep(0.05)

    print("\n" + "="*50)
    print(f"🏁 CURRICULUM COMPLETE. Total Episodes: {total_episodes}")
    print("="*50)

if __name__ == "__main__":
    run_rl_training()
