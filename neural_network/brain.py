import torch    
import torch.nn as nn
import torch.optim as optim
import os

from collections import deque
import random


class Brain(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(11 , 64),
            nn.ReLU(),

            nn.Linear(64, 128),  
            nn.ReLU(),
            
            nn.Linear(128, 64),  
            nn.ReLU(),

            nn.Linear(64, 3)
        )
        self.memory = deque(maxlen=10_000)

        self.gamma = 0.9
        self.batch_size = 64
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

        self.epsilon = 1.0
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.995
        
        # --- NAYA CODE: Saved brain (.pth file) ko load karne ke liye ---
        model_path = 'yashika_brain.pth'
        if os.path.exists(model_path):
            self.load_state_dict(torch.load(model_path))
            print(f"Expert Brain Loaded: {model_path}")
        else:
            print("No trained brain found, starting fresh.")
        # ----------------------------------------------------------------

    def forward(self, state):
        return self.layers(state)

    def choose_action(self , state , epsilon = 1.0):
        # --- BUG FIX: 'self.epsilon' hata kar sirf 'epsilon' kar diya ---
        if random.random() < epsilon:
            return random.randrange(3)  # Random action

        state_tensor = torch.tensor(state , dtype=torch.float32)
        with torch.no_grad():
            scores = self(state_tensor)

        return int(scores.argmax().item()) 

    def remember(self , state , action , reward , next_state , done):
        experience = (state , action , reward , next_state , done)
        self.memory.append(experience)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory , self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_tensor = torch.tensor(states , dtype=torch.float32)
        actions_tensor = torch.tensor(actions , dtype=torch.int64).unsqueeze(1)
        rewards_tensor = torch.tensor(rewards , dtype=torch.float32).unsqueeze(1)
        next_states_tensor = torch.tensor(next_states , dtype=torch.float32)
        dones_tensor = torch.tensor(dones , dtype=torch.float32).unsqueeze(1)

        current_q_values = self(states_tensor).gather(1 , actions_tensor)
        next_q_values = self(next_states_tensor).max(1)[0].detach().unsqueeze(1)
        target_q_values = rewards_tensor + (self.gamma * next_q_values * (1 - dones_tensor))

        loss = self.loss_fn(current_q_values , target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()