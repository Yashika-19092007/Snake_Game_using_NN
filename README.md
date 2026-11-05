#  AI Snake Agent with Deep Q-Learning (DQN)

An artificial intelligence agent that learns to play the classic Snake game autonomously using **Reinforcement Learning (Deep Q-Network)**. The project features a custom-built neural network architecture and a real-time visualizer to observe the AI's decision-making process.

##  Technical Architecture

### 1. Neural Network Design
The agent uses a Feed-Forward Neural Network built with **PyTorch** to predict the best possible action based on the current game state.
* **Input Layer:** 11 Neurons (representing the game state)
* **Hidden Layers:** 64 -> 128 -> 64 Neurons (with ReLU activation)
* **Output Layer:** 3 Neurons (representing Q-values for actions)

### 2. State Space (11 Inputs)
The AI "sees" the environment through an 11-value array:
* **Danger detection:** Danger Straight, Danger Right, Danger Left
* **Current Direction:** Moving Up, Down, Left, Right
* **Food Location:** Food is Up, Down, Left, Right

### 3. Action Space (3 Outputs)
Based on the Q-values, the agent chooses to move:
* `[1, 0, 0]` -> Keep going straight
* `[0, 1, 0]` -> Turn right
* `[0, 0, 1]` -> Turn left

##  Key Features & Optimizations

* **Stable Inference Mode:** Fixed the testing phase (`play.py`) by setting `epsilon = 0.0` and disabling `brain.learn()`. This prevents the model from taking random actions during evaluation and locks the neural weights to purely test the trained memory.
* **Real-Time NN Visualizer:** Integrated a dynamic Pygame visualizer (`network.py`) that displays node activations and weighted connections (Green for positive, Red for negative) frame-by-frame.
* **Headless Fast-Training:** Optimized `train.py` by decoupling the Pygame clock tick and disabling UI rendering during the training loop, allowing the model to play thousands of games in just a few minutes.

## 🛠️ Tech Stack
* **Language:** Python
* **ML Framework:** PyTorch
* **Graphics/UI:** Pygame-CE

##  Project Structure
* `train.py`: The training loop (Epsilon-greedy strategy, Backpropagation).
* `play.py`: The inference loop (Evaluation mode using the trained `.pth` model).
* `neural_network/brain.py`: The PyTorch DQN architecture and Q-learning logic.
* `vizualizer/network.py`: Real-time node and weight visualization logic.
* `yashika_brain.pth`: The final trained model weights.

## 💻 How to Run Locally

**1. Clone the repository and install dependencies:**
```bash
pip install -r requirements.txt


2. Watch the trained AI play (Inference Mode):
python play.py

3. Train a new model from scratch:
python train.py