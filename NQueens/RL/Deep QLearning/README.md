

---

# ♟️ N-Queens with Simulated Annealing and Q-Learning 

## 📌 Overview

This project presents an advanced method for solving the **N-Queens problem** by combining **Simulated Annealing (SA)** with a **Q-Learning agent** powered by a **neural network-based Q-function**. Unlike traditional SA or tabular Q-learning, the Q-function here is approximated using a deep neural network, and decisions are made using Boltzmann exploration for probabilistic move acceptance.

---

## 🚀 Key Features

* 🔍 **State Representation**: Board is encoded as a one-hot vector along with normalized conflict features.
* 🧠 **Q-Network**: A fully connected neural network predicts Q-values for (state, action) pairs.
* 🔁 **Simulated Annealing Guided by Q-values**: Replaces handcrafted objective functions with learned Q-value estimations.
* 🧪 **Replay Buffer + Batch Training**: Uses experience replay for stable Q-network training.
* 🎯 **Target Network**: Periodically updated target network stabilizes Q-learning.
* 📈 **Performance Tracking**: Tracks success rate, training steps, and visualizes learning progress.

---

## 🧠 Algorithms

### 🔹 Simulated Annealing (SA)

* Probabilistic local search algorithm.
* Accepts worse moves with a probability based on temperature.
* Temperature decays with each step.

### 🔸 Q-Learning (with Function Approximation)

* Learns a Q-function `Q(s, a)` estimating expected reward.
* Uses a neural network to generalize across unseen states and actions.
* Learns from transitions `(s, a, r, s', done)` stored in a replay buffer.

---



---

## 📦 Dependencies

* Python 3.8+
* NumPy
* PyTorch
* Matplotlib

Install using:

```bash
pip install numpy torch matplotlib
```

---

## ▶️ How to Run

### ✅ Train the Agent

```bash
python nqueens_qlearning.py
```

You can adjust the parameters in `train_agent()` such as board size (`n`), number of episodes, learning rate, etc.

### 🧪 Test the Trained Agent

After training, the model is tested on new N-Queens boards using:

```python
test_trained_agent(agent, n=8, num_tests=100)
```

---

## 📊 Visualization

Training produces two plots:

* **Success Rate vs Episodes**
* **Steps to Solution (moving average)**

These help assess convergence and search efficiency over time.

---

## 🔬 Details

### 🧩 State Vector

Length = `n^2 + n + 2*(2n - 1)`

* One-hot board encoding (`n^2`)
* Normalized column, major and minor diagonal conflict vectors.

### 🧠 Q-Network Architecture

* Input: `[state_vector] + [row/n, col/n]`
* Layers: 3 hidden layers (ReLU + Dropout)
* Output: Scalar Q-value

### 🏃 SA + RL Episode

* At each step:

  * Q-network selects next action via Boltzmann exploration.
  * If move accepted (SA criterion), update state and add to replay buffer.
* Trains Q-network using MSE loss over sampled transitions.

---

## ✅ Sample Output

```
Episode 0: Success rate: 100.00%, Avg steps: 780.0, Epsilon: 0.1
...
Test Results:
Success Rate: 100.00%
Average Steps to Solution: 780
```

---

## 💡 Potential Enhancements

* Replace Q-network with a **Graph Neural Network** for better spatial generalization.
* Add **Double DQN** or **Dueling Network** architecture.
* Apply to generalized **Constraint Satisfaction Problems (CSPs)** beyond N-Queens.

---

## 📌 License

MIT License. Free to use and modify for academic or personal purposes.

---

