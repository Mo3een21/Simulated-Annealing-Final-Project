
---

# ♟️ N-Queens Solver using Reinforcement Learning and Simulated Annealing

## 📌 Overview

This project presents an advanced approach to solving the classic **N-Queens problem** using a combination of **Simulated Annealing (SA)** and **Reinforcement Learning (RL)**. Rather than relying on a static objective function, our solver learns a custom potential (heuristic) function through training episodes. This approach allows for adaptive, intelligent navigation of the solution space, even as the board size increases.

---

## 🚀 Key Features

* **Dynamic Heuristic Learning**: A neural-like potential function is trained to guide the search using reinforcement signals from successful SA trajectories.
* **Simulated Annealing Agent**: Optimizes queen positions using probabilistic move acceptance, enabling escapes from local minima.
* **Feature Engineering**: Potential is computed based on column and diagonal conflicts, with each feature having learnable weights.
* **Online Training Loop**: Gradients are computed from successful episodes, and weights are updated using momentum-based SGD.
* **Visual Training Diagnostics**: Includes plots for success rate, loss, convergence speed, and weight evolution.

---

## 🧠 Algorithms

### Simulated Annealing (SA)

* Explores the solution space probabilistically.
* Moves that reduce conflict (phi) are always accepted.
* Moves that increase phi may be accepted with a probability based on temperature `T`.
* Adaptive cooling and reheating ensure optimal exploration.

### Reinforcement Learning (RL)

* Learns to shape the potential function via self-play.
* Uses trajectories from successful SA episodes to compute loss and gradient.
* Loss function:

  ```
  L = Σ_t (phi(state_t) - c * (T - t))^2
  ```

  where `T` is total trajectory length and `c` is a scaling constant.
* Momentum SGD updates learned weights across features.

---

## 🧪 Applications

This methodology is highly generalizable to other constraint satisfaction and combinatorial optimization problems, such as:

* Sudoku solving
* Traveling salesman variations
* Puzzle-15
* Hungarian cube

---


---

## 🧰 Requirements

* Python 3.8+
* NumPy
* Matplotlib

Install dependencies:

```bash
pip install numpy matplotlib
```

---


### Parameters:

Default training configuration (inside `main()`):

```python
config = {
    'board_size': 8,
    'num_episodes': 2000,
    'max_steps': 10000,
    'learning_rate': 0.001,
    'c': 0.02,
    'momentum': 0.9,
    'weight_decay': 0.0001
}
```

---

## 📊 Output

* Prints stats every `plot_interval` episodes.

* Displays training plots:

  * ✅ Success Rate
  * 📉 Loss
  * ⏱️ Steps per episode
  * 🔁 Weight evolution

* Saves learned weights as `learned_weights_n8.npy`.

---

## 🧪 Testing

After training, the learned weights are tested on `num_tests` (default: 100) random board configurations. The test evaluates:

* ✅ Success rate
* ⏱️ Average steps to solution

You can also visualize a final demonstration run with the trained agent.

---

## 🧩 Why This Is Powerful

Unlike traditional heuristics that require domain knowledge, this RL+SA framework:

* Learns what "good" intermediate states look like from experience.
* Generalizes better across instances.
* Can be extended to dynamic or hybrid problems where rules shift over time.

---

## 📈 Sample Results (for N=8)

```
Training Complete!
Total Time: 115.3 seconds
Final Success Rate: 82.50%
Final Avg Loss: 0.007613
Final Avg Steps: 208.1

Test Success Rate: 84.00%
Average Steps (successful): 201.5
```

---

## 🧩 Credits

Developed as part of a project exploring smart optimization techniques using AI. Combines core ideas from:

* **Reinforcement Learning**
* **Simulated Annealing**
* **Neural Heuristics**

---

