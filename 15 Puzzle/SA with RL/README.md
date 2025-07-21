
---

# 🧠 Advanced 15-Puzzle Solver with Reinforcement Learning

This project implements a powerful and adaptive solver for the classic **15-Puzzle** using a combination of:

* **Reinforcement Learning** (with weighted heuristic learning),
* **Simulated Annealing** search,
* **Curriculum Learning** to adapt difficulty over time.

The goal is to **learn better heuristics** (than classic Manhattan distance) to solve more difficult puzzle configurations efficiently.

---

## 🚀 Features

* ✅ Simulated annealing-based solving
* ✅ Reinforcement learning of weighted heuristics
* ✅ Curriculum manager to adapt difficulty dynamically
* ✅ Linear conflict enhanced baseline heuristic
* ✅ Full tracking of training statistics (success rates, steps, losses)
* ✅ Evaluation framework comparing learned vs standard heuristics
* ✅ Visual training plots (via `matplotlib`)

---

## 📦 Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Mo3een21/15puzzle.git
   cd rl-15puzzle-solver
   ```

2. Install required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

   Or install manually:

   ```bash
   pip install numpy matplotlib
   ```

---

## 🧪 Running the Solver

```bash
python main.py
```

This will:

* Train the solver over multiple episodes using curriculum learning
* Learn heuristic weights using gradient descent
* Evaluate and compare performance against traditional heuristic
* Display final learned weights
* Plot progress statistics

---

## 🧠 Architecture Overview

```
.
├── Puzzle15Env            # Core 15-puzzle environment with linear conflict + learned heuristic support
├── ImprovedWeightLearner  # Learns heuristic weights from episodes via least-squares regression
├── AdaptiveSAAgent        # Simulated annealing agent with adaptive exploration
├── SmartCurriculumManager # Automatically adapts difficulty based on past performance
├── train_with_adaptive_curriculum() # Main training loop
├── evaluate_comprehensive()         # Testing and performance benchmarking
└── plot_training_progress()        # (Optional) training plots
```

---

## 📊 Example Output

```text
Episode 150:
  Difficulty: 30 shuffle steps
  Success Rate: 0.80
  Avg Steps (successful): 212.5
  Weights: min=0.95, max=2.85, mean=1.34
  Recent loss: 0.0421
```

Plotting shows learning curves over episodes:

* 📈 Success rate over time
* 📉 Average steps (if successful)
* 🧮 Loss convergence
* 🧊 Learned heuristic heatmaps

---

## 🧪 Evaluation

Compares performance of:

* Standard heuristic (Manhattan + linear conflicts)
* Learned heuristic (adaptive weights based on position)

Metrics:

* ✅ Solve rate (%)
* ⏱ Avg steps to solution
* 🔁 Timeouts

---

## 🤝 Contributing

1. Fork the project
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/my-feature`)
5. Create a Pull Request

---

## 📄 License

This project is licensed under the MIT License.

---

## ✨ Acknowledgments

Inspired by curriculum learning in reinforcement learning and heuristic optimization techniques in combinatorial problems.

---


