# RL and Adversarial Search for Tic Tac Toe and Connect 4

## Overview

This repository contains a Python project that demonstrates how different artificial intelligence algorithms can play two popular games: **Tic Tac Toe** and **Connect 4**.

The code (in the file `rl_advesarialSearch.py`) implements three key AI strategies:
- **Minimax**: An algorithm that simulates all possible moves to choose the best move, assuming both players are perfect.
- **Minimax with Alpha-Beta Pruning**: A more efficient version of Minimax that skips over moves that will not affect the final decision.
- **Q-Learning**: A reinforcement learning method where the AI learns the best moves by trying them out over many games.

In addition, the repository includes a default opponent that:
1. Attempts to win if possible on the next move.
2. Blocks the opponent’s winning move.
3. Chooses a random valid move if neither win nor block is available.

The code also sets up tournaments where these algorithms play:
- Against the default opponent.
- Against each other.

It then prints out the win/loss/draw statistics and displays graphs (bar graphs and a heatmap) to help visualize the performance of each algorithm.

## What to Expect When Running the Code

When you run `rl_advesarialSearch.py` on your local system:
- The program will simulate 200 games per matchup between different agents.
- You will see printed output summarizing:
  - How often each algorithm wins, loses, or draws.
- Graphs will be displayed:
  - Bar graphs for each experiment, with colors indicating wins, losses, and draws.
  - A heatmap summarizing all experiment results.

In simpler terms, the code shows which strategy works better in these games. For example, it reveals that the standard Minimax often wins or draws, while Q-Learning may not perform as well without extensive training.

## Installation and Setup Instructions (Windows & VS Code)

1. **Clone or Download the Repository:**
   - You can download the repository from GitHub and unzip it on your system.
   - Alternatively, use Git to clone the repo:
     ```
     git clone <repository_url>
     ```

2. **Open the Repository in Visual Studio Code:**
   - Launch VS Code and open the folder containing the code.

3. **Create a Virtual Environment:**
   - Open the Terminal in VS Code (Ctrl + `).
   - Run the following command to create a virtual environment:
     ```
     python -m venv venv
     ```

4. **Activate the Virtual Environment:**
   - In the terminal, run:
     ```
     .\venv\Scripts\activate
     ```

5. **Install Dependencies:**
   - Ensure you have an internet connection.
   - Install all required packages by running:
     ```
     pip install numpy matplotlib seaborn
     ```

6. **Run the Code:**
   - In the terminal, execute:
     ```
     python rl_advesarialSearch.py
     ```
   - The program will run the simulations and display the results along with the graphs.

## Concepts Explained

- **Games (Tic Tac Toe & Connect 4):**  
  These are two games with simple rules. Tic Tac Toe is a 3x3 grid game, while Connect 4 is played on a larger board with 6 rows and 7 columns.

- **Minimax and Alpha-Beta Pruning:**  
  Imagine playing a game and thinking about every possible move and its outcome. Minimax does exactly that. It simulates every possibility to pick the best move. Alpha-beta pruning is like being smart about it — it stops looking at moves that wouldn’t change the result, saving time and effort.

- **Q-Learning:**  
  Think of Q-Learning like learning to ride a bike. At first, you might make mistakes, but over time, you learn which moves work best by getting rewards (or penalties) for each move. The computer does something similar: it learns which moves are best by playing many games and updating its “memory.”

- **Tournaments and Experiments:**  
  The code sets up multiple game matches to see how well each strategy performs. It records wins, losses, and draws, then uses simple graphs to make these comparisons easy to understand, even if you aren’t familiar with the underlying algorithms.

## Report Summary

This project was part of an assignment (*RL and Adversarial Search*) by Archit Biswas. It explores how the three algorithms perform in games with different levels of complexity. The report highlights:
- The strong performance of Minimax strategies (both with and without pruning) against a simple opponent.
- The challenges Q-Learning faces in larger games like Connect 4.
- How exploring all moves (even if it takes longer) can sometimes yield better decisions than more “efficient” approaches.

The report provides detailed insights into the trade-offs of each approach.

## Conclusion

This repository is designed to show how artificial intelligence can be applied to game playing. By comparing different strategies and visualizing the results, even someone without a technical background can appreciate the differences in how these algorithms think and make decisions.
