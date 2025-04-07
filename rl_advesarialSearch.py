import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# Defining CConnect 4 and Tic Tac Toe
# ---------------------------

class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3,3), dtype=int)
        self.currentPlayer = 1

    def reset(self):
        self.board.fill(0)
        self.currentPlayer = 1

    def getValidMoves(self):
        return [(i, j) for i in range(3) for j in range(3) if self.board[i, j] == 0]

    def make_a_move(self, move):
        i, j = move
        if self.board[i, j] != 0:
            raise ValueError("Invalid move!")
        self.board[i, j] = self.currentPlayer
        self.currentPlayer *= -1

    def winner(self):
        for i in range(3):
            if abs(sum(self.board[i, :])) == 3:
                return np.sign(sum(self.board[i, :]))
        for j in range(3):
            if abs(sum(self.board[:, j])) == 3:
                return np.sign(sum(self.board[:, j]))
        diagonal1 = self.board[0,0] + self.board[1,1] + self.board[2,2]
        diagonal2 = self.board[0,2] + self.board[1,1] + self.board[2,0]
        if abs(diagonal1) == 3:
            return np.sign(diagonal1)
        if abs(diagonal2) == 3:
            return np.sign(diagonal2)
        return 0

    def draw(self):
        return np.all(self.board != 0) and self.winner() == 0

    def gameOver(self):
        return self.winner() != 0 or self.draw()

    def evaluate(self):
        return self.winner()

    def copy(self):
        newGame = TicTacToe()
        newGame.board = self.board.copy()
        newGame.currentPlayer = self.currentPlayer
        return newGame

class Connect4:
    rows = 6
    cols = 7

    def __init__(self):
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.currentPlayer = 1

    def reset(self):
        self.board.fill(0)
        self.currentPlayer = 1

    def getValidMoves(self):
        return [j for j in range(self.cols) if self.board[0, j] == 0]

    def make_a_move(self, col):
        if col not in self.getValidMoves():
            raise ValueError("Invalid move!")
        for i in range(self.rows-1, -1, -1):
            if self.board[i, col] == 0:
                self.board[i, col] = self.currentPlayer
                break
        self.currentPlayer *= -1

    def winner(self):
        # horizontal check
        for i in range(self.rows):
            for j in range(self.cols - 3):
                line = self.board[i, j:j+4]
                if abs(sum(line)) == 4 and np.all(line != 0):
                    return np.sign(sum(line))
        # vertical check
        for i in range(self.rows - 3):
            for j in range(self.cols):
                line = self.board[i:i+4, j]
                if abs(sum(line)) == 4 and np.all(line != 0):
                    return np.sign(sum(line))
        # diagonal (down-right)
        for i in range(self.rows - 3):
            for j in range(self.cols - 3):
                line = [self.board[i+k, j+k] for k in range(4)]
                if abs(sum(line)) == 4 and all(v != 0 for v in line):
                    return np.sign(sum(line))
        # diagonal (up-right)
        for i in range(3, self.rows):
            for j in range(self.cols - 3):
                line = [self.board[i-k, j+k] for k in range(4)]
                if abs(sum(line)) == 4 and all(v != 0 for v in line):
                    return np.sign(sum(line))
        return 0

    def draw(self):
        return len(self.getValidMoves()) == 0 and self.winner() == 0

    def gameOver(self):
        return self.winner() != 0 or self.draw()

    def evaluate(self):
        return self.winner()

    def copy(self):
        newGame = Connect4()
        newGame.board = self.board.copy()
        newGame.currentPlayer = self.currentPlayer
        return newGame

# ---------------------------
# Minimax (with and without alpha-beta pruning)
# ---------------------------

def MiniMax(game, depth, maximizing_player, use_alpha_beta=False, alpha=float("-inf"), beta=float("inf")):
    if depth == 0 or game.gameOver():
        return game.evaluate(), None

    best_move = None
    if maximizing_player:
        max_eval = float("-inf")
        for move in game.getValidMoves():
            child = game.copy()
            if isinstance(move, tuple):
                child.make_a_move(move)
            else:
                child.make_a_move(move)
            eval, _ = MiniMax(child, depth - 1, False, use_alpha_beta, alpha, beta)
            if eval > max_eval:
                max_eval = eval
                best_move = move
            if use_alpha_beta:
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
        return max_eval, best_move
    else:
        min_eval = float("inf")
        for move in game.getValidMoves():
            child = game.copy()
            if isinstance(move, tuple):
                child.make_a_move(move)
            else:
                child.make_a_move(move)
            eval, _ = MiniMax(child, depth - 1, True, use_alpha_beta, alpha, beta)
            if eval < min_eval:
                min_eval = eval
                best_move = move
            if use_alpha_beta:
                beta = min(beta, eval)
                if beta <= alpha:
                    break
        return min_eval, best_move

# ---------------------------
# Q-Learning Agent
# ---------------------------

class Qlearning:
    def __init__(self, game_class, alpha=0.5, gamma=0.9, epsilon=0.1):
        self.game_class = game_class
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}

    def state_to_key(self, game):
        return (tuple(game.board.flatten()), game.currentPlayer)

    def get_q(self, state, move):
        return self.q_table.get((state, move), 0.0)

    def choose_action(self, game):
        valid_moves = game.getValidMoves()
        state = self.state_to_key(game)
        if random.random() < self.epsilon:
            return random.choice(valid_moves)
        qs = [self.get_q(state, move) for move in valid_moves]
        max_q = max(qs)
        best_moves = [move for move, q in zip(valid_moves, qs) if q == max_q]
        return random.choice(best_moves)

    def update(self, state, move, reward, next_state, done):
        current_q = self.get_q(state, move)
        if done:
            target = reward
        else:
            next_qs = [self.q_table.get((next_state, m), 0.0) for m in self.game_class().getValidMoves()]
            target = reward + self.gamma * max(next_qs) if next_qs else reward
        self.q_table[(state, move)] = current_q + self.alpha * (target - current_q)

    def train(self, episodes=1000, max_steps=100):
        wins = 0
        for episode in range(episodes):
            game = self.game_class()
            state = self.state_to_key(game)
            step = 0
            while not game.gameOver() and step < max_steps:
                move = self.choose_action(game)
                state = self.state_to_key(game)
                if isinstance(move, tuple):
                    game.make_a_move(move)
                else:
                    game.make_a_move(move)
                reward = 0
                done = game.gameOver()
                if done:
                    result = game.evaluate()
                    if result == 1:
                        reward = 1
                        wins += 1
                    elif result == -1:
                        reward = -1
                next_state = self.state_to_key(game)
                self.update(state, move, reward, next_state, done)
                step += 1
        return wins / episodes

# ---------------------------
# Default Opponent
# ---------------------------

def default_opponent_move(game):
    # Check if the default opponent can win
    for move in game.getValidMoves():
        CopyGame = game.copy()
        if isinstance(move, tuple):
            CopyGame.make_a_move(move)
        else:
            CopyGame.make_a_move(move)
        if CopyGame.gameOver() and CopyGame.evaluate() == game.currentPlayer:
            return move
    # Check if blocking the opponent is needed
    for move in game.getValidMoves():
        CopyGame = game.copy()
        CopyGame.make_a_move(move)
        if CopyGame.gameOver() and CopyGame.evaluate() == -game.currentPlayer:
            return move
    return random.choice(game.getValidMoves())

# ---------------------------
# Experiment Runner
# ---------------------------

def tournament(game_class, agent1_func, agent2_func, num_games=200):
    results = {"agent1_win": 0, "agent2_win": 0, "draw": 0}
    for _ in range(num_games):
        game = game_class()
        while not game.gameOver():
            if game.currentPlayer == 1:
                move = agent1_func(game)
            else:
                move = agent2_func(game)
            if isinstance(move, tuple):
                game.make_a_move(move)
            else:
                game.make_a_move(move)
        result = game.evaluate()
        if result == 1:
            results["agent1_win"] += 1
        elif result == -1:
            results["agent2_win"] += 1
        else:
            results["draw"] += 1
    return results

# ---------------------------
# Agent Definitions for Tic Tac Toe
# ---------------------------

def agent_MiniMax(game):
    _, move = MiniMax(game, depth=4, maximizing_player=(game.currentPlayer==1), use_alpha_beta=False)
    return move if move is not None else random.choice(game.getValidMoves())

def agent_MiniMax_ab(game):
    _, move = MiniMax(game, depth=4, maximizing_player=(game.currentPlayer==1), use_alpha_beta=True)
    return move if move is not None else random.choice(game.getValidMoves())

q_agent_tt = Qlearning(TicTacToe, epsilon=0.1)
def agent_qlearning(game):
    return q_agent_tt.choose_action(game)

def agent_default(game):
    return default_opponent_move(game)

# ---------------------------
# Agent Definitions for Connect 4
# ---------------------------

def agent_MiniMax_c4(game):
    _, move = MiniMax(game, depth=4, maximizing_player=(game.currentPlayer==1), use_alpha_beta=False)
    return move if move is not None else random.choice(game.getValidMoves())

def agent_MiniMax_ab_c4(game):
    _, move = MiniMax(game, depth=4, maximizing_player=(game.currentPlayer==1), use_alpha_beta=True)
    return move if move is not None else random.choice(game.getValidMoves())

q_agent_c4 = Qlearning(Connect4, epsilon=0.1)
def agent_qlearning_c4(game):
    return q_agent_c4.choose_action(game)

# ---------------------------
# Running Experiments for Tic Tac Toe: Algorithm vs Default Opponent
# ---------------------------

resTTT_AB = tournament(TicTacToe, agent_MiniMax_ab, agent_default, num_games=200)
resTTT = tournament(TicTacToe, agent_MiniMax, agent_default, num_games=200)
resTTT_QL = tournament(TicTacToe, agent_qlearning, agent_default, num_games=200)

print("Tic Tac Toe Performance: Algorithm vs Default Opponent:")
print("Minimax with AB vs Default:", resTTT_AB)
print("Minimax without AB vs Default:", resTTT)
print("Q-Learning vs Default:", resTTT_QL)

# ---------------------------
# Running Experiments for Connect 4: Algorithm vs Default Opponent
# ---------------------------

resC4_AB = tournament(Connect4, agent_MiniMax_ab_c4, agent_default, num_games=200)
resC4 = tournament(Connect4, agent_MiniMax_c4, agent_default, num_games=200)
resC4_QL = tournament(Connect4, agent_qlearning_c4, agent_default, num_games=200)

print("\nConnect 4 Performance: Algorithm vs Default Opponent:")
print("Minimax with AB vs Default:", resC4_AB)
print("Minimax without AB vs Default:", resC4)
print("Q-Learning vs Default:", resC4_QL)

# ---------------------------
# 1. Minimax with AB vs Minimax without AB
# 2. Minimax with AB vs Q-Learning
# 3. Q-Learning vs Minimax without AB
# ---------------------------

print("\n--- Algorithm vs Algorithm: Tic Tac Toe ---")
resTTT_R1 = tournament(TicTacToe, agent_MiniMax_ab, agent_MiniMax, num_games=200)
print("Minimax with AB vs Minimax without AB:", resTTT_R1)
resTTT_R2 = tournament(TicTacToe, agent_MiniMax_ab, agent_qlearning, num_games=200)
print("Minimax with AB vs Q-Learning:", resTTT_R2)
resTTT_R3 = tournament(TicTacToe, agent_qlearning, agent_MiniMax, num_games=200)
print("Q-Learning vs Minimax without AB:", resTTT_R3)

print("\n--- Algorithm vs Algorithm: Connect 4 ---")
resC4_R1 = tournament(Connect4, agent_MiniMax_ab_c4, agent_MiniMax_c4, num_games=200)
print("Minimax with AB vs Minimax without AB:", resC4_R1)
resC4_R2 = tournament(Connect4, agent_MiniMax_ab_c4, agent_qlearning_c4, num_games=200)
print("Minimax with AB vs Q-Learning:", resC4_R2)
resC4_R3 = tournament(Connect4, agent_qlearning_c4, agent_MiniMax_c4, num_games=200)
print("Q-Learning vs Minimax without AB:", resC4_R3)

# ---------------------------
# Plotting Bar Graphs in a Grid Format
# ---------------------------

experiment_titles = [
    "TTT: Minimax AB vs Default", "TTT: Minimax vs Default", "TTT: Q-Learning vs Default",
    "TTT: Minimax AB vs Minimax", "TTT: Minimax AB vs Q-Learning", "TTT: Q-Learning vs Minimax",
    "C4: Minimax AB vs Default", "C4: Minimax vs Default", "C4: Q-Learning vs Default",
    "C4: Minimax AB vs Minimax", "C4: Minimax AB vs Q-Learning", "C4: Q-Learning vs Minimax",
]

experiment_results = [
    resTTT_AB, resTTT, resTTT_QL,
    resTTT_R1, resTTT_R2, resTTT_R3,
    resC4_AB, resC4, resC4_QL,
    resC4_R1, resC4_R2, resC4_R3,
]

fig, axs = plt.subplots(4, 3, figsize=(16, 16))
axs = axs.flatten()

for ax, title, res in zip(axs, experiment_titles, experiment_results):
    labels = list(res.keys())
    values = [res[label] for label in labels]
    ax.bar(labels, values, color=['green', 'red', 'gray'])
    ax.set_title(title)
    ax.set_xlabel("Outcome")
    ax.set_ylabel("Number of Games")
    ax.set_ylim(0, 210)  # since num_games=200

plt.tight_layout(rect=[0, 0.05, 1, 1])  # leave space at bottom for legend

# Add a figure-level legend as a text box below the subplots.
legend_text = (
    "Legend:\n"
    "For each experiment title, Agent1 is the algorithm before 'vs' and Agent2 is the algorithm after 'vs'.\n"
    "Bar Colors: Green = Agent1 win, Red = Agent2 win, Gray = Draw."
)
plt.figtext(0.5, 0.01, legend_text, ha="center", fontsize=12, bbox=dict(facecolor="white", edgecolor="black", boxstyle="round"))
plt.show()

# ---------------------------
# Heatmap of All Experiment Results
# ---------------------------
# Aggregate all experiment results into a single dictionary.


all_results = {
    "TT: Minimax AB vs Default": resTTT_AB,
    "TT: Minimax vs Default": resTTT,
    "TT: Q-Learning vs Default": resTTT_QL,
    "TT: Minimax AB vs Minimax": resTTT_R1,
    "TT: Minimax AB vs Q-Learning": resTTT_R2,
    "TT: Q-Learning vs Minimax": resTTT_R3,
    "C4: Minimax AB vs Default": resC4_AB,
    "C4: Minimax vs Default": resC4,
    "C4: Q-Learning vs Default": resC4_QL,
    "C4: Minimax AB vs Minimax": resC4_R1,
    "C4: Minimax AB vs Q-Learning": resC4_R2,
    "C4: Q-Learning vs Minimax": resC4_R3,
}

# Prepare data for the heatmap: rows are experiments and columns are outcomes.
experiment_names = list(all_results.keys())
outcomes = ["agent1_win", "agent2_win", "draw"]
data = np.array([[res[outcome] for outcome in outcomes] for res in all_results.values()])

plt.figure(figsize=(12, 8))
sns.heatmap(data, annot=True, fmt="d", xticklabels=outcomes, yticklabels=experiment_names, cmap="YlGnBu")
plt.title("Heatmap of All Experiment Results")
plt.xlabel("Outcome")
plt.ylabel("Experiment")
plt.show()