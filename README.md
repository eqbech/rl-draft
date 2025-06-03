# Reinforcement Learning Simple Example

This repository is a simple demonstration of reinforcement learning using Q‑learning.

## Overview

- **Environment**  
  The environment is modeled as a **5×5 grid**.  
  - **Agent Positioning:** The agent’s state is its current position on the grid, defined as a tuple `(row, column)`.  
  - **Movement Rules:**  
    - The agent can move one step at a time in one of four directions: Up, Down, Left, or Right.  
    - The goal is to **reach the center of the grid**.  
    - If the agent moves out of bounds, the episode fails.

- **Agent**  
  The agent is built using a Q‑learning algorithm with an epsilon‑greedy action selection strategy.  
  - **Q‑Learning Update Rule:**  
    The update rule is given by:

    ```
    Q(s, a) ← Q(s, a) + α · (r + γ · maxₐ' Q(s', a') − Q(s, a))
    ```

    - **α (alpha):** Learning rate.  
    - **γ (gamma):** Discount factor for future rewards.
  - The agent learns to choose actions based on its Q‑table, which estimates the expected reward for each (state, action) pair.

- **Trainer**  
  The Trainer module runs the training loop where:
  - The environment is reset at the beginning of each episode.
  - The agent interacts with the environment step-by-step.
  - The agent receives rewards and updates its Q‑table via the Q‑learning update rule until it either reaches the center or goes out of bounds.

- **Server**  
  The server uses Actix‑web to expose a REST API endpoint that provides access to the agent’s learned Q‑table (or can be extended to serve predictions). This is useful for demonstration and observation of how the agent makes decisions.


## Project Structure

- `src/agent.rs` – Contains the implementation of the Q‑learning agent.
- `src/environment.rs` – Contains the simulation of the environment and its actions.
- `src/train.rs` – Contains the trainer that handles the training loop.
- `src/bin/train.rs` – A binary for training the agent.
- `src/bin/server.rs` – A binary for running the REST API server.

## How to Use

### Prerequisites

- Rust (latest stable version)
- Cargo

### Building the Project

Clone the repository and navigate to its directory, then build with Cargo:

```bash
git clone https://github.com/your-org/rust_rl.git
cd rust_rl
cargo build --release
```

### Training the Agent

You can train the agent by running the training binary. This will run the training loop for a specified number of episodes and save the resulting Q‑table as a CSV file.

```bash
cargo run --bin train
```

After training, the Q‑table is saved to a CSV file (e.g., `data/q_table.csv`) and printed on the terminal.

### Running the Server

After training (or if you load a saved Q‑table), you can start the server. The server exposes a `/q_table` endpoint that returns the agent’s `Q-table`

```bash
cargo run --bin server
```

Access the server by opening your browser or using a tool like `curl`:

```bash
curl "http://127.0.0.1:8080/q_table"
```
