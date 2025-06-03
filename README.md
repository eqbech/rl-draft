# Reinforcement Learning Simple Example

This repository is a simple demonstration of reinforcement learning using Q‑learning.

## Overview

The repository contains a basic implementation of a Q‑learning agent that learns to make decisions based on the state of an environment. Key components include:

- **Agent** – Implements the Q‑learning algorithm with an epsilon‑greedy strategy for action selection. The agent updates its Q‑table using the standard update rule:
  
  ```
  Q(s, a) ← Q(s, a) + α · (r + γ · maxₐ' Q(s', a') − Q(s, a))
  ```

- **Environment** – Simulates the environment in which the agent operates. It provides the current state, processes actions, and returns rewards.

- **Trainer** – Manages the training loop by running episodes where the agent interacts with the environment, learns from rewards, and updates its `Q‑table`.

- **Server** – Exposes a RESTful endpoint (using Actix‑web) to retrieve the agent’s `Q.-table`.

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
