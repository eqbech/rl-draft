use std::io::{BufRead, Write, BufReader};
use rand::Rng;

use crate::environment::Action;

/// The Agent struct represents the agent that is going to interact and learn from the environment.
/// It contains methods for learning and acting with the environment and useful utils such as loading and saving Q-tables.
pub struct Agent {
    /// Q-table is a 3D array where containing the Q-values for each state-action pair.
    pub q_table: [[[f32; 4]; 5]; 5],
    /// Epsilon-greedy parameters for exploration vs exploitation ε where (0 ≤ ε ≤ 1)
    /// A higher epsilon means more exploration, while a lower epsilon means more exploitation.
    epsilon: f32,
    /// Learning rate α where (0 < α ≤ 1)
    /// A higher alpha means the agent learns more quickly from new information.
    alpha: f32,
    /// Discount factor 𝛾 for future rewards where (0 ≤ γ < 1)
    /// A higher gamma means the agent values future rewards more.
    gamma: f32,
}
impl Agent {
    /// Creates a new Agent with an initialized Q-table and parameters for learning.
    pub fn new() -> Self {
        Agent { 
            // Q-table initialized with zeros
            q_table: [[[0.; 4]; 5]; 5],
            epsilon: 0.2,
            alpha: 0.1,
            gamma: 0.9
        }
    }

    /// Returns an action for the given state using an epsilon-greedy strategy.
    ///
    /// If a randomly generated number is less than `epsilon`, a random action is chosen
    /// (exploration). Otherwise, the agent selects the action with the highest Q-value 
    /// in the current state (exploitation).
    ///
    /// # Arguments
    ///
    /// * `state` - A tuple representing the current state.
    pub fn act(&mut self, state: (usize, usize)) -> Action {
        let mut rng = rand::rng();
        let action = if rng.random::<f32>() < self.epsilon {
            // Exploration: choose a random action
            Action::get_random_state()
        } else {
            // Exploitation: choose the best action based on Q-values
            let mut best_action = 0;
            let mut best_value = f32::MIN;
            for action in 0..4 {
                let q_value: f32 = self.q_table[state.0][state.1][action as usize];
                if q_value > best_value {
                    best_value = q_value;
                    best_action = action;
                }
            }
            Action::from(best_action)
        };
        action
    }

    /// Applies the **Q‑learning update** to the Q‑table.
    ///
    /// Given a state `s`, action `a`, reward `r`, and next state `s'`, the Q‑value update is computed as:
    ///
    /// ```math
    /// Q(s, a) ← Q(s, a) + α · (r + γ · maxₐ' Q(s', a') − Q(s, a))
    /// ```
    ///
    /// where:
    /// - **α** is the learning rate,
    /// - **γ** is the discount factor.
    ///
    /// **Parameters:**
    ///
    /// - `state`: The current state, as a tuple `(usize, usize)`.
    /// - `action`: The action taken.
    /// - `reward`: The immediate reward received.
    /// - `next_state`: The state resulting after taking the action.
    pub fn learn(&mut self, state: (usize, usize), action: Action, reward: f32, next_state: (usize, usize)) {        
        // Find the maximum Q-value for the next state over all actions.
        let mut max_q_next = 0.0;
        for a in 0..4 {
            let q_val = self.q_table[next_state.0][next_state.1][a as usize];
            if q_val > max_q_next {
                max_q_next = q_val;
            }
        }

        self.q_table[state.0][state.1][action as usize] += self.alpha * (reward + self.gamma * max_q_next - self.q_table[state.0][state.1][action as usize]);
    }

    /// Loads a Q-table from a CSV file.
    pub fn load(path: &str) -> Result<Self, std::io::Error> {
        let file = std::fs::File::open(path)?;
        let reader = BufReader::new(file);
        let mut agent = Agent {
            q_table: [[[0.; 4]; 5]; 5],
            epsilon: 0.2,
            alpha: 0.1,
            gamma: 0.9
        };
        reader.lines().enumerate().for_each(|(i, val)| {
            let line = val.unwrap();
            if i > 0 {
                let values: Vec<f32> = line.split(',')
                    .map(|s| s.trim().parse().unwrap_or(0.0))
                    .collect();
                agent.q_table[(i - 1)/ 5][(i - 1) % 5] = [values[0], values[1], values[2], values[3]];
        }
        });
        Ok(agent)
    }
    
    /// Predicts the best action for a given state based on the Q-table.
    pub fn predict(&self, state: (usize, usize)) -> Action {
        let mut best_action = 0;
        let mut best_value = f32::MIN;
        for a in 0..=4 {
            let q_value: f32 = self.q_table[state.0][state.1][a as usize];
            if q_value > best_value {
                best_value = q_value;
                best_action = a;
            }
        }
        Action::from(best_action)
    }

    /// Saves the Q-table to a CSV file.
    pub fn save_q_table_to_file(&self, path: &str) -> Result<(), std::io::Error> {
        let mut file = std::fs::File::create(path)?;
        writeln!(file, "Up, Down, Left, Right")?;
        for row in self.q_table.iter() {
            for col in row.iter() {
                let line = col.iter().map(|v| format!("{:.2}", v)).collect::<Vec<String>>().join(",");
                writeln!(file, "{}", line)?;
            }
        }
        Ok(())
    }

    /// Converts the Q-table to a 2D grid of actions, where each cell contains the action with the highest Q-value.
    /// This is useful for visualizing the agent's Q-table in a more human-readable format.
    pub fn q_table_to_2_dim_grid(&self) -> [[Action; 5]; 5] {
        let mut grid = [[Action::Down; 5]; 5];
        for i in 0..5 {
            for j in 0..5 {
                grid[i][j] = Action::from(self.q_table[i][j].iter().enumerate().max_by(|(_, a), (_, b)| a.total_cmp(b)).map(|(index, _)| index).unwrap() as i8);

            }
        }
        grid
    }
}