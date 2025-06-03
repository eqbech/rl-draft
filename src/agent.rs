use std::io::{BufRead, Write, BufReader};
use rand::Rng;

use crate::environment::Action;

/// The Agent struct represents the agent in the environment.
/// It contains methods for the agent to act and learn from the environment.
pub struct Agent {
    pub q_table: [[[f32; 4]; 5]; 5],
    epsilon: f32,
    alpha: f32,
    gamma: f32,
}
impl Agent {
    pub fn new() -> Self {
        Agent { 
            // Q-table initialized with zeros
            q_table: [[[0.; 4]; 5]; 5],
            epsilon: 0.2,
            alpha: 0.1,
            gamma: 0.9
        }
    }

    /// The act method takes the current state of the environment and returns an action.
    /// The actions space consists of integers from 2 - 14, representing the possible card values.
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

    /// The learn method applies the Q‑learning update using the state, action, reward,
    /// and next_state.
    pub fn learn(&mut self, state: (usize, usize), action: Action, reward: f32, next_state: (usize, usize)) {        
        // Find the maximum Q-value for the next state over all actions.
        let mut max_q_next = 0.0;
        for a in 0..4 {
            let q_val = self.q_table[next_state.0][next_state.1][a as usize];
            if q_val > max_q_next {
                max_q_next = q_val;
            }
        }
        
        // Q-learning update: Q(s,a) = Q(s,a) + α * (reward + γ * max_a' Q(s′,a') – Q(s,a))
        self.q_table[state.0][state.1][action as usize] = self.alpha * (reward + self.gamma * max_q_next - self.q_table[state.0][state.1][action as usize]);
    }

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