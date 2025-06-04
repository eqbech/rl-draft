use rand::Rng;
use serde::{Deserialize, Serialize};

pub mod agents;
pub mod environment;
pub mod train;

pub const GRID_SIZE: (usize, usize) = (13, 13);

/// The Action enum represents the possible actions the agent can take in the environment.
#[derive(Debug, Copy, Clone, Deserialize, Serialize)]
pub enum Action {
    Up,
    Down,
    Left,
    Right,
}
impl Action {
    /// Returns a random action from the available actions.
    /// This is useful for exploration in reinforcement learning.
    pub fn get_random_state() -> Self {
        let mut rng = rand::rng();
        match rng.random_range(0..4) {
            0 => Action::Up,
            1 => Action::Down,
            2 => Action::Left,
            _ => Action::Right,
        }
    }
    /// Converts a usize to an Action enum.
    pub fn from_usize(action: usize) -> Self {
        match action {
            0 => Action::Up,
            1 => Action::Down,
            2 => Action::Left,
            3 => Action::Right,
            _ => panic!("Invalid action"),
        }
    }
    /// Converts an Action enum to a usize.
    pub fn to_usize(&self) -> usize {
        match self {
            Action::Up => 0,
            Action::Down => 1,
            Action::Left => 2,
            Action::Right => 3,
        }
    }
}

pub trait Agent {
    /// Returns the action to be taken in the current state.
    ///
    /// # Arguments
    ///
    /// * `state` - A tuple representing the current state.
    fn act(&mut self, state: (usize, usize)) -> Action;

    /// Updates the agent's knowledge based on the action taken and the received reward.
    ///
    /// # Arguments
    ///
    /// * `state` - The current state before taking the action.
    /// * `action` - The action taken.
    /// * `reward` - The immediate reward received after taking the action.
    /// * `next_state` - The state after taking the action.
    fn learn(
        &mut self,
        state: (usize, usize),
        action: Action,
        reward: f32,
        next_state: (usize, usize),
    );

    /// Predicts the next action based on the current state.
    /// # Arguments
    /// * `state` - A tuple representing the current state.
    /// Returns the predicted action.
    fn predict(&self, state: (usize, usize)) -> Action;

    /// Saves the agent's state to a file.
    /// # Arguments
    /// /// * `file_path` - The path to the file where the agent's state will be saved.
    /// Returns a `Result` indicating success or failure.
    fn save_to_file(&self, file_path: &str) -> std::io::Result<()>;

    /// Loads the agent's state from a file.
    /// # Arguments
    /// /// * `file_path` - The path to the file from which the agent's state will be loaded.
    /// Returns a `Result` containing the agent if successful, or an error if not.
    fn load(path: &str, rows: usize, cols: usize) -> Result<Self, std::io::Error>
    where
        Self: Sized;
}
