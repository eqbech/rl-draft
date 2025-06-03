use serde::{Deserialize, Serialize};
use rand::{rng, seq::IndexedRandom, Rng};

/// The OPTIONS array contains the possible starting positions for the agent in the environment.
/// We exclude 2 as we dont to have the possibility of starting in the center of the grid. e.g. winning immediately.
const OPTIONS: [usize; 4] = [0, 1, 3, 4];

/// The GameState enum represents the current state of the game.
pub enum GameState {
    /// Represents the game is in progress.
    Started,
    /// Represents the game has ended, either by reaching a goal or failing.
    Finished,
}
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
    /// Converts an integer action to an Action enum.
    pub fn from(action: i8) -> Self {
        match action {
            0 => Action::Up,
            1 => Action::Down,
            2 => Action::Left,
            3 => Action::Right,
            _ => panic!("Invalid action"),
        }
    }
}

/// The Environment struct represents the environment in which the agent operates.
/// It contains the current position of the agent, the board state which is a `5x5` grid, the reward for the last action, and the current game state.
pub struct Environment {
    pub position: (usize, usize),
    pub board: [[f32; 5]; 5],
    pub reward: f32,
    pub game_state: GameState
}

impl Environment {
    /// Creates a new Environment with an initial position, empty board, and default values for reward and game state.
    pub fn new() -> Self {
        Environment {
            position: (0, 0),
            board: [[0.; 5]; 5],
            reward: 0.0,
            game_state: GameState::Started,
        }
    }
    /// Resets the environment and sets a new random starting position so that our agent does not always start in the top-left corner.
    pub fn reset(&mut self) {
        self.position = (OPTIONS.choose(&mut rng()).unwrap().clone(), OPTIONS.choose(&mut rng()).unwrap().clone());
        self.reward = 0.0;
        self.game_state = GameState::Started;
    }

    /// Steps through the environment based on the action taken by the agent.
    /// It updates the agent's position, calculates the reward, and checks if the game is finished.
    pub fn step(&mut self, action: Action) {
        match action {
            Action::Up => {
                if self.position.0 > 0 {
                    self.position.0 -= 1;
                    self.calc_reward();
                } else {
                    self.reward = -1.0;
                    self.game_state = GameState::Finished;
                }
            },
            Action::Down => {
                if self.position.0 < 4 {
                    self.position.0 += 1;
                    self.calc_reward();
                } else {
                    self.reward = -1.0;
                    self.game_state = GameState::Finished;
                }
            },
            Action::Left => {
                if self.position.1 > 0 {
                    self.position.1 -= 1;
                    self.calc_reward();
                } else {
                    self.reward = -1.0;
                    self.game_state = GameState::Finished;
                }
            },
            Action::Right => {
                if self.position.1 < 4 {
                    self.position.1 += 1;
                    self.calc_reward();
                } else {
                    self.reward = -1.0;
                    self.game_state = GameState::Finished;
                }
            },
        }
    }
    /// Calculates the reward based on the agent's current position.
    fn calc_reward(&mut self) {
        if self.position == (2, 2) {
            self.reward = 10.0; // Reward for reaching the center
            self.game_state = GameState::Finished;
        } else {
            self.reward = 0.0; // Penalty for wrong guesses
        }
    }
}