use rand::{rng, seq::IndexedRandom};

use crate::Action;

/// The GameState enum represents the current state of the game.
pub enum GameState {
    /// Represents the game is in progress.
    Started,
    /// Represents the game has ended, either by reaching a goal or failing.
    Finished,
}

/// The Environment struct represents the environment in which the agent operates.
/// It contains the current position of the agent, the board state which is a `5x5` grid, the reward for the last action, and the current game state.
pub struct Environment {
    pub position: (usize, usize),
    pub board: Vec<Vec<f32>>,
    // pub walls: Vec<(usize, usize)>,
    pub reward: f32,
    pub game_state: GameState,
}

impl Environment {
    /// Creates a new Environment with an initial position, empty board, and default values for reward and game state.
    pub fn new(rows: usize, cols: usize) -> Self {
        Environment {
            position: (0, 0),
            board: vec![vec![0.0; cols]; rows], // Initialize a 5x5 grid
            reward: 0.0,
            game_state: GameState::Started,
        }
    }
    /// Resets the environment and sets a new random starting position so that our agent does not always start in the top-left corner.
    pub fn reset(&mut self) {
        let possible_options: Vec<usize> = (0..self.board.len())
            .filter(|x| *x != self.board.len() / 2)
            .collect();
        self.position = (
            *possible_options.choose(&mut rng()).unwrap(),
            *possible_options.choose(&mut rng()).unwrap(),
        );
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
                    self.calc_reward();
                    self.game_state = GameState::Finished;
                }
            }
            Action::Down => {
                if self.position.0 < self.board[0].len() - 1 {
                    self.position.0 += 1;
                    self.calc_reward();
                } else {
                    self.calc_reward();
                    self.game_state = GameState::Finished;
                }
            }
            Action::Left => {
                if self.position.1 > 0 {
                    self.position.1 -= 1;
                    self.calc_reward();
                } else {
                    self.calc_reward();
                    self.game_state = GameState::Finished;
                }
            }
            Action::Right => {
                if self.position.1 < self.board.len() - 1 {
                    self.position.1 += 1;
                    self.calc_reward();
                } else {
                    self.calc_reward();
                    self.game_state = GameState::Finished;
                }
            }
        }
    }
    /// Calculates the reward based on the agent's current position.
    /// If the agent reaches the center of the board, it receives a reward of 100.0 and the game ends.
    /// Otherwise, it calculates the reward based on the Euclidean distance from the center of the board.
    /// Using the formula `r = 1 / √(x2 – x1)^2 + (y2 – y1)^2`
    fn calc_reward(&mut self) {
        if self.position == (self.board.len() / 2, self.board[0].len() / 2) {
            self.reward = 100.0;
            self.game_state = GameState::Finished;
        } else {
            self.reward = 1.
                / f32::sqrt(
                    (self.position.0 as f32 - (self.board.len() / 2) as f32).powi(2)
                        + (self.position.1 as f32 - (self.board[0].len() / 2) as f32).powi(2),
                );
        }
    }
}
