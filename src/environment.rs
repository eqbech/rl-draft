use serde::{Deserialize, Serialize};
use rand::{rng, seq::IndexedRandom, Rng};
const OPTIONS: [usize; 4] = [0, 1, 3, 4];
pub enum GameState {
    Started,
    Finished,
}
#[derive(Debug, Copy, Clone, Deserialize, Serialize)]
pub enum Action {
    Up,
    Down,
    Left,
    Right,
}
impl Action {
    pub fn get_random_state() -> Self {
        let mut rng = rand::rng();
        match rng.random_range(0..4) {
            0 => Action::Up,
            1 => Action::Down,
            2 => Action::Left,
            _ => Action::Right,
        }
    }
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
/// It contains the current state, the card to guess, the reward, and the number of guesses made.
pub struct Environment {
    pub position: (usize, usize),
    pub board: [[f32; 5]; 5],
    pub reward: f32,
    pub game_state: GameState
}

impl Environment {
    pub fn new() -> Self {
        Environment {
            position: (0, 0),
            board: [[0.; 5]; 5],
            reward: 0.0,
            game_state: GameState::Started,
        }
    }

    pub fn reset(&mut self) {
        self.position = (OPTIONS.choose(&mut rng()).unwrap().clone(), OPTIONS.choose(&mut rng()).unwrap().clone());
        self.reward = 0.0;
        self.game_state = GameState::Started;
    }

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
    fn calc_reward(&mut self) {
        if self.position == (2, 2) {
            self.reward = 10.0; // Reward for reaching the center
            self.game_state = GameState::Finished;
        } else {
            self.reward = 0.0; // Penalty for wrong guesses
        }
    }
}