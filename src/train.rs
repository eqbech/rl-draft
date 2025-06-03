use std::fmt::Write;

use indicatif::{ProgressBar, ProgressState, ProgressStyle};

use crate::{agent::Agent, environment::Environment};

/// The Trainer struct is responsible for managing the training process of the agent in the environment.
pub struct Trainer {
    pub agent: Agent,
    pub environment: Environment,
}

impl Trainer {
    /// Creates a new Trainer with an initialized Agent and Environment.
    pub fn new(rows: usize, cols: usize) -> Self {
        if (rows % 2 == 0) || (cols % 2 == 0) {
            panic!("Rows and columns must be odd numbers for the environment.");
        }
        Trainer {
            agent: Agent::new(rows, cols),
            environment: Environment::new(rows, cols),
        }
    }
    /// Trains the agent by running a specified number of episodes in the environment.
    /// Each episode consists of the agent taking actions in the environment until a terminal state is reached. e.g. the agent either won or lost.
    pub fn train(&mut self, episodes: u64) {
        let pb = ProgressBar::new(episodes);
        pb.set_style(ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] [{bar:80.cyan/blue}] {pos}/{len} ({eta})")
        .unwrap()
        .with_key("eta", |state: &ProgressState, w: &mut dyn Write| write!(w, "{:.1}s", state.eta().as_secs_f64()).unwrap())
        .progress_chars("#>-"));
        for episode in 1..=episodes {
            self.environment.reset();
            let mut state = self.environment.position;
            let mut done = false;

            while !done {
                let action = self.agent.act(state);
                self.environment.step(action);
                let reward = self.environment.reward;
                let next_state = self.environment.position;

                // Update the Q-table using Q-learning update
                self.agent.learn(state, action, reward, next_state);

                // Set the next state as current for the following iteration
                state = next_state;

                // If the state is Correct, consider the episode finished.
                match self.environment.game_state {
                    crate::environment::GameState::Started => (),
                    crate::environment::GameState::Finished => done = true,
                }
            }
            pb.set_position(episode);
        }
    }
}