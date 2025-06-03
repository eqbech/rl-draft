use crate::{agent::Agent, environment::Environment};

pub struct Trainer {
    pub agent: Agent,
    pub environment: Environment,
}

impl Trainer {
    pub fn new() -> Self {
        Trainer {
            agent: Agent::load("q_table.csv").expect("Failed to load Agent with Q-table"),
            environment: Environment::new(),
        }
    }

    pub fn train(&mut self, episodes: i32) {
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
            if episode % 1000 == 0 {
                println!("{} Episodes completed!", episode);
            }
        }
    }
}