use crate::Agent;

pub struct DQNAgent {
    pub q_table: Vec<Vec<Vec<f32>>>,
    pub epsilon: f32,
    pub alpha: f32,
    pub gamma: f32,
}
impl Agent for DQNAgent {
    fn act(&mut self, state: (usize, usize)) -> crate::Action {
        todo!()
    }

    fn learn(&mut self, state: (usize, usize), action: crate::Action, reward: f32, next_state: (usize, usize)) {
        todo!()
    }

    fn predict(&self, state: (usize, usize)) -> crate::Action {
        todo!()
    }

    fn save_to_file(&self, file_path: &str) -> std::io::Result<()> {
        todo!()
    }

    fn load(path: &str, rows: usize, cols: usize) -> Result<Self, std::io::Error> where Self: Sized {
        todo!()
    }
}