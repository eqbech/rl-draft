use std::time::Instant;

use rust_rl::{train::Trainer, GRID_SIZE};


fn main() {
    let start = Instant::now();
    let mut trainer = Trainer::new(GRID_SIZE.0, GRID_SIZE.1);
    trainer.train(10_000_000);
    trainer.agent.save_q_table_to_file("data/q_table.csv").expect("Failed to save Q-table to file");

    let elapsed = start.elapsed();
    println!("Training completed and Q-table saved to data/q_table.csv in {:?}", elapsed);
}