use rust_rl::train::Trainer;


fn main() {
    let trainer = Trainer::new();
    // trainer.train(1_000);
    for t in trainer.agent.q_table.iter() {
        println!("{:?}", t);
    }
    trainer.agent.save_q_table_to_file("q_table_v2.csv").expect("Failed to save Q-table to file");
}