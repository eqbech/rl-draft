use rust_rl::train::Trainer;


fn main() {
    let mut trainer = Trainer::new();
    trainer.train(10_000);
    for t in trainer.agent.q_table.iter() {
        println!("{:?}", t);
    }
    trainer.agent.save_q_table_to_file("data/q_table.csv").expect("Failed to save Q-table to file");
}