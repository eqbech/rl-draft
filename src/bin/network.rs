use rust_rl::agents::network::nn::{self, NeuralNetwork};

fn main() {
    let mut network = 
    NeuralNetwork::new(vec![2, 2, 1], 0.1,
         nn::ActivationFunction::Sigmoid, nn::ActivationFunction::Tanh, nn::LossFunction::MeanSquaredError);

    // Input data represents the height and weight of a individual, the model is to guess male or female.
    let input_data = vec![vec![189.0, 100.0], vec![165.0, 55.0], 
                          vec![190.0, 70.0], vec![160.0, 80.0], vec![178.0, 180.0]];
    // Output
    let output_data = vec![vec![1.0], vec![0.0], vec![1.0], vec![0.0], vec![1.0]];
    
    network.train(input_data, output_data);
}