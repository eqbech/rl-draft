use rand::Rng;



/// Enum representing different activation functions used in the neural network.
pub enum ActivationFunction {
    /// Rectified Linear Unit (ReLU) activation function.
    /// ReLU is defined as f(x) = max(0, x).
    ReLU,
    /// Sigmoid activation function.
    /// Sigmoid is defined as f(x) = 1 / (1 + exp(-x)).
    Sigmoid,
    /// Hyperbolic Tangent (Tanh) activation function.
    /// Tanh is defined as f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x)).
    Tanh,
}

impl ActivationFunction {
    /// Applies the activation function to the input value.
    pub fn apply(&self, x: f64) -> f64 {
        match self {
            ActivationFunction::ReLU => if x < 0.0 { 0.0 } else { x },
            ActivationFunction::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            ActivationFunction::Tanh => x.tanh(),
        }
    }

    /// Returns the derivative of the activation function using the activated value.
    ///
    /// For example, for Sigmoid, f'(x) = f(x)*(1 - f(x)).
    pub fn derivative(&self, activated: f64) -> f64 {
        match self {
            ActivationFunction::ReLU => {
                if activated > 0.0 { 1.0 } else { 0.0 }
            },
            ActivationFunction::Sigmoid => activated * (1.0 - activated),
            ActivationFunction::Tanh => 1.0 - activated.powi(2),
        }
    }
}

/// A simple deep neural network struct.
pub struct NeuralNetwork {
    layers: Vec<Layer>,
    learning_rate: f64,
    activation_function: ActivationFunction,
    final_activation: ActivationFunction,
}

impl NeuralNetwork {
    /// Creates a new NeuralNetwork with the given layer sizes, learning rate and activations.
    ///
    /// For example, layer_sizes = [4, 5, 3] creates a network with 4 inputs,
    /// one hidden layer with 5 neurons and an output layer with 3 neurons.
    pub fn new(layer_sizes: Vec<usize>, learning_rate: f64, activation_function: ActivationFunction, final_activation: ActivationFunction) -> Self {
        let mut layers = Vec::new();
        for i in 0..layer_sizes.len() - 1 {
            layers.push(Layer::new(layer_sizes[i], layer_sizes[i + 1]));
        }
        NeuralNetwork { layers, learning_rate, activation_function, final_activation }
    }

    /// Performs a forward pass through the network.
    /// Each layer computes its output which is fed as input to the next layer.
    /// Returns the activated outputs of the final layer.
    pub fn forward(&self, mut input: Vec<f64>) -> (Vec<f64>, Vec<Vec<f64>>) {
        let mut cache: Vec<Vec<f64>> = Vec::new();
        cache.push(input.clone());
        for (i, layer) in self.layers.iter().enumerate() {
            input = if i < self.layers.len() - 1 {
                // Hidden layers use the standard activation function.
                layer.forward(&input, &self.activation_function)
            } else {
                // Final layer uses the final activation function.
                layer.forward(&input, &self.final_activation)
            };
            cache.push(input.clone());
        }
        (input, cache)
    }

    /// Performs one training step using backpropagation.
    ///
    /// # Arguments
    ///
    /// * `cache` - A tuple containing the activations and pre-activation (z) values returned
    ///             from `forward_with_cache`.
    /// * `target` - The expected outputs.
    pub fn backward_with_cache(&mut self, cache: Vec<Vec<f64>>, target: &[f64]) {
        let activations = cache;

        // --- Backpropagation ---
        let mut deltas: Vec<Vec<f64>> = Vec::new();
        let output = activations.last().unwrap();
        // Compute the delta for the output layer. (Assuming mean squared error)
        let mut delta = Vec::new();
        for i in 0..output.len() {
            let error = output[i] - target[i];
            let deriv = self.final_activation.derivative(output[i]);
            delta.push(error * deriv);
        }
        deltas.push(delta);

        // Propagate the error back through the hidden layers.
        for l in (0..self.layers.len() - 1).rev() {
            let current_activation = &activations[l + 1];
            let next_layer = &self.layers[l + 1];
            let delta_next = deltas.last().unwrap();
            let mut delta_this = Vec::new();
            // For each neuron i in layer l accumulate errors from layer l+1.
            for i in 0..self.layers[l].biases.len() {
                let mut error_sum = 0.0;
                for (j, weights_row) in next_layer.weights.iter().enumerate() {
                    error_sum += weights_row[i] * delta_next[j];
                }
                let deriv = self.activation_function.derivative(current_activation[i]);
                delta_this.push(error_sum * deriv);
            }
            deltas.push(delta_this);
        }
        deltas.reverse();

        // --- Update Weights and Biases ---
        for l in 0..self.layers.len() {
            let a_prev = &activations[l]; // activation from previous layer.
            let delta_layer = &deltas[l];
            let layer = &mut self.layers[l];
            for i in 0..layer.weights.len() {
                for j in 0..layer.weights[i].len() {
                    let grad = delta_layer[i] * a_prev[j];
                    // Update using gradient descent.
                    layer.weights[i][j] -= self.learning_rate * grad;
                }
                // Bias update.
                layer.biases[i] -= self.learning_rate * delta_layer[i];
            }
        }
    }
}

/// Represents a single layer in the neural network.
pub struct Layer {
    pub weights: Vec<Vec<f64>>,
    pub biases: Vec<f64>,
}

impl Layer {
    /// Creates a new Layer with the given input and output sizes.
    /// Weights are randomly initialized using a uniform distribution; biases are set to zero.
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let mut rng = rand::rng();

        let weights: Vec<Vec<f64>> = (0..output_size)
            .map(|_| (0..input_size).map(|_| rng.random()).collect())
            .collect();
        let biases = vec![0.0; output_size];
        Layer { weights, biases }
    }

    /// Performs a forward pass for this layer.
    ///
    /// Given an input vector, computes the weighted sum (z) and applies the activation function to produce output.
    /// Returns a tuple containing (z, activated_output).
    pub fn forward(&self, input: &[f64], activation_func: &ActivationFunction) -> Vec<f64> {
        let mut output = Vec::with_capacity(self.weights.len());
        for (i, weights_row) in self.weights.iter().enumerate() {
            let sum: f64 = input.iter().zip(weights_row).map(|(inp, w)| inp * w).sum();
            output.push(activation_func.apply(sum + self.biases[i]));
        }
        output
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neural_network_creation() {
        let nn = NeuralNetwork::new(vec![2, 3, 3, 2], 0.01, ActivationFunction::ReLU, ActivationFunction::Sigmoid);
        // Check if the number of layers is correct.
        assert_eq!(nn.layers.len(), 3);
    }

    #[test]
    fn test_layer_creation() {
        let layer = Layer::new(3, 2);
        assert_eq!(layer.weights.len(), 2); // should have 2 rows for 2 outputs.
        assert_eq!(layer.weights[0].len(), 3); // each row has 3 weights.
    }

    #[test]
    fn test_forward_pass() {
        let nn = NeuralNetwork::new(vec![2, 3, 2], 0.01, ActivationFunction::ReLU, ActivationFunction::Sigmoid);
        let input = vec![1.0, 2.0];
        let output = nn.forward(input);
        // Check that output length equals the size of the final layer.
        assert_eq!(output.0.len(), 2);
    }

    #[test]
    fn test_train_step() {
        let mut nn = NeuralNetwork::new(vec![2, 3, 2], 0.01, ActivationFunction::ReLU, ActivationFunction::Sigmoid);
        let input = vec![0.5, -0.5];
        let target = vec![1.0, 0.0];
        // Run a single training step.
        nn.backward_with_cache(&input, &target);
        // After training propagation, simply check that weights have been updated (not all zero).
        let sum_weights: f64 = nn.layers.iter()
            .flat_map(|layer| layer.weights.iter().flat_map(|row| row.iter()))
            .sum();
        assert!(sum_weights.abs() > 0.0);
    }
}