use rand;

pub struct NeuralNetwork {
    pub layers: Vec<Vec<f64>>,
    pub weights: Vec<Vec<Vec<f64>>>,
    pub biases: Vec<Vec<f64>>,
    pub activation_values: Vec<Vec<f64>>,
}

impl NeuralNetwork {
    pub fn new(
        hidden_layer_count: usize,
        hidden_layer_neuron_count: usize,
        output_layer_neuron_count: usize,
        input_layer: Vec<f64>,
    ) -> Self {
        // not including input layer
        let mut layer_sizes = vec![hidden_layer_neuron_count; hidden_layer_count];
        layer_sizes.push(output_layer_neuron_count);

        // input, hidden, and out layers
        let mut layers: Vec<Vec<f64>> = vec![input_layer];
        layers.extend(
            layer_sizes
                .iter()
                .map(|size| vec![0.0; *size])
                .collect::<Vec<Vec<f64>>>(),
        );

        // biases, init to 0 and will be adjusted by asymmetry breaking during back propagation
        // https://ai.stackexchange.com/questions/14292/should-the-biases-be-zero-or-randomly-initialised
        let mut biases: Vec<Vec<f64>> = layer_sizes.iter().map(|size| vec![0.0; *size]).collect();

        // weights, init to random between 0 and 1 f64
        let mut weights: Vec<Vec<Vec<f64>>> = layers
            .iter()
            .map(|l| l.len())
            .zip(layer_sizes.iter())
            .map(|(pre_layer, next_layer)| {
                (0..pre_layer)
                    .map(|_| {
                        (0..*next_layer)
                            .map(|_| rand::random_range(0.0..=1.0))
                            .collect()
                    })
                    .collect()
            })
            .collect();

        NeuralNetwork {
            layers,
            weights,
            biases,
            activation_values: Vec::new(),
        }
    }

    pub fn feed_forward(&mut self) {
        for i in 0..self.layers.len() - 1 {
            let z = weighted_sum(&self.weights[i], &self.layers[i], &self.biases[i]);
            self.layers[i + 1] = z.iter().map(|product| sigmoid(*product)).collect();
            self.activation_values.push(z);
        }
    }

    fn back_propagation(&mut self) {
        // http://neuralnetworksanddeeplearning.com/chap2.html
    }
}

fn weighted_sum(weights: &[Vec<f64>], layer: &[f64], biases: &[f64]) -> Vec<f64> {
    // (aka z values/preactivation value) summation of weights * preceding_layer neurons + biases
    weights
        .iter()
        .map(|row: &Vec<f64>| {
            row.iter()
                .zip(layer.iter())
                .map(|(x, y)| x * y)
                .sum::<f64>()
        })
        .zip(biases.iter())
        .map(|(product, bias)| product + bias)
        .collect()
}

fn _sigmoid_derivative(x: f64) -> f64 {
    sigmoid(x) * (1.0 - sigmoid(x))
}

fn sigmoid(x: f64) -> f64 {
    // https://calculus.subwiki.org/wiki/Logistic_function
    1.0 / (1.0 + std::f64::consts::E.powf(x))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feed_forward() {
        let weights = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
        ];
        let input_layer = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let bias = vec![1.0, 2.0, 4.0];
        let expected = vec![
            sigmoid(55.0 + 1.0),
            sigmoid(55.0 + 2.0),
            sigmoid(55.0 + 4.0),
        ];
        assert_eq!(weighted_sum(&weights, &input_layer, &bias), expected);
    }

    #[test]
    fn test_sigmoid() {
        assert_eq!(sigmoid(0.0), 0.5);
        assert_eq!(sigmoid(1.0), 1.0 / (1.0 + std::f64::consts::E));
    }
}
