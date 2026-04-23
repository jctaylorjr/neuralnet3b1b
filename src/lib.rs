use std::iter;

use rand;

pub struct NeuralNetwork {
    pub layers: Vec<Vec<f64>>,
    pub weights: Vec<Vec<Vec<f64>>>,
    pub biases: Vec<Vec<f64>>,
    pub z_values: Vec<Vec<f64>>,
    pub cost_vector: Vec<f64>,
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
            z_values: Vec::new(),
            cost_vector: Vec::new(),
        }
    }

    pub fn feed_forward(&mut self) {
        // pub fn feed_forward(&mut self, input_layer: Vec<f64>) {
        // self.layers[0] = input_layer;
        for i in 0..self.layers.len() - 1 {
            let z = weighted_sum(&self.weights[i], &self.layers[i], &self.biases[i]);
            self.layers[i + 1] = z.iter().map(|product| sigmoid(*product)).collect();
            self.z_values.push(z);
        }
    }

    pub fn back_propagation(&mut self, expected_output: &Vec<f64>) {
        // http://neuralnetworksanddeeplearning.com/chap2.html
        // BP1, delta is da/dz * dC/da (which is sigma'(z) * 2(a-y) in video, and delta used by article 3b1b used)
        let mut delta = self
            .layers
            .last()
            .unwrap()
            .iter()
            .zip(expected_output.iter())
            .map(|(a, y)| 2.0 * (a - y))
            .zip(self.z_values.last().unwrap())
            .map(|(cost, z)| cost * sigmoid_derivative(*z))
            .collect();

        // Adjusting weights
        let len = self.layers.len();
        for i in (1..len - 1).rev() {
            // Implementation for adjusting weights
        }

        // Adjusting biases
        // self.cost = cost;
        self.cost_vector = delta;
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

fn sigmoid_derivative(x: f64) -> f64 {
    sigmoid(x) * (1.0 - sigmoid(x))
}

fn sigmoid(x: f64) -> f64 {
    // https://calculus.subwiki.org/wiki/Logistic_function
    1.0 / (1.0 + std::f64::consts::E.powf(x))
}

fn cost(actual: &[f64], expected: &[f64]) -> f64 {
    // sum of (actual - expected)^2 / number of outputs
    actual
        .iter()
        .zip(expected.iter())
        .map(|(a, e)| (a - e).powf(2.0))
        .sum::<f64>()
        / actual.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weighted_sum() {
        let weights = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
        ];
        let input_layer = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let bias = vec![1.0, 2.0, 4.0];
        let expected = vec![55.0 + 1.0, 55.0 + 2.0, 55.0 + 4.0];
        assert_eq!(weighted_sum(&weights, &input_layer, &bias), expected);
    }

    #[test]
    fn test_sigmoid() {
        assert_eq!(sigmoid(0.0), 0.5);
        assert_eq!(sigmoid(1.0), 1.0 / (1.0 + std::f64::consts::E));
    }
}
