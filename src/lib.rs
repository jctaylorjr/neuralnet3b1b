use rand::random_range;
// fn dot_product(a: &[f64], b: &[f64]) -> f64 {
//     a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
// }

// fn matrix_vector_multiply(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
//     a.iter().map(|row| dot_product(row, b)).collect()
// }

pub struct NeuralNetwork {
    hidden_layer_count: usize,
    hidden_layer_neuron_count: usize,
    // input_layer_neuron_count: usize,
    output_layer_neuron_count: usize,
    layers: Vec<Vec<f64>>,
    weights: Vec<Vec<Vec<f64>>>,
    biases: Vec<Vec<f64>>,
}

impl NeuralNetwork {
    fn new(
        hidden_layer_count: usize,
        hidden_layer_neuron_count: usize,
        input_layer_neuron_count: usize,
        output_layer_neuron_count: usize,
        input_layer: Vec<f64>,
    ) -> Self {
        // in, hidden, out layers
        let mut layers: Vec<Vec<f64>> = Vec::new();
        layers.push(input_layer);
        for _ in 0..hidden_layer_count {
            layers.push(vec![0.0; hidden_layer_neuron_count]);
        }
        layers.push(vec![0.0; output_layer_neuron_count]);

        // biases, init to 0 and will be adjusted by asymmetry breaking during back propagation
        // https://ai.stackexchange.com/questions/14292/should-the-biases-be-zero-or-randomly-initialised
        let mut biases: Vec<Vec<f64>> = Vec::new();
        biases.push(vec![0.0; input_layer_neuron_count]);
        for _ in 0..hidden_layer_count {
            biases.push(vec![0.0; hidden_layer_neuron_count]);
        }
        biases.push(vec![0.0; output_layer_neuron_count]);

        // weights, init to random between 0 and 1 f64
        let mut weights: Vec<Vec<Vec<f64>>> = Vec::new();
        weights.push(vec![
            vec![random_range(0.0..=1.0); input_layer_neuron_count];
            hidden_layer_neuron_count
        ]);
        for _ in 1..hidden_layer_count {
            weights.push(vec![
                vec![random_range(0.0..=1.0); hidden_layer_neuron_count];
                hidden_layer_neuron_count
            ]);
        }
        weights.push(vec![
            vec![random_range(0.0..=1.0); hidden_layer_neuron_count];
            output_layer_neuron_count
        ]);

        NeuralNetwork {
            hidden_layer_count,
            hidden_layer_neuron_count,
            // input_layer_neuron_count,
            output_layer_neuron_count,
            layers,
            weights,
            biases,
        }
    }

    fn load_input_layer(&mut self, input: Vec<f64>) {
        self.layers.insert(0, input);
    }

    fn feed_forward(&mut self) {
        for i in 0..self.layers.len() {
            let z = weighted_sum(&self.weights[i], &self.layers[i], &self.biases[i]);
            self.layers
                .push(z.iter().map(|product| sigmoid(*product)).collect());
        }
    }
}

fn sigmoid_derivative(x: f64) -> f64 {
    sigmoid(x) * (1.0 - sigmoid(x))
}

fn sigmoid(x: f64) -> f64 {
    // https://calculus.subwiki.org/wiki/Logistic_function
    1.0 / (1.0 + std::f64::consts::E.powf(x))
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

fn back_propagation() {
    // http://neuralnetworksanddeeplearning.com/chap2.html
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

    // #[test]
    // fn test_dot_product() {
    //     let a = [1.0, 2.0, 3.0];
    //     let b = [4.0, 5.0, 6.0];
    //     assert_eq!(dot_product(&a, &b), 32.0);
    // }

    // #[test]
    // fn test_matrix_multiply() {
    //     let a = vec![
    //         vec![1.0, 2.0, 3.0],
    //         vec![1.0, 2.0, 3.0],
    //         vec![1.0, 2.0, 3.0],
    //     ];
    //     let b = vec![1.0, 2.0, 3.0];

    //     assert_eq!(matrix_vector_multiply(&a, &b), vec![14.0, 14.0, 14.0]);
    // }
}
