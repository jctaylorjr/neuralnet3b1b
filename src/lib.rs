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

    // pub fn sgd(&mut self, batch_size, epochs, learn_rate, training_data, test_data) {
    //     let shuffled = training_data.shuffle();
    //     let batches: Vec<example> = training_data.iter().split_into_vecs().collect();
    //     // multithread backprop
    //     // mutarc nn for batches, join

    // }

    // redo
    // pub fn train(&mut self, training_data: Vec<(Vec<f64>, Vec<f64>)>, epochs: usize) {
    //     for _ in 0..epochs {
    //         // shuffle training data
    //         let mut shuffled_data = training_data.clone();
    //         rand::thread_rng().shuffle(&mut shuffled_data);

    //         // create mini-batches
    //         let mini_batches: Vec<Vec<(Vec<f64>, Vec<f64>)>> = shuffled_data
    //             .chunks(self.batch_size)
    //             .map(|chunk| chunk.to_vec())
    //             .collect();

    //         for mini_batch in mini_batches {
    //             self.update_mini_batch(mini_batch);
    //         }
    //     }
    // }

    pub fn feed_forward(&mut self, input_values: &Vec<f64>) {
        // pub fn feed_forward(&mut self, input_layer: Vec<f64>) {
        let weights = self.weights.clone();
        let mut layers = self.layers.clone();
        let biases = self.biases.clone();
        self.layers[0] = input_values.clone();
        for i in 0..layers.len() - 1 {
            let z = weighted_sum(&weights[i], &layers[i], &biases[i]);
            layers[i + 1] = z.iter().map(|product| sigmoid(*product)).collect();
            self.z_values.push(z);
        }
    }

    pub fn back_propagation(&mut self, expected_output: &Vec<f64>) {
        // http://neuralnetworksanddeeplearning.com/chap2.html
        // BP1, delta is da/dz * dC/da (which is sigma'(z) * 2(a-y) in video, and delta used by article 3b1b used)
        // output error of final layer deltaL
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

        let mut example_activations = self.layers.clone().iter().rev();
        let mut example_bias = self.biases.clone().iter().rev();
        let mut example_weight = self.weights.clone().iter().rev();

        example_bias = delta;
        example_weight =

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

// https://gudok.xyz/transpose/
fn transpose(matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
    // from https://stackoverflow.com/questions/64498617/how-to-transpose-a-vector-of-vectors-in-rust
    let len = matrix[0].len();
    let mut matrix_t: Vec<_> = matrix
        .to_owned()
        .into_iter()
        .map(|n| n.into_iter())
        .collect();
    (0..len)
        .map(|_| {
            matrix_t
                .iter_mut()
                .map(|n| n.next().unwrap())
                .collect::<Vec<f64>>()
        })
        .collect()
}

// minibatch element has x, y fields. x is activations  and y is expected output.
// update_mini_batch given minibatch struct and learning rate.
// clone all of the weights and biases from current network, and pass in results from the feed forward
// for each
// i dont think this is even batching??? he runs backprop on each training example
// struct MiniBatch {
//     pub layers: Vec<Vec<Vec<f64>>>,
//     pub weights: Vec<Vec<Vec<Vec<f64>>>>,
// }

// impl MiniBatch {
//     pub fn new(layers: Vec<Vec<Vec<f64>>>, weights: Vec<Vec<Vec<Vec<f64>>>>) -> Self {
//         MiniBatch {
//             layers: Vec::new(),
//             weights: Vec::new(),
//         }
//     }
// }

// nn struct makes base vecs for each minibatch to use during training
// nn.train() groups examples into mini batches, then taken by a thread to do the calculations
// train mini batch has each example borrow the weights, biases, layers, to output a final cost vector
// final cost vector is used in a single backprop using the nn struct's weights, biases, layers

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

    #[test]
    fn test_transpose() {
        let t0: Vec<Vec<f64>> = vec![vec![0.0, 1.0, 2.0], vec![3.0, 4.0, 5.0]];
        assert_eq!(
            transpose(&t0),
            vec![vec![0.0, 3.0], vec![1.0, 4.0], vec![2.0, 5.0]]
        );
    }

    #[test]
    fn reverse() {
        let v = vec![0, 1, 2, 3, 4, 5];
        let len = v.len();
        for i in (1..len - 1).rev() {
            println!("{}", v[i]);
            // Implementation for adjusting weights
        }
    }
}
