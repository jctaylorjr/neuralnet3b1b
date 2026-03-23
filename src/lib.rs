// fn dot_product(a: &[f64], b: &[f64]) -> f64 {
//     a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
// }

// fn matrix_vector_multiply(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
//     a.iter().map(|row| dot_product(row, b)).collect()
// }

fn sigmoid_derivative(x: f64) -> f64 {
    sigmoid(x) * (1.0 - sigmoid(x))
}

fn sigmoid(x: f64) -> f64 {
    // https://calculus.subwiki.org/wiki/Logistic_function
    1.0 / (1.0 + std::f64::consts::E.powf(x))
}

fn feed_forward(weights: &[Vec<f64>], preceding_layer: &[f64], biases: &[f64]) -> Vec<f64> {
    // activations = sigmoid(summation of all weights * preceding_layer neurons + biases)
    weights
        .iter()
        .map(|row: &Vec<f64>| {
            row.iter()
                .zip(preceding_layer.iter())
                .map(|(x, y)| x * y)
                .sum::<f64>()
        })
        .zip(biases.iter())
        .map(|(product, bias)| sigmoid(product + bias))
        .collect()
}

fn z_values(weights: &[Vec<f64>], preceding_layer: &[f64], biases: &[f64]) -> Vec<f64> {
    // preactivation values (z^l) = summation of weights * preceding_layer neurons + biases
    weights
        .iter()
        .map(|row: &Vec<f64>| {
            row.iter()
                .zip(preceding_layer.iter())
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
        assert_eq!(feed_forward(&weights, &input_layer, &bias), expected);
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
