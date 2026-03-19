// fn dot_product(a: &[f64], b: &[f64]) -> f64 {
//     a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
// }

// fn matrix_vector_multiply(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
//     a.iter().map(|row| dot_product(row, b)).collect()
// }

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + std::f64::consts::E.powf(x))
}

fn feed_forward(a: &[Vec<f64>], b: &[f64], bias: &[f64]) -> Vec<f64> {
    a.iter()
        .map(|row: &Vec<f64>| row.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f64>())
        .zip(bias.iter())
        .map(|(product, bias)| sigmoid(product + bias))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feed_forward() {
        let a = vec![
            vec![1.0, 2.0, 3.0],
            vec![1.0, 2.0, 3.0],
            vec![1.0, 2.0, 3.0],
        ];
        let b = vec![1.0, 2.0, 3.0];
        let bias = vec![0.95, 0.5, 0.23];
        let expected = vec![
            sigmoid(14.0 + 0.95), // 1*1 + 2*2 + 3*3 + 0.95
            sigmoid(14.0 + 0.5),  // 1*1 + 2*2 + 3*3 + 0.5
            sigmoid(14.0 + 0.23), // 1*1 + 2*2 + 3*3 + 0.23
        ];
        assert_eq!(feed_forward(&a, &b, &bias), expected);
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
