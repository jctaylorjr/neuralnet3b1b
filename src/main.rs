use rand::random;

use neuralnet3b1b::z_values;

fn main() {
    // for every image there is going to be a random set of weights and biases for the input layer
    // weights should be random between 0 and 1
    // biases should start at 0 and will be adjusted by asymmetry breaking during back propagation
    // https://ai.stackexchange.com/questions/14292/should-the-biases-be-zero-or-randomly-initialised
    let hidden_layer_count: usize = 2;
    let hidden_layer_neuron_count: usize = 16;

    let mnist_images = vec![]; // TODO: load MNIST images
    let image_count: usize = mnist_images.len();
    let pixel_count: usize = image.height() * image.width();

    let mut starting_weights = vec![vec![rand::random_range(0.0..=1.0); pixel_count]; image_count];
    starting_weights.extend()
    let mut starting_biases = vec![vec![0.0; pixel_count]; image_count];

    let z_values_all = vec![vec![vec![0.0; 5]; 5]; 5];
    for image in mnist_images {
        image.pixels().iter().map(|pixel| pixel as f64 / 255.0)
    }
}
