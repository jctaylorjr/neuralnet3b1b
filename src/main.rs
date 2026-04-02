use neuralnet3b1b::NeuralNetwork;

fn main() {
    // for every image there is going to be a random set of weights and biases for the input layer
    // weights should be random between 0 and 1
    // biases should start at 0 and will be adjusted by asymmetry breaking during back propagation
    // https://ai.stackexchange.com/questions/14292/should-the-biases-be-zero-or-randomly-initialised
    let mut nn = NeuralNetwork::new(
        2,
        16,
        10,
        vec![1.0; 784], // TODO: load MNIST image and convert to input layer
    );

    let mut i = 0;
    for x in nn.weights.iter() {
        for y in x.iter() {
            for z in y.iter() {
                i += 1;
                println!("{}", z);
            }
        }
    }

    println!("Total weights: {}", i);

    nn.feed_forward();
    for layer in nn.layers {
        println!("Length: {}\nValues: {:?}", layer.len(), layer);
    }
    for activation in nn.activation_values {
        println!("Length: {}\nValues: {:?}", activation.len(), activation);
    }
    for weight in nn.weights {
        println!("Length: {}\nValues: {:?}", weight.len(), weight);
    }
}
