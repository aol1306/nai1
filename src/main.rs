extern crate rand;

use rand::Rng;

#[derive(Debug)]
struct Neuron {
    weights: Vec<f64>,
    bias: f64,
}

impl Neuron {
    fn activate(&self, x: f64) -> f64 {
        // binary step activation function
        match x {
            x if x >= 0.0 => 1.0,
            _ => 0.0,
        }
    }

    fn calculate_value(&self, x: &Vec<f64>) -> f64 {
        if x.len() != self.weights.len() {
            panic!("Data length is not equal to count of neuron inputs");
        }
        let mut output = 0.0;
        for i in 0..x.len() {
            output += x[i] * self.weights[i];
        }
        output += self.bias;
        //output -= self.bias; // alternative version with -1.0 constant input
        return self.activate(output);
    }

    fn delta(&self, expected: f64, y: f64, x: &Vec<f64>, alpha: f64) -> Vec<f64> {
        let mut delta: Vec<f64> = vec![];
        for i in 0..x.len() {
            delta.push(alpha * (expected - y) * x[i])
        }
        return delta;
    }
}

#[derive(Debug)]
struct Network {
    neurons: Vec<Neuron>,
    alpha: f64,
}

impl Network {
    fn create(inputs: usize, neuron_count: usize, alpha: f64) -> Network {
        println!(
            "Initialize network with {} inputs and {} neurons",
            inputs, neuron_count
        );

        let mut rng = rand::thread_rng();

        let mut neurons = vec![];
        for _ in 0..neuron_count {
            let mut neuron = Neuron {
                weights: Vec::with_capacity(inputs),
                bias: rng.gen_range(-10.0, 10.0),
            };
            for _ in 0..inputs {
                neuron.weights.push(rng.gen_range(-10.0, 10.0))
            }
            neurons.push(neuron);
        }
        Network {
            neurons: neurons,
            alpha: alpha,
        }
    }

    fn calculate(&self, x: &Vec<f64>) -> Vec<f64> {
        let mut ret: Vec<f64> = vec![];
        for i in 0..self.neurons.len() {
            ret.push(self.neurons[i].calculate_value(&x))
        }
        return ret;
    }

    fn calculate_new_weights(&mut self, expected: &Vec<f64>, real: &Vec<f64>, x: &Vec<f64>) {
        let mut deltas = vec![];
        for i in 0..self.neurons.len() {
            deltas.push(self.neurons[i].delta(expected[i], real[i], &x, self.alpha));
        }
        //println!("\tDeltas: {:?}", deltas);
        // all deltas calculated => recalculate weights and bias
        let mut i = 0;
        for neuron in self.neurons.iter_mut() {
            let mut j = 0;
            //println!("\tNeuron {} before learning:\t{:?}", i, neuron);
            for w in neuron.weights.iter_mut() {
                *w += deltas[i][j];
                j += 1;
            }

            neuron.bias += self.alpha * (expected[i] - real[i]);
            //neuron.bias -= 1.0 * (expected[i] - real[i]); // alternative version with -1.0 const input
            //println!("\tNeuron {} after learning:\t{:?}", i, neuron);
            i += 1;
        }
    }
}

struct TrainingData {
    inputs: Vec<f64>,
    expected: Vec<f64>,
}

fn plot_error_log(error_log : Vec<f64>) {
    println!("Error chart:");
    for (i, error) in error_log.iter().enumerate() {
        print!("{:02}: ", i);
        for _ in 0..(*error*2.0) as u64 {
            print!("*");
        }
        println!("\t\t\t\t{}", error);
    }
}

fn main() {
    let mut net = Network::create(24, 3, 0.5);

    let training_set = vec![
        TrainingData {
            inputs: vec![
                0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0,
                1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0,
            ],
            expected: vec![1.0, 0.0, 0.0],
        },
        TrainingData {
            inputs: vec![
                0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            ],
            expected: vec![0.0, 1.0, 0.0],
        },
        TrainingData {
            inputs: vec![
                1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0,
            ],
            expected: vec![0.0, 0.0, 1.0],
        },
    ];

    let mut error_log = vec![];

    for i in 1..100 {
        println!("*** Epoch {} ***", i);
        println!("Calculate all Ys (network outputs for all X)");
        println!("Recalculate weights for all inputs, for all neurons");
        let mut epoch_error = 0.0;
        for training_entry in training_set.iter() {
            let calculated = net.calculate(&training_entry.inputs);
            net.calculate_new_weights(
                &training_entry.expected,
                &calculated,
                &training_entry.inputs,
            );

            let mut entry_error = 0.0;
            for k in 0..training_entry.expected.len() {
                entry_error += (training_entry.expected[k] - calculated[k]).powi(2);
            }
            epoch_error += entry_error;
        }
        epoch_error /= 2.0;
        error_log.push(epoch_error);
        println!("Epoch error: {}", epoch_error);
        println!();

        if epoch_error == 0.0 {
            println!("* Error reached 0, stop learning.");
            println!();
            break;
        }
    }

    // check results
    println!(
        "Zero, training set:\t{:?}",
        net.calculate(&training_set[0].inputs)
    );
    println!(
        "Zero, 2 filled corners:\t{:?}",
        net.calculate(&vec![
            0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
            0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0
        ])
    );
    println!(
        "Zero, full corners:\t{:?}",
        net.calculate(&vec![
            1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
            0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0
        ])
    );
    println!();
    println!(
        "One, training set:\t{:?}",
        net.calculate(&training_set[1].inputs)
    );
    println!(
        "One, with dash: \t{:?}",
        net.calculate(&vec![
            0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        ])
    );
    println!(
        "One, with noise:\t{:?}",
        net.calculate(&vec![
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        ])
    );
    println!();
    println!(
        "Two, training set:\t{:?}",
        net.calculate(&training_set[2].inputs)
    );
    println!(
        "Two, with noise:\t{:?}",
        net.calculate(&vec![
            1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
            0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0,
        ])
    );

    println!();
    plot_error_log(error_log);
}
