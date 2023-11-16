# Maurice D. Hanisch mhanisc@ethz.ch
# Chrysander D. Hagen chhagen@ethz.ch
# 15.11.2023

from tqdm import tqdm

import perceval as pcvl
import numpy as np


def generate_gaussian_0_1(mean, std_dev, num_samples):
    """
    Generate Gaussian samples on the interval [0,1].

    :param mean: The mean of the Gaussian distribution.
    :param std_dev: The standard deviation of the Gaussian distribution.
    :param num_samples: The number of samples to generate.
    """
    samples = []
    while len(samples) < num_samples:
        sample = np.random.normal(mean, std_dev)
        if sample >= 0 and sample <= 1:
            samples.append(sample)
    return np.array(samples)

class PhotonicReservoirSimulator:
    def __init__(self, m, t_max, overlapping=False):
        self.m = m  # Number of modes
        self.t_max = t_max  # Number of time layerse
        self.overlapping = overlapping
        self.layers = []  # Stores generated layers

    def generate_and_store_layers(self):
        """Generate and store layers up to max_t."""
        self.layers = [self.full_layer(t) for t in range(self.t_max)]

    def set_circuit_with_stored_layers(self, num_layers=None):
        """Creates a circuit using the first num_layers from stored layers."""

        if num_layers is None:
            num_layers = self.t_max

        # Generate additional layers if needed
        if num_layers > len(self.layers):
            missing_layers = num_layers - len(self.layers)
            print(f"WARNING: Requested more layers than generated. Generating {missing_layers} more layers...")
            for t in tqdm(range(len(self.layers), len(self.layers) + missing_layers), desc="Generating layers"):
                self.layers.append(self.full_layer(t))

        # Create the circuit using the stored layers
        circuit = pcvl.Circuit(self.m)
        for layer in self.layers[:num_layers]:
            circuit = circuit.add(0, layer)
        #print("setting self.circuit...")
        self.circuit = circuit

    def set_circuit_with_memory(self, memory_length, num_layers):
        """Creates a circuit using layers starting from num_layer - memory up to num_layer."""

        # Adjust starting point based on memory
        start_layer = max(num_layers - memory_length, 0)

        # Generate additional layers if needed
        if num_layers > len(self.layers):
            missing_layers = num_layers - len(self.layers)
            print(f"WARNING: Requested more layers than generated. Generating {missing_layers} more layers...")
            for t in tqdm(range(len(self.layers), num_layers), desc="Generating layers"):
                self.layers.append(self.full_layer(t))

        # Create the circuit using the specified range of stored layers
        circuit = pcvl.Circuit(self.m)
        for layer in self.layers[start_layer:num_layers]:
            circuit = circuit.add(0, layer)

        self.circuit = circuit

    def U_ij_t(self, i, j, t=0):
        """
        Returns the unitary acting on mode i and j.
        
        :param i: The first mode.
        :param j: The second mode.
        :param t: The time layer, defaults to 0.
        """
        return (pcvl.Circuit(2, name=f"U_{i}{j}_t{t}")
                .add(0, pcvl.PS(phi=pcvl.P(f"phi_tl_{i}{j}_t{t}")))
                .add(1, pcvl.PS(phi=pcvl.P(f"phi_bl_{i}{j}_t{t}")))
                .add(0, pcvl.BS(theta=pcvl.P(f'theta_{i}{j}_t{t}')))
                .add(0, pcvl.PS(phi=pcvl.P(f"phi_tr_{i}{j}_t{t}")))
                .add(1, pcvl.PS(phi=pcvl.P(f"phi_br_{i}{j}_t{t}"))))

    def full_layer(self, t=0):
        """Generate one full time layer"""
        layer = pcvl.Circuit(self.m)
        if self.overlapping:
            for i in range(self.m - 1):
                layer = layer.add(i, self.U_ij_t(i, (i + 1), t))
        else:
            if self.m % 2 != 0:
                print("WARNING: Number of modes is not even; last mode will be ignored.")
            for i in range(self.m // 2):
                layer = layer.add(2 * i, self.U_ij_t(2 * i, 2 * i + 1, t))
        return layer
    
    def full_layer_loss(self):
        """
        Generates a full layer with loss.
        """
        layer = pcvl.Processor("SLOS", self.m, source=pcvl.Source(emission_probability=.6, multiphoton_component=.01))
        if self.overlapping:
            for i in range(self.m-1):
                layer = layer.add(i, self.U_ij_t(i, (i+1), self.t))
        else:
            if self.m % 2 != 0:
                print("WARNING: The number is not even and _overlapping is False => the last mode will be ignored.")
            for i in range((self.m)//2):
                layer = layer.add(2*i, self.U_ij_t(i, i+1, self.t))
        if len(self.noise_modes) != 0:
            random_noise = generate_gaussian_0_1(self.noise_mean, self.noise_std, len(self.noise_modes))
            for i in range(len(self.noise_modes)):
                layer.add(int(self.noise_modes[i]), pcvl.LC(random_noise[i]), keep_port=True)
        return layer

    def create_circuit(self):
        """
        Creates a circuit with t layers.
        """
        circuit = pcvl.Circuit(self.m)
        for t in range(self.t_max):
            circuit = circuit.add(0, self.full_layer(t))
        return circuit
    
    def create_circuit_loss(self):
        """
        Creates a processor with t layers and loss.
        """
        main_circuit = pcvl.Processor(self.m, source=pcvl.Source(emission_probability=.6, multiphoton_component=.01))
        print(self.noise_modes)
        for t in range(t):
            main_circuit = main_circuit.add(0, self.full_layer_loss())
        return main_circuit
    

    def generate_rndm_param_matrix(self, num_layers=None):
        """Generates a random parameter matrix of size (t, num_parameters)."""
        if num_layers is None:
            print("WARNING: No number of layers provided for rndm data matrix. Using t_max.")
            num_layers = self.t_max
        num_parameters = len(self.circuit.get_parameters())
        # Random dataset of angles
        return np.random.rand(num_layers, num_parameters//num_layers)*2*np.pi

    def set_circuit_parameters(self, parameter_matrix):
        """
        Set the parameters of the circuit to the values in the matrix.
        
        Args:
            parameter_matrix (np.ndarray): A matrix of size (t, num_parameters).

        Description:
            The parameter matrix is flattened to match the list of parameters.
            The parameters are set in the order they appear in the circuit.
        """
        flattened_params = parameter_matrix.flatten()
        params = self.circuit.get_parameters()
        assert len(params) == len(
            flattened_params), f"Parameter length mismatch. Expected {len(params)} parameters, got {len(flattened_params)}."
        for param, value in zip(params, flattened_params):
            param.set_value(value)

    def calculate_mode_expectations(self, input_state : pcvl.BasicState = None):
        """
        Calculate the mode expectations of the circuit.

        :param input_state (pcvl.BasicState): The input state, defaults to the vacuum state.
        """
        if input_state is None:
            print("WARNING: No input state provided. Using the vacuum state.")
            input_state = pcvl.BasicState([0] * self.m)

        backend = pcvl.BackendFactory.get_backend("SLOS")
        backend.set_circuit(self.circuit)
        backend.set_input_state(input_state)
        prob_distribution = backend.prob_distribution()

        # source = pcvl.Source(1, 0, 1, 0)
        # processor = pcvl.Processor("SLOS", self.circuit, source=source)      
        # processor.min_detected_photons_filter(0)
        # processor.with_input(input_state)
        # sampler = pcvl.algorithm.Sampler(processor)
        # prob_distribution = sampler.probs()["results"]


        expectations = [0.0 for _ in range(self.m)]
        for state, probability in prob_distribution.items():
            for mode in range(self.m):
                expectations[mode] += state[mode] * probability
        return expectations

    def sequential_expectation_calculation(self, data, input_state):
        # Initialize an empty matrix to store expectation values
        expectations_matrix = []

        for t in tqdm(range(self.t_max), desc="Processing time steps"):
            # Generate a circuit with num_layers = t
            self.set_circuit_with_stored_layers(num_layers=t)

            # Take the first t rows of the data matrix
            params_subset = data[:t]

            # Set the parameters of the circuit
            self.set_circuit_parameters(params_subset)

            # Evaluate the expectation values
            expectation_values = self.calculate_mode_expectations(input_state)

            # Append the expectation values to the matrix
            expectations_matrix.append(expectation_values)

        return np.array(expectations_matrix)
    

    def sequential_expectation_with_memory(self, data, input_state, memory_length):
        """Calculates sequential expectation values using a specified number of memory layers."""

        # Initialize an empty matrix to store expectation values
        expectations_matrix = []

        for t in tqdm(range(self.t_max), desc="Processing time steps"):
            # Generate a circuit with memory layers starting from t - memory
            self.set_circuit_with_memory(memory_length=memory_length, num_layers=t)

            # Take the latest 'memory' rows of the data matrix up to the current time step t
            # Ensure we don't go below index 0
            start_index = max(0, t - memory_length)
            params_subset = data[start_index:t]

            # Set the parameters of the circuit
            self.set_circuit_parameters(params_subset)

            # Evaluate the expectation values
            expectation_values = self.calculate_mode_expectations(input_state)

            # Append the expectation values to the matrix
            expectations_matrix.append(expectation_values)

        return np.array(expectations_matrix)


    def sequential_expectation_with_memory_and_decay(self, data, input_state, memory_length, decay_rate=None):
        """Calculates sequential expectation values using a specified number of memory layers and a decay parameter."""

        # Initialize an empty matrix to store expectation values
        expectations_matrix = []

        for t in tqdm(range(self.t_max), desc="Processing time steps"):
            # Generate a circuit with memory layers starting from t - memory
            self.set_circuit_with_memory(memory_length=memory_length, num_layers=t)

            # Take the latest 'memory' rows of the data matrix up to the current time step t
            # Ensure we don't go below index 0
            start_index = max(0, t - memory_length)
            params_subset = data[start_index:t]

            if decay_rate is not None:
                # Apply decay to the parameters subset
                for i in range(len(params_subset)):
                    decay_factor = (1 / (decay_rate * (len(params_subset) - i) + 1))
                    params_subset[i] *= decay_factor

            # Set the parameters of the circuit
            self.set_circuit_parameters(params_subset)

            # Evaluate the expectation values
            expectation_values = self.calculate_mode_expectations(input_state)

            # Append the expectation values to the matrix
            expectations_matrix.append(expectation_values)

        return np.array(expectations_matrix)
