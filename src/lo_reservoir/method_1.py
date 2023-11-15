# Maurice D. Hanisch mhanisc@ethz.ch
# Chrysander D. Hagen chhagen@ethz.ch
# 15.11.2023

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
    def __init__(self, m, t, overlapping=False, noise_mean=0.1, noise_std=0.2, noise_modes=[]):
        """
        Initialize the simulator.

        :param m: The number of modes.
        :param t: The number of time layers.
        :param overlapping: Whether the modes overlap or not.
        :param noise_mean: The mean of the Gaussian noise.
        :param noise_std: The standard deviation of the Gaussian noise.
        :param noise_modes: The modes to which the noise is applied.
        """
        self.m = m
        self.t = t
        self.overlapping = overlapping
        self.circuit = self.create_circuit()
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.noise_modes = noise_modes

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
        for t in range(self.t):
            circuit = circuit.add(0, self.full_layer(t))
        return circuit
    
    def create_circuit_loss(self):
        """
        Creates a circuit with t layers and loss.
        """
        main_circuit = pcvl.Processor(self.m, source=pcvl.Source(emission_probability=.6, multiphoton_component=.01))
        print(self.noise_modes)
        for t in range(t):
            main_circuit = main_circuit.add(0, self.full_layer_loss())
        return main_circuit
    
    def generate_rndm_param_matrix(self):
        """Generates a random parameter matrix of size (t, num_parameters)."""
        num_parameters = len(self.circuit.get_parameters())
        return np.random.rand(self.t, num_parameters//self.t)*2*np.pi # Random dataset of angles

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
        assert len(params) == len(flattened_params), f"Parameter length mismatch. Expected {len(params)} parameters, got {len(flattened_params)}."
        for param, value in zip(params, flattened_params):
            param.set_value(value)

    def calculate_mode_expectations(self, input_state : pcvl.BasicState = None):

        if input_state is None:
            print("WARNING: No input state provided. Using the vacuum state.")
            input_state = pcvl.BasicState([0] * self.m)

        backend = pcvl.BackendFactory.get_backend("SLOS")
        backend.set_circuit(self.circuit)
        backend.set_input_state(input_state)
        prob_distribution = backend.prob_distribution()

        expectations = [0.0 for _ in range(self.m)]
        for state, probability in prob_distribution.items():
            for mode in range(self.m):
                expectations[mode] += state[mode] * probability
        return expectations