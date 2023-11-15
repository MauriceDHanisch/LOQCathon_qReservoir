import numpy as np

from lo_reservoir import PhotonicReservoirSimulator

def test_mode_expectations():
    # Create an instance of the simulator with known parameters
    m = 4  # Smaller number of modes for simplicity
    t = 2  # Number of layers
    simulator = PhotonicReservoirSimulator(m, t, overlapping=True)

    # Set a known parameter matrix
    test_param_matrix = simulator.generate_rndm_param_matrix()
    simulator.set_circuit_parameters(test_param_matrix)

    # Calculate expectations
    expectations = simulator.calculate_mode_expectations()

    # Expected values (replace with actual expected values)
    expected_values = [0.0]*m

    # Assert the length of the expectation values
    assert len(expectations) == len(expected_values), "Length of expectation values mismatch"

    # Assert each expectation value (allow some margin for floating-point comparisons)
    for exp_val, expected_val in zip(expectations, expected_values):
        assert abs(exp_val - expected_val) < 1e-5, "Expectation value mismatch"
