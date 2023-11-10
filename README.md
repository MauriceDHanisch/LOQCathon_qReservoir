# unloqc-prediction_of_energy_consumption_EDF
## Abstract 
It is a challenging problem to predict time-series i.e. a state of a system at a given timestep given its state at the previous timestep. One way to do it is using neural networks, more specifically recurrent neural networks where the output of some node affects subsequent input of the same node.
Reservoir computing is derived from recurrent neural network and input a signal into a higher dimensional space, thanks to the dynamics of a reservoir (fixed, non-linear system). The output of the reservoir is then plainly trained to map the desired output. Reservoir computing is computationally less costly compared to RNNs.
Actual quantum computers are prone to noise and give an access to a great computational space. There are thus ideal candidates for reservoir computing. Lot of works have tried to use a noisy quantum computer as a reservoir. 
Moreover, single-photon Quantum Computing navigate in a computational space that is much bigger than the qubit space. Your challenge, if you accept it, would be to predict time-series provided by EDF using photonic quantum reservoir.
This challenge is proposed in partnership with EDF. Your Quandela mentor will be Samuel Horsch. Please commit your code in the repository so we can keep track of your contribution. This will be part of the evaluation afterwards.

## Description of natural quantum reservoir process on a superconducting quantum computer

The state of the Quantum Reservoir is evolving according to :

$$\begin{aligned} \rho_t & =\mathscr{T}_{u_t}\left(\rho_{t-1}\right) \\ & =\mathscr{E}_{\text {device }}\left(U\left(u_t\right) \rho_{t-1} U\left(u_t\right)^{\dagger}\right)\end{aligned}$$
