# unloqc-prediction_of_energy_consumption_EDF
## Abstract 
It is a challenging problem to predict time-series i.e. a state of a system at a given timestep given its state at the previous timestep. One way to do it is using neural networks, more specifically recurrent neural networks where the output of some node affects subsequent input of the same node.
Reservoir computing is derived from recurrent neural network and input a signal into a higher dimensional space, thanks to the dynamics of a reservoir (fixed, non-linear system). The output of the reservoir is then plainly trained to map the desired output. Reservoir computing is computationally less costly compared to RNNs.
Actual quantum computers are prone to noise and give an access to a great computational space. There are thus ideal candidates for reservoir computing. Lot of works have tried to use a noisy quantum computer as a reservoir. 
Moreover, single-photon Quantum Computing navigate in a computational space that is much bigger than the qubit space. Your challenge, if you accept it, would be to predict time-series provided by EDF using photonic quantum reservoir.
This challenge is proposed in partnership with EDF. Your Quandela mentor will be Samuel Horsch. Please commit your code in the repository so we can keep track of your contribution. This will be part of the evaluation afterwards.

## Why it is important for EDF ?

The project focuses on the forecasting of renewable energy (RE) production. Robust and accurate forecasting models need to be developed if these energies are to be efficiently integrated into the grid and the use of fossil fuels is to be reduced. This is a highly complex task as renewables are highly dependent on meteorological conditions. 
Furthermore, these models, which are based on deep learning methods, require large data sets and can be very energy-intensive: one of the aims of the project is to develop models that consume less data and energy. In particular, we propose an approach based on Quantum Reservoir Computing, a promising method for processing time series.

## Reservoir computing (RC)

As mentioned above reservoir computing offeres a less computing intensive way to replace recurrent neural networks (RNNs). The idea of reservoir computing is to map the input state into a higher deimensional space and then train an output function for a given task, such as time series prediction or classification. The reservoir is this intermediate higher dimentional space where the input is mapped into. The advantage of such a setup is that one can keep the reservoir fixed an has to only train the output function. This can be done via a simple linear regression.[3] gives some further introduction to reservoir computing. This reservoir can in theory be anything as shown in [4]. Specifically it can be a quantum system. [1] showed how to use a superconducting quantum computer as such a reservoir. In [2] you can find a more complete description of quantum reservoir computing.

## Description of natural quantum reservoir process on a superconducting quantum computer [1]

The state of the Quantum Reservoir is evolving according to :

<p align="center"><img src="https://github.com/LOQCathon2-0/unloqc-prediction_of_energy_consumption_EDF/blob/main/Images/eq_1_EDF.png" width="330" height="30" /></p>

Where $u_t$ are some scalar inputs. $\rho_t$ is the state of the system at time step $t$ and $U(u_t)$ an input dependent unitary. $\mathscr{E}_{\text{device}}$ is a CPTP (completely positive trace preserving) map that correspond to the real device during operation. In [1] $U(u_t)$ is defined like:

<p align="center"><img src="https://github.com/LOQCathon2-0/unloqc-prediction_of_energy_consumption_EDF/blob/main/Images/eq_2_EDF.png" width="450" height="30" /></p>

A tensor product of 2-qubit unitaries, defined as :

<p align="center"><img src="https://github.com/LOQCathon2-0/unloqc-prediction_of_energy_consumption_EDF/blob/main/Images/eq_3_EDF.png" width="395" height="30" /></p>

$S_{u_t} = a_{u_t}$ with $a \in R$. This has been used for a gate-based QC, perhaps more adapted unitaries shall be used with a single-photon QC. The initial state is an equal superposition of all states. We repeatedly apply the above sequence:

<p align="center"><img src="https://github.com/LOQCathon2-0/unloqc-prediction_of_energy_consumption_EDF/blob/main/Images/eq_4_EDF.png" width="312" height="30" /></p>


and measure in the computational basis:


<p align="center"><img src="https://github.com/LOQCathon2-0/unloqc-prediction_of_energy_consumption_EDF/blob/main/Images/eq_5_EDF.png" width="283" height="30" /></p>

To get the signal :

<p align="center"><img src="https://github.com/LOQCathon2-0/unloqc-prediction_of_energy_consumption_EDF/blob/main/Images/eq_6_EDF.png" width="101" height="30" /></p>

Where,

<p align="center"><img src="https://github.com/LOQCathon2-0/unloqc-prediction_of_energy_consumption_EDF/blob/main/Images/eq_7_EDF.png" width="322" height="30" /></p>

that are then fitted by tuning $W_{\text{out}}$, in a supervised fashion using mean-squared error with the target output ($y_t$):

<p align="center"><img src="https://github.com/LOQCathon2-0/unloqc-prediction_of_energy_consumption_EDF/blob/main/Images/eq_8_EDF.png" width="196" height="40" /></p>

## Goals of the challenge
As mentioned in the abstract, the goal of the challenge is to implement a photonic quantum reservoir to predict time-series. You may :\
•	Pick a unitary that is more adapted to a photonic QC.\
•	Check if your photonic reservoir enables to predict the NARMA sequence.\
•	Check if your photonic reservoir enables to predict the EDF sequence.\
•	Benchmark your results by computing the mean and standard deviation of the overall sequence with the results from [1], for the training and testing sequences.

## About EDF :
At EDF, our raison d’être is to build a net zero energy future with electricity and innovative solutions and services, to help save the planet and drive wellbeing and economic development. Wherever our Group operates, we want to invent a new energy model to address the climate crisis: lower-carbon, more efficient, less of an impact on the environment and on people.
To serve these Goal, in 2022, EDF invested €649 million in research and development in three areas of research: energy transition, climate transition, and digital and societal transition

<img src="https://github.com/LOQCathon2-0/unloqc-prediction_of_energy_consumption_EDF/blob/main/Images/EDG_logo.png" width="705" height="300"/>



## References

[1] Suzuki, Y., Gao, Q., Pradel, K.C. et al. Natural quantum reservoir computing for temporal information processing. Sci Rep **12**, 1353 (2022). https://doi.org/10.1038/s41598-022-05061-w (https://arxiv.org/pdf/2107.05808.pdf) <br>
[2] Fujii, K., Nakajima, K. Harnessing disordered quantum dynamics for machine learning. Phys. Rev. Applied **8**, 024030 (2017). https://doi.org/10.1103/PhysRevApplied.8.024030 (http://arxiv.org/abs/1602.08159) <br>
[3] Cucchi, M. et al. Hands-on reservoir computing: a tutorial for practical implementation. Neuromorph. Comput. Eng. **2**, 032002 (2022). https://doi.org/10.1088/2634-4386/ac7db7 <br>
[4] Nakajima, K. Physical reservoir computing—an introductory perspective. Jpn. J. Appl. Phys. **59**, 060501 (2020). https://doi.org/10.35848/1347-4065/ab8d4f <br>
