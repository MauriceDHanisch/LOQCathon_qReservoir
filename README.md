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


## Description of natural quantum reservoir process on a superconducting quantum computer [1]

The state of the Quantum Reservoir is evolving according to :

<center><img src="https://github.com/LOQCathon2-0/unloqc-prediction_of_energy_consumption_EDF/blob/main/Images/eq_1_EDF.png" width="330" height="30" /></center>

Where $u_t$ are some scalar inputs. $\rho_t$ is the state of the system at time step $t$ and $U(u_t)$ an input dependent unitary. $\mathscr{E}_{\text{device}}$ is a CPTP map that correspond to the real device during operation. $U(u_t)$ is define like :

<center><img src="https://github.com/LOQCathon2-0/unloqc-prediction_of_energy_consumption_EDF/blob/main/Images/eq_2_EDF.png" width="450" height="30" /></center>

A tensor product of 2-qubit unitaries, defined as :

<center><img src="https://github.com/LOQCathon2-0/unloqc-prediction_of_energy_consumption_EDF/blob/main/Images/eq_3_EDF.png" width="395" height="30" /></center>

$S_{u_t} = a_{u_t}$ with $a \in R$. (side remark this has been used for a gate-based QC, perhaps more adapted unitaries shall be used with a single-photon QC). The initial state is an equal superposition of all states. We repeatedly apply the above sequence:

<center><img src="https://github.com/LOQCathon2-0/unloqc-prediction_of_energy_consumption_EDF/blob/main/Images/eq_4_EDF.png" width="312" height="30" /></center>


and measure in the computational basis:


<center><img src="https://github.com/LOQCathon2-0/unloqc-prediction_of_energy_consumption_EDF/blob/main/Images/eq_5_EDF.png" width="283" height="30" /></center>

To get the signal :

<center><img src="https://github.com/LOQCathon2-0/unloqc-prediction_of_energy_consumption_EDF/blob/main/Images/eq_6_EDF.png" width="101" height="30" /></center>

Where,

<center><img src="https://github.com/LOQCathon2-0/unloqc-prediction_of_energy_consumption_EDF/blob/main/Images/eq_7_EDF.png" width="322" height="30" /></center>

that are then fitted by tuning $W_{\text{out}}$, in a supervised fashion using mean-squared error with the target output ($y_t$):

<center><img src="https://github.com/LOQCathon2-0/unloqc-prediction_of_energy_consumption_EDF/blob/main/Images/eq_8_EDF.png" width="147" height="30" /></center>

## Goals of the challenge
As mentioned in the abstract, the goal of the challenge is to implement a photonic quantum reservoir to predict time-series. You may :\
•	Pick a unitary that is more adapted to a photonic QC.\
•	Check if your photonic reservoir enables to predict the NARMA sequence.\
•	Check if your photonic reservoir enables to predict the EDF sequence.\
•	Benchmark your results by computing the mean and standard deviation of the overall sequence with the results from REF  , for the training and testing sequences.

## About EDF :
At EDF, our raison d’être is to build a net zero energy future with electricity and innovative solutions and services, to help save the planet and drive wellbeing and economic development. Wherever our Group operates, we want to invent a new energy model to address the climate crisis: lower-carbon, more efficient, less of an impact on the environment and on people.
To serve these Goal, in 2022, EDF invested €649 million in research and development in three areas of research: energy transition, climate transition, and digital and societal transition

![image info](./Images/logo_EDF.png)


## References

[1] Suzuki, Y., Gao, Q., Pradel, K.C. et al. Natural quantum reservoir computing for temporal information processing. Sci Rep 12, 1353 (2022). https://doi.org/10.1038/s41598-022-05061-w




