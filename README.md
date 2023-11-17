# LOQCathon2.0 — Challenge EDF — Team 1
# Quantum Linear Optics Reservoir Computing

## Team members

- **Mentor:** Samuel HORSH

- Chrysander HAGEN
- Maurice HANISCH
- Vinicius MOHR
- Samuel REY
- Clément RONFAUT

## Project summary

It is a challenging problem to predict time series.
One way to do it is using neural networks, more specifically recurrent neural networks where the output of some node affects subsequent input of the same node.

Reservoir computing is derived from recurrent neural networks and input a signal into a higher dimensional space, thanks to the dynamics of a reservoir (fixed, non-linear system).
The output of the reservoir is then plainly trained to map the desired output.
Reservoir computing is computationally less costly compared to RNNs.

Actual quantum computers are prone to noise and give an access to a great computational space.
There are thus ideal candidates for reservoir computing.
Lot of works have tried to use a noisy quantum computer as a reservoir.

Moreover, single-photon quantum computing navigate in a computational space that is much bigger than the qubit space.
**The challenge is to predict a time series provided by EDF using a linear optics quantum reservoir.**

## Installation

*Python version must be 3.10.12*

*Open a terminal in the repository folder*

```
pip install pipenv
pipenv install
```

*Select the right kernel in your IDE*

## Description of natural quantum reservoir process on a superconducting quantum computer [1]

The state of the quantum reservoir is evolving according to:

<p align="center"><img src="https://github.com/LOQCathon2-0/unloqc-EDF-1/blob/main/Images/eq_1_EDF.png" width="330" height="30" /></p>

Where $u_t$ are some scalar inputs. $\rho_t$ is the state of the system at time step $t$ and $U(u_t)$ an input dependent unitary.
$\mathscr{E}_{\text{device}}$ is a CPTP (completely positive trace preserving) map that correspond to the real device during operation.
In [1] $U(u_t)$ is defined like:

<p align="center"><img src="https://github.com/LOQCathon2-0/unloqc-EDF-1/blob/main/Images/eq_2_EDF.png" width="450" height="30" /></p>

A tensor product of 2-qubit unitaries, defined as:

<p align="center"><img src="https://github.com/LOQCathon2-0/unloqc-EDF-1/blob/main/Images/eq_3_EDF.png" width="395" height="30" /></p>

$S_{u_t} = a_{u_t}$ with $a \in R$.
This has been used for a gate-based QC, perhaps more adapted unitaries shall be used with a single-photon QC.
The initial state is an equal superposition of all states.
We repeatedly apply the above sequence:

<p align="center"><img src="https://github.com/LOQCathon2-0/unloqc-EDF-1/blob/main/Images/eq_4_EDF.png" width="312" height="30" /></p>


and measure in the computational basis:


<p align="center"><img src="https://github.com/LOQCathon2-0/unloqc-EDF-1/blob/main/Images/eq_5_EDF.png" width="283" height="30" /></p>

to get the signal:

<p align="center"><img src="https://github.com/LOQCathon2-0/unloqc-EDF-1/blob/main/Images/eq_6_EDF.png" width="101" height="30" /></p>

Where,

<p align="center"><img src="https://github.com/LOQCathon2-0/unloqc-EDF-1/blob/main/Images/eq_7_EDF.png" width="322" height="30" /></p>

that are then fitted by tuning $W_{\text{out}}$, in a supervised fashion using mean-squared error with the target output ($y_t$):

<p align="center"><img src="https://github.com/LOQCathon2-0/unloqc-EDF-1/blob/main/Images/eq_8_EDF.png" width="196" height="40" /></p>

## Goals of the challenge
Implement a linear optics quantum reservoir computing model that can lead to reduced processing time and energy consumption
- Pick a unitary that is more adapted to a photonic quantum circuit.
- Check if your photonic reservoir enables to predict the NARMA sequence.
- Check if your photonic reservoir enables to predict the EDF sequence.

## About EDF :
At EDF, our raison d’être is to build a net zero energy future with electricity and innovative solutions and services, to help save the planet and drive wellbeing and economic development.
Wherever our group operates, we want to invent a new energy model to address the climate crisis: lower-carbon, more efficient, less of an impact on the environment and on people.
To serve these Goal, in 2022, EDF invested €649 million in research and development in three areas of research: energy transition, climate transition, and digital and societal transition

<img src="https://github.com/LOQCathon2-0/unloqc-EDF-1/blob/main/Images/EDG_logo.png" width="705" height="300"/>

## References

[1] Suzuki, Y., Gao, Q., Pradel, K.C. et al. — Natural quantum reservoir computing for temporal information processing. (2022). https://doi.org/10.1038/s41598-022-05061-w (https://arxiv.org/pdf/2107.05808.pdf) <br>
[2] Fujii, K., Nakajima, K. — Harnessing disordered quantum dynamics for machine learning. (2017). https://doi.org/10.1103/PhysRevApplied.8.024030 (http://arxiv.org/abs/1602.08159) <br>
[3] Cucchi, M. et al. — Hands-on reservoir computing: a tutorial for practical implementation. (2022). https://doi.org/10.1088/2634-4386/ac7db7 <br>
[4] Nakajima, K. — Physical reservoir computing—an introductory perspective. (2020). https://doi.org/10.35848/1347-4065/ab8d4f <br>
