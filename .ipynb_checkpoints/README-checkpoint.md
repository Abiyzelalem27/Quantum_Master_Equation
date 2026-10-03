

# Quantum Master Equation

Exploring open quantum systems, quantum noise channels, and the Lindblad master equation through mathematical explanations and numerical simulations.

## Physics overview

An isolated quantum system evolves unitarily. An **open quantum system** interacts with an environment, allowing its reduced state to exhibit decay and decoherence.

![The total system consists of the system of interest and its environment.](images/total_system_diagram.png)

*The total system consists of the system we study and its surrounding environment.*

![Circuit representation of an atom interacting with an environment, followed by a partial trace over the environment.](images/open-system-dynamics.png)

*The atom and environment evolve together. Tracing out the environment gives the reduced density matrix of the atom.*

## Notes on the approximations

The combined system and environment are treated as an isolated system that evolves unitarily. A common microscopic derivation of the Lindblad master equation uses weak system–environment coupling, a short environment memory time, and a secular approximation. The validity of these assumptions depends on the physical model.
