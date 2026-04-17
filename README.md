# Quantum Experiment Simulation with Crosstalk Noise

## Overview
This project provides a **reproducible** and **Dockerized** environment for running **quantum experiments** with **crosstalk noise modeling** using Qiskit. The experiment allows the user to apply  **error mitigation techniques** like **Pauli Twirling** and **Dynamical Decoupling (DD)** while simulating the effect of **crosstalk noise** on different circuit topologies.

## Features
- **Supports Crosstalk Noise Models**: 
  - `"cxneighbors"`: Noise occurs for 2-qubit gates that share a qubit.
  - `"ncx"`: Noise added based on simultaneous exececution of two two qubit gates.
  - `"topology"`: Noise applied based on close physical qubit proximity.
- **Error Mitigation**:
  - **Pauli Twirling**
  - **Dynamical Decoupling (DD)**
- **Parallelized Execution** using `multiprocessing`
- **Customizable Parameters**: Run experiments with different connectivity densities, optimization levels, and fidelity values.
- **Dockerized for Reproducibility**

---

## Installation

### **Option 1: Run with Docker (Recommended)**
Docker ensures all dependencies and environments are properly set up.

#### **Step 1: Build the Docker Image**
```sh
docker build -t quantum_experiment .
```
#### **Step 2: Run the experiment with default settings or set parameters (use --rm only if you want to delete the container right after the calculations)**
```sh
docker run --rm -it quantum_expeirment .
```

#### **Step 3: Set parameter (guided)**

