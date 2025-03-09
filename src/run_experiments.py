from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, pauli_error, QuantumError
from qiskit.transpiler.passes import DynamicalDecoupling
from noise_models import create_crosstalk_noise_from_circuit, create_crosstalk_noise_from_proximity, \
    create_crosstalk_noise_from_layers
from topology_functions import *
from qiskit.circuit.library import IGate, XGate, YGate, ZGate
import random
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler.passes import ALAPSchedule
from qiskit.transpiler import InstructionDurations
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.providers.fake_provider import Fake127QPulseV1
import sys, os
import logging
import csv
import time
import multiprocessing
from multiprocessing import Manager, Pool, cpu_count
import argparse


class RunExperiments:
    def __init__(self,
                 filename,
                 backend_,
                 crosstalk_version,
                 crosstalk_fidelity,
                 neighbor_fidelity,
                 connectivity_density,
                 optimization_level,
                 coupling_map,
                 directory,
                 gate_set):
        self.filename = filename
        self.backend_ = backend_
        self.crosstalk_version = crosstalk_version
        self.crosstalk_fidelity = crosstalk_fidelity
        self.neighbor_fidelity = neighbor_fidelity
        self.connectivity_density = connectivity_density
        self.optimization_level = optimization_level
        self.coupling_map = coupling_map
        self.directory = directory
        self.gate_set = gate_set

        if connectivity_density is None:
            self.connectivity_density = [0.013895, 0.015, 0.018, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.1, 0.15, 0.2, 0.25,
                                         0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            self.connectivity_density =  [0.01]
        if gate_set is None:
            self.gate_set = ['id', 'rz', 'sx', 'x', 'cx', 'swap', 'cz']


    def load_and_prepare_circuit_measure(self, quantum_circuit):

        # Check if the circuit has any measurement gates
        if not any(inst[0].name == "measure" for inst in quantum_circuit.data):
            # Add measurements to all qubits if none exist
            quantum_circuit.measure_all()
            print("Measurement gates added to the circuit.")
        else:
            print("Circuit already contains measurement gates.")

        return quantum_circuit

    def apply_pauli_twirling(self, circuit):
        """Applies Pauli twirling to a circuit by inserting random Pauli gates before and after each CX gate."""
        twirled_circuit = QuantumCircuit(*circuit.qregs, *circuit.cregs)  # Preserve classical bits

        for instruction in circuit.data:
            op = instruction.operation
            qubits = instruction.qubits
            clbits = instruction.clbits

            if op.name == "cx":
                # Select random Pauli gates (I, X, Y, Z)
                pauli1 = random.choice([IGate(), XGate(), YGate(), ZGate()])
                pauli2 = random.choice([IGate(), XGate(), YGate(), ZGate()])

                # Apply Pauli gates before and after CX
                twirled_circuit.append(pauli1, [qubits[0]])
                twirled_circuit.append(pauli2, [qubits[1]])
                twirled_circuit.append(op, qubits)  # Apply CX
                twirled_circuit.append(pauli2, [qubits[1]])  # Same Pauli gate after
            else:
                twirled_circuit.append(op, qubits, clbits)  # Handle classical bits properly

        return twirled_circuit

    def get_instruction_durations(self, backend):
        """Extract instruction durations from backend properties or set defaults."""
        durations = InstructionDurations()

        if backend and hasattr(backend, "properties") and backend.properties():
            backend_properties = backend.properties()
            for gate in backend_properties.gates:
                for qubits in gate.qubits:
                    duration = backend_properties.gate_length(gate.gate, qubits)
                    if duration:
                        durations.update([(gate.gate, qubits, duration)])

        else:
            num_qubits = backend.configuration().n_qubits if hasattr(backend,
                                                                     "configuration") else 2  # Default to 2 qubits

            for qubit in range(num_qubits):
                durations.update([("cx", [qubit, (qubit + 1) % num_qubits], 300)])  # CX duration
                durations.update([("x", [qubit], 100)])  # X gate duration
                durations.update([("measure", [qubit], 1000)])  # Measure duration for all qubits

        return durations

    def apply_dynamic_decoupling(self, circuit, backend):
        durations = InstructionDurations()

        # Extract durations from backend properties if available
        if backend and hasattr(backend, "properties") and backend.properties():
            backend_properties = backend.properties()
            for gate in backend_properties.gates:
                for qubits in gate.qubits:
                    duration = backend_properties.gate_length(gate.gate, qubits)
                    if duration:
                        durations.update([(gate.gate, qubits, duration)])
        else:
            num_qubits = backend.configuration().n_qubits if hasattr(backend, "configuration") else 2

            for qubit in range(num_qubits):
                durations.update([("cx", [qubit, (qubit + 1) % num_qubits], 300)])  # CX duration
                durations.update([("x", [qubit], 100)])  # X gate duration
                durations.update([("measure", [qubit], 1000)])  # Ensure measure duration is set for all qubits

        # Apply ALAP scheduling
        scheduler = ALAPSchedule(durations)
        dag = circuit_to_dag(circuit)
        try:
            scheduled_dag = scheduler.run(dag)  # Apply scheduling
        except TranspilerError as e:
            print(f"⚠ "
                  f": {e}")
            return circuit  # Return original circuit if scheduling fails

        scheduled_qc = dag_to_circuit(scheduled_dag)

        #  Apply Dynamical Decoupling (DD)
        dd_sequence = [XGate(), XGate()]  # Simple DD sequence (XY4)
        dd_pass = DynamicalDecoupling(durations, dd_sequence)

        dag = circuit_to_dag(scheduled_qc)
        try:
            dd_dag = dd_pass.run(dag)  # Apply DD on scheduled circuit
        except TranspilerError as e:
            print(f"⚠ DD application failed: {e}")
            return scheduled_qc  # Return scheduled circuit if DD fails

        dd_qc = dag_to_circuit(dd_dag)  # Convert back to QuantumCircuit

        return dd_qc

    def apply_twirling_and_dynamic_decoupling(self, circuit, backend):
        """Applies Pauli Twirling, schedules the circuit, and then applies Dynamic Decoupling (DD)."""
        twirled_qc = self.apply_pauli_twirling(circuit)  # Step 1: Twirling
        dd_qc = self.apply_dynamic_decoupling(twirled_qc, backend)  # Step 2: DD

        return dd_qc

    def no_mitigation(self, circuit, model):
        backend = AerSimulator()

        job_ideal = backend.run(circuit, shots=1024)
        ideal_results = job_ideal.result()
        ideal_counts = ideal_results.get_counts()


        job_no_mitigation = backend.run(circuit, shots=1024, noise_model=model)
        results_no_mitigation = job_no_mitigation.result()
        counts_no_mitigation = results_no_mitigation.get_counts()

        ## ================================
        # Apply Twirling Only
        # ================================
        twirled_qc = self.apply_pauli_twirling(circuit)

        job_twirled = backend.run(twirled_qc, shots=1024, noise_model=model)
        results_twirled = job_twirled.result()
        counts_twirled = results_twirled.get_counts()

        ## ================================
        # Apply Dynamic Decoupling Only
        # ================================
        dd_qc_only = self.apply_dynamic_decoupling(circuit, backend)

        job_dd_only = backend.run(dd_qc_only, shots=1024, noise_model=model)
        results_dd_only = job_dd_only.result()
        counts_dd_only = results_dd_only.get_counts()

        ## ================================
        # Apply Both Twirling + DD
        # ================================
        twirling_dd_qc = self.apply_twirling_and_dynamic_decoupling(circuit, backend)

        job_dd = backend.run(twirling_dd_qc, shots=1024, noise_model=model)
        results_dd = job_dd.result()
        counts_dd = results_dd.get_counts()

        return (ideal_results, ideal_counts, results_no_mitigation, counts_no_mitigation,
                results_twirled, counts_twirled,
                results_dd_only, counts_dd_only,
                results_dd, counts_dd)

    # ================================
    # 6. Compute Fidelities
    # ================================
    def compute_fidelity(self, ideal_counts, noisy_counts):
        """Compute classical fidelity between ideal and noisy distributions."""
        all_keys = set(ideal_counts.keys()).union(noisy_counts.keys())

        ideal_probs = {key: ideal_counts.get(key, 0) / sum(ideal_counts.values()) for key in all_keys}
        noisy_probs = {key: noisy_counts.get(key, 0) / sum(noisy_counts.values()) for key in all_keys}

        return sum(np.sqrt(ideal_probs[key] * noisy_probs[key]) for key in all_keys)

    def run_experiment_for_file(self, filename, save_file):
        backend = AerSimulator()
        cur_dir = os.getcwd()
        try:
            if cur_dir not in sys.path:
                sys.path.append(cur_dir)
        except Exception as e:
            print(f"Error occurred: {e}")
        directory = self.directory
        circuit_directory = os.path.join(cur_dir, directory)
        simulator_logs = "logs_" + save_file + "_" + filename + "_" + self.crosstalk_version + "_" + str(
            self.crosstalk_fidelity) + self.filename
        file_csv = "csv_" + save_file + "_" + filename[:-5] + "_" + self.crosstalk_version + "_" + str(
            self.crosstalk_fidelity) + self.filename + '.csv'
        logging.basicConfig(filename=simulator_logs, level=logging.INFO, filemode="w" )

        with open(file_csv, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                ['connectivity', 'crosstalk_fidelity', 'crosstalk_version', 'optimisation_level',
                 'num_qubits_qc', 'num_qubits_transpiled', 'depth', 'depth_transpiled_circ',
                 'time', 'gate_counts', 'gate_names ', 'gate_counts_transpiled_circ', 'gate_names_transpiled_circ',
                 'fidelity_crosstalk', "fidelity_crosstalk_twirling", "fidelity_crosstalk_decoupling",
                  "fidelity_crosstalk_both"])

        file_path = os.path.join(circuit_directory, filename)
        quantum_circuit = QuantumCircuit.from_qasm_file(file_path)
        quantum_circuit = self.load_and_prepare_circuit_measure(quantum_circuit)

        # Check the number of qubits before proceeding
        if quantum_circuit.num_qubits > 10:
            print(f"Skipping circuit {filename} as it has {quantum_circuit.num_qubits} qubits.")
            return

        for connectivity in range(len(self.connectivity_density)):
            cmap_ext = increase_coupling_density(self.coupling_map, self.connectivity_density[connectivity])
            start = time.time()

            transpiled_qc = transpile(quantum_circuit, coupling_map=cmap_ext,
                                        optimization_level=self.optimization_level, basis_gates=self.gate_set
                                        )

            if self.crosstalk_version == "cxneighbors":
                noiseModel = create_crosstalk_noise_from_circuit(transpiled_qc, self.crosstalk_fidelity,
                                                                 self.neighbor_fidelity)
            elif self.crosstalk_version == "ncx":
                noiseModel = create_crosstalk_noise_from_layers(transpiled_qc, self.crosstalk_fidelity,
                                                                self.neighbor_fidelity)
            elif self.crosstalk_version == "topology":
                noiseModel = create_crosstalk_noise_from_proximity(transpiled_qc, self.crosstalk_fidelity,
                                                                   self.neighbor_fidelity, max_distance=2)

            end = time.time()
            times = []
            depth = []
            depth_transpiled = []
            gate_counts = []
            gate_counts_transpiled = []
            gate_names = []
            gate_names_transpiled = []
            crosstalk_fidelities = []
            fidelity_twirl = []
            fidelity_decoupling = []
            fidelity_both = []

            (ideal_results, ideal_counts, results_no_mitigation, counts_no_mitigation, results_twirled,
            counts_twirled, results_dd_only,
            counts_dd_only,
            results_dd, counts_dd) = self.no_mitigation(transpiled_qc, noiseModel)
            print(ideal_results, ideal_counts, results_no_mitigation, counts_no_mitigation, results_twirled, counts_twirled, results_dd_only)

            fidelity_no_mitigation = self.compute_fidelity(ideal_counts, counts_no_mitigation)
            fidelity_twirling = self.compute_fidelity(ideal_counts, counts_twirled)
            fidelity_twirling_dd = self.compute_fidelity(ideal_counts, counts_dd)
            fidelity_dd = self.compute_fidelity(ideal_counts, counts_dd_only)

            crosstalk_fidelities.append(fidelity_no_mitigation)
            fidelity_twirl.append(fidelity_twirling)
            fidelity_decoupling.append(fidelity_dd)
            fidelity_both.append(fidelity_twirling_dd)

            times.append(end - start)
            depth.append(quantum_circuit.depth())
            depth_transpiled.append(transpiled_qc.depth())
            for gate, count in quantum_circuit.count_ops().items():
                gate_names.append(gate)
                gate_counts.append(count)

            for gate, count in transpiled_qc.count_ops().items():
                gate_names_transpiled.append(gate)
                gate_counts_transpiled.append(count)

            with open(file_csv, 'a', newline='') as file:
                print("we are writing our file")
                writer = csv.writer(file)
                writer.writerow([self.connectivity_density[connectivity], self.crosstalk_fidelity,
                                 self.crosstalk_version, self.optimization_level,
                                 quantum_circuit.num_qubits, transpiled_qc.num_qubits,
                                 depth, depth_transpiled, times, gate_counts, gate_names,
                                 gate_counts_transpiled, gate_names_transpiled,
                                 crosstalk_fidelities, fidelity_twirling, fidelity_decoupling, fidelity_both
                                 ])
    def run_experiment(self, save_file):
        cur_dir = os.getcwd()
        if cur_dir not in sys.path:
            sys.path.append(cur_dir)
        circuit_directory = os.path.join(cur_dir, self.directory)
        files = os.listdir(circuit_directory)
        qasm_files = [f for f in files if f.endswith('.qasm')]

        # Parallel execution using multiprocessing
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count() //2, maxtasksperchild=1)
        for filename in qasm_files:
            result = pool.apply_async(self.run_experiment_for_file, args=(filename, save_file))
            try:
                result.get(timeout=600)
            except multiprocessing.TimeoutError:
                print(f"Process for {filename} timed out and will be skipped. ")

        pool.close()
        pool.join()


def main():
    parser = argparse.ArgumentParser(description="TWiDDle: Run Quantum Experiments with Crosstalk Noise and Mitigation"
                                                     "techniques")

    default_connectivity_density = [0.01]
    connectivity_density = [0.013895, 0.015, 0.018, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.1, 0.15, 0.2, 0.25,
                            0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    default_gate_set = ['id', 'rz', 'sx', 'x', 'cx', 'swap', 'cz']
    default_crosstalk_fidelity = 0.98847
    default_neighbor_fidelity = 0.9997285
    default_opt_level = 0
    default_rows, default_cols = 6, 4
    default_directory = "circuits"
    default_crosstalk_version = "cxneighbors"

    # Ask user for values interactively
    experiment_name = input(f"Enter experiment name (default: Exp): ") or "Exp"
    crosstalk_version = input(f"Enter crosstalk version [cxneighbors/ncx/topology] (default: {default_crosstalk_version}): ") or default_crosstalk_version
    crosstalk_fidelity = float(input(f"Enter crosstalk fidelity (default: {default_crosstalk_fidelity}): ") or default_crosstalk_fidelity)
    neighbor_fidelity = float(input(f"Enter neighbor fidelity (default: {default_neighbor_fidelity}): ") or default_neighbor_fidelity)
    opt_level = int(input(f"Enter optimization level [0-3] (default: {default_opt_level}): ") or default_opt_level)
    rows = int(input(f"Enter number of rows for coupling map (default: {default_rows}): ") or default_rows)
    cols = int(input(f"Enter number of columns for coupling map (default: {default_cols}): ") or default_cols)
    directory = input(f"Enter directory containing QASM circuits (default: {default_directory}): ") or default_directory

    # Convert input to list if multiple values are provided
    conn_density_input = input(f"Enter connectivity density values separated by space (default: {default_connectivity_density}): ")
    connectivity_density = [float(x) for x in conn_density_input.split()] if conn_density_input else default_connectivity_density

    backend = AerSimulator()
    coupling_map = create_heavy_hex_IBMQ(rows, cols)

    exp = RunExperiments(experiment_name, backend, crosstalk_version, crosstalk_fidelity,
                         neighbor_fidelity, connectivity_density, opt_level,
                         coupling_map, directory, default_gate_set)

    filename = f"{experiment_name}_hhex_{rows}_{cols}_opt{opt_level}_runs"
    exp.run_experiment(filename)


if __name__ == "__main__":
    main()
