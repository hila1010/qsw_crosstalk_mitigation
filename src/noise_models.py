from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error, QuantumError
from qiskit_aer.noise.errors import ReadoutError
import networkx as nx
from qiskit.converters import circuit_to_dag
from qiskit.providers.fake_provider import Fake127QPulseV1
from qiskit import transpile
import numpy as np


def create_crosstalk_noise_from_circuit(transpiled_circuit, crosstalk_fidelity, neighbor_fidelity):
    """
    Creates a noise model based on a transpiled circuit:
    - Crosstalk occurs when two CX gates share a common qubit.
    - A weaker error applies to single-qubit gates on neighboring qubits.

    Args:
        transpiled_circuit (QuantumCircuit): Transpiled quantum circuit.
        crosstalk_fidelity (float): Fidelity of CX gates under crosstalk.
        neighbor_fidelity (float): Fidelity penalty for single-qubit neighboring operations.

    Returns:
        NoiseModel: Custom noise model.
    """
    noise_model = NoiseModel()
    qubit_graph = nx.Graph()
    two_qubit_gates = set()  # Track CX operations

    # Build qubit interaction graph from the transpiled circuit
    for gate in transpiled_circuit.data:
        qubits = [transpiled_circuit.find_bit(q).index for q in gate.qubits]

        if len(qubits) == 2 and gate.operation.name == "cx":
            qubit_graph.add_edge(qubits[0], qubits[1])
            two_qubit_gates.add(tuple(sorted([qubits[0], qubits[1]])))  # Store CX gate pairs
        elif len(qubits) == 1:
            qubit_graph.add_node(qubits[0])  # Include single-qubit operations

    # Define noise models
    cx_crosstalk_error = depolarizing_error(1 - crosstalk_fidelity, 2)
    single_qubit_neighbor_error = depolarizing_error(1 - neighbor_fidelity, 1)

    # Identify CX gate pairs that share a qubit (crosstalk)
    for edge1 in qubit_graph.edges:
        for edge2 in qubit_graph.edges:
            if edge1 == edge2:
                continue  # Skip self-pairs

            shared_qubit = set(edge1).intersection(set(edge2))  # Find common qubit
            if shared_qubit:  # If there's a shared qubit, crosstalk occurs
                shared_qubit = list(shared_qubit)[0]

                # Apply crosstalk noise to both CX gates
                noise_model.add_quantum_error(cx_crosstalk_error, ["cx"], list(edge1))
                noise_model.add_quantum_error(cx_crosstalk_error, ["cx"], list(edge2))

                # Apply lower noise to all neighboring single-qubit gates
                involved_qubits = set(edge1) | set(edge2)  # Qubits involved in both CX gates
                for qubit in involved_qubits:
                    for neighbor in qubit_graph.neighbors(qubit):
                        if neighbor not in involved_qubits:  # Only affect external neighbors
                            noise_model.add_quantum_error(single_qubit_neighbor_error, ["u1", "u2", "u3"], [neighbor])

    return noise_model


def create_crosstalk_noise_from_layers(transpiled_circuit, crosstalk_fidelity, neighbor_fidelity):
    """
    Creates a noise model where crosstalk occurs when multiple two-qubit gates are executed simultaneously.
    Additionally, applies weaker noise to neighboring single-qubit gates.

    Args:
        transpiled_circuit (QuantumCircuit): Transpiled quantum circuit.
        crosstalk_fidelity (float): Fidelity of CX gates under crosstalk.
        neighbor_fidelity (float): Fidelity penalty for single-qubit neighboring operations.

    Returns:
        NoiseModel: Custom noise model.
    """
    noise_model = NoiseModel()
    qubit_graph = nx.Graph()

    # Get physical qubit mapping safely
    if transpiled_circuit.layout is not None and transpiled_circuit.layout.final_layout is not None:
        physical_qubits = transpiled_circuit.layout.final_layout._v2p
    else:
        physical_qubits = {q: i for i, q in enumerate(transpiled_circuit.qubits)}  # Fallback to direct mapping

    # Extract layers of CX gates using DAG representation
    dag = circuit_to_dag(transpiled_circuit)
    layerwise_gates = []
    for layer in dag.layers():
        layer_cx_gates = []
        for node in layer["graph"].nodes():
            if hasattr(node, "op") and node.op.name == "cx":
                qubits = tuple(sorted(physical_qubits[q] for q in node.qargs))
                layer_cx_gates.append(qubits)
                qubit_graph.add_edge(*qubits)
        if len(layer_cx_gates) > 1:
            layerwise_gates.append(layer_cx_gates)

    # Debugging: Print detected simultaneous CX layers
    print("Identified CX layers:", layerwise_gates)

    # Define noise models
    cx_crosstalk_error = depolarizing_error(1 - crosstalk_fidelity, 2)
    single_qubit_neighbor_error = depolarizing_error(1 - neighbor_fidelity, 1)

    # Apply noise where multiple CX gates execute in a layer
    for layer in layerwise_gates:
        involved_qubits = set()
        for cx_pair in layer:
            print(f"Applying crosstalk noise to CX gate: {cx_pair}")  # Debugging output
            noise_model.add_quantum_error(cx_crosstalk_error, ["cx"], list(cx_pair))
            involved_qubits.update(cx_pair)

        # Apply neighbor noise to adjacent single-qubit operations
        for qubit in involved_qubits:
            for neighbor in qubit_graph.neighbors(qubit):
                if neighbor not in involved_qubits:
                    print(f"Applying neighbor noise to qubit: {neighbor}")  # Debugging output
                    noise_model.add_quantum_error(single_qubit_neighbor_error, ["u1", "u2", "u3"], [neighbor])

    # Debugging: Print noise model summary
    print("Final noise model:", noise_model)

    return noise_model


def create_crosstalk_noise_from_proximity_(transpiled_circuit, crosstalk_fidelity, neighbor_fidelity, max_distance=2):
    """
    Creates a noise model where crosstalk occurs when multiple two-qubit gates are executed in close physical proximity.
    Additionally, applies weaker noise to neighboring single-qubit gates.

    Args:
        transpiled_circuit (QuantumCircuit): Transpiled quantum circuit.
        crosstalk_fidelity (float): Fidelity of CX gates under crosstalk.
        neighbor_fidelity (float): Fidelity penalty for single-qubit neighboring operations.
        max_distance (int): Maximum physical distance between two CX gates to consider crosstalk.

    Returns:
        NoiseModel: Custom noise model.
    """
    noise_model = NoiseModel()
    fake_backend = Fake127QPulseV1()  # Use the closest fake backend to Heron
    coupling_map = fake_backend.configuration().coupling_map
    qubit_graph = nx.Graph()
    qubit_graph.add_edges_from(coupling_map)

    # Define noise models
    cx_crosstalk_error = depolarizing_error(1 - crosstalk_fidelity, 2)
    single_qubit_neighbor_error = depolarizing_error(1 - neighbor_fidelity, 1)

    # Extract CX operations and track qubits involved
    cx_gates = []
    for gate in transpiled_circuit.data:
        qubits = [transpiled_circuit.find_bit(q).index for q in gate.qubits]
        if len(qubits) == 2 and gate.operation.name == "cx":
            cx_gates.append(tuple(qubits))

    # Check for crosstalk based on proximity in hardware coupling map
    for i, (q1a, q1b) in enumerate(cx_gates):
        for j, (q2a, q2b) in enumerate(cx_gates):
            if i >= j:
                continue  # Avoid duplicate pairs

            # Check if CX gates are within max_distance on hardware topology
            if (nx.shortest_path_length(qubit_graph, q1a, q2a) <= max_distance or
                    nx.shortest_path_length(qubit_graph, q1a, q2b) <= max_distance or
                    nx.shortest_path_length(qubit_graph, q1b, q2a) <= max_distance or
                    nx.shortest_path_length(qubit_graph, q1b, q2b) <= max_distance):

                # Apply crosstalk noise
                noise_model.add_quantum_error(cx_crosstalk_error, ["cx"], [q1a, q1b])
                noise_model.add_quantum_error(cx_crosstalk_error, ["cx"], [q2a, q2b])

                # Apply neighbor noise to surrounding qubits
                for q in {q1a, q1b, q2a, q2b}:
                    for neighbor in qubit_graph.neighbors(q):
                        if neighbor not in {q1a, q1b, q2a, q2b}:  # Exclude main CX qubits
                            noise_model.add_quantum_error(single_qubit_neighbor_error, ["u1", "u2", "u3"], [neighbor])

    return noise_model


def create_crosstalk_noise_from_proximity(transpiled_circuit, crosstalk_fidelity, neighbor_fidelity, max_distance=2):
    """
    Creates a noise model where crosstalk occurs when multiple two-qubit gates are executed in close physical proximity.
    Additionally, applies weaker noise to neighboring single-qubit gates.

    Args:
        transpiled_circuit (QuantumCircuit): Transpiled quantum circuit.
        crosstalk_fidelity (float): Fidelity of CX gates under crosstalk.
        neighbor_fidelity (float): Fidelity penalty for single-qubit neighboring operations.
        max_distance (int): Maximum physical distance between two CX gates to consider crosstalk.

    Returns:
        NoiseModel: Custom noise model.
    """
    noise_model = NoiseModel()
    fake_backend = Fake127QPulseV1()  # Use the closest fake backend to Heron

    # Extract qubit coordinates from backend properties
    qubit_coordinates = {i: (q[0].value, q[1].value) for i, q in enumerate(fake_backend.properties().qubits)}

    # Define noise models
    cx_crosstalk_error = depolarizing_error(1 - crosstalk_fidelity, 2)
    single_qubit_neighbor_error = depolarizing_error(1 - neighbor_fidelity, 1)

    # Extract CX operations and track qubits involved
    cx_gates = []
    for gate in transpiled_circuit.data:
        qubits = [transpiled_circuit.find_bit(q).index for q in gate.qubits]
        if len(qubits) == 2 and gate.operation.name == "cx":
            cx_gates.append(tuple(qubits))

    # Check for crosstalk based on physical distances
    for i, (q1a, q1b) in enumerate(cx_gates):
        for j, (q2a, q2b) in enumerate(cx_gates):
            if i >= j:
                continue  # Avoid duplicate pairs

            # Compute Euclidean distances
            dist1 = np.linalg.norm(np.array(qubit_coordinates[q1a]) - np.array(qubit_coordinates[q2a]))
            dist2 = np.linalg.norm(np.array(qubit_coordinates[q1a]) - np.array(qubit_coordinates[q2b]))
            dist3 = np.linalg.norm(np.array(qubit_coordinates[q1b]) - np.array(qubit_coordinates[q2a]))
            dist4 = np.linalg.norm(np.array(qubit_coordinates[q1b]) - np.array(qubit_coordinates[q2b]))

            if min(dist1, dist2, dist3, dist4) <= max_distance:
                # Apply crosstalk noise
                noise_model.add_quantum_error(cx_crosstalk_error, ["cx"], [q1a, q1b])
                noise_model.add_quantum_error(cx_crosstalk_error, ["cx"], [q2a, q2b])

                # Apply neighbor noise to surrounding qubits
                involved_qubits = {q1a, q1b, q2a, q2b}
                for q in involved_qubits:
                    for neighbor, coord in qubit_coordinates.items():
                        if neighbor not in involved_qubits:
                            neighbor_dist = np.linalg.norm(np.array(qubit_coordinates[q]) - np.array(coord))
                            if neighbor_dist <= max_distance:
                                noise_model.add_quantum_error(single_qubit_neighbor_error, ["u1", "u2", "u3"],
                                                              [neighbor])

    return noise_model

def create_thermal_noise_model(T1, T2, gate_time, num_qubits):
    noise_model = NoiseModel()

    # Thermal relaxation error for each qubit individually
    for qubit in range(num_qubits):
        error = thermal_relaxation_error(T1, T2, gate_time)
        noise_model.add_quantum_error(error, 'id', [qubit])  # Identity gate noise
        noise_model.add_quantum_error(error, 'u3', [qubit])  # U3 gate noise (generic single-qubit gate)

    return noise_model


def create_depolarization_noise_model(error_prob, num_qubits):
    noise_model = NoiseModel()

    # Depolarizing error for each qubit
    for qubit in range(num_qubits):
        error = depolarizing_error(error_prob, 1)  # Single-qubit depolarizing error
        noise_model.add_quantum_error(error, ['u3'], [qubit])  # Add error to all single-qubit u3 gates

    return noise_model


def create_measurement_noise_model(readout_error_prob, num_qubits):
    noise_model = NoiseModel()

    # Measurement error for each qubit
    meas_error = ReadoutError(readout_error_prob)
    for qubit in range(num_qubits):
        noise_model.add_readout_error(meas_error, [qubit])

    return noise_model





