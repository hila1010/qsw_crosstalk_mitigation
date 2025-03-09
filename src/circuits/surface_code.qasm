// Simplified Surface Code Example in QASM
    OPENQASM 2.0;
    include "qelib1.inc";

    qreg q[5];
    creg c[5];

    // Initialize qubits
    h q[0];
    h q[1];
    h q[2];

    // Create stabilizer measurements (simplified example)
    cx q[0], q[3];
    cx q[1], q[3];
    cx q[1], q[4];
    cx q[2], q[4];

    // Introduce an example error (bit flip on q[2])
    x q[2];

    // Measure stabilizers
    measure q[3] -> c[3];
    measure q[4] -> c[4];
    