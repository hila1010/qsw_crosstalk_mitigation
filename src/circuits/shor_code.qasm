// Shor Code Example in QASM
    OPENQASM 2.0;
    include "qelib1.inc";

    qreg q[9];
    creg c[9];

    // Encode |0>_L using Shor Code
    barrier q;
    h q[0]; cx q[0], q[1]; cx q[0], q[2];

    // Bit-flip code (3-qubit repetition)
    barrier q;
    h q[0]; cx q[0], q[3]; cx q[0], q[6];
    h q[1]; cx q[1], q[4]; cx q[1], q[7];
    h q[2]; cx q[2], q[5]; cx q[2], q[8];

    // Introduce an example error (bit flip on q[2])
    x q[2];

    // Syndrome measurement
    measure q[0] -> c[0];
    measure q[1] -> c[1];
    measure q[2] -> c[2];
    