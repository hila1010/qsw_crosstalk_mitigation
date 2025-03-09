// Repetition Code Example in QASM
    OPENQASM 2.0;
    include "qelib1.inc";

    qreg q[3];
    creg c[3];

    // Encode |0> using 3-qubit repetition code
    barrier q;
    h q[0];
    cx q[0], q[1];
    cx q[0], q[2];

    // Introduce an example bit-flip error on q[1]
    x q[1];

    // Measure for error detection
    measure q[0] -> c[0];
    measure q[1] -> c[1];
    measure q[2] -> c[2];
    