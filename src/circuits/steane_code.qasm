// Steane Code Example in QASM
    OPENQASM 2.0;
    include "qelib1.inc";

    qreg q[7];
    creg c[7];

    // Encoding logical |0> using Steane code
    barrier q;
    h q[0]; cx q[0], q[1]; cx q[0], q[2];

    h q[0]; h q[1]; h q[2];

    cx q[0], q[3];
    cx q[1], q[4];
    cx q[2], q[5];

    cx q[0], q[6];
    cx q[1], q[6];
    cx q[2], q[6];

    // Introduce an example phase flip error on q[5]
    z q[5];

    // Syndrome measurement for error correction
    measure q[0] -> c[0];
    measure q[1] -> c[1];
    measure q[2] -> c[2];
    