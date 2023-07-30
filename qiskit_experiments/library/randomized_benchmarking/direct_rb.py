from qiskit_experiments.framework import BaseExperiment, Options
from qiskit import QuantumCircuit
from qiskit.providers import Options
from qiskit.quantum_info import random_unitary
from qiskit.quantum_info.operators import Operator
from qiskit.circuit import Parameter
from qiskit.circuit.library.standard_gates import RZGate, SXGate

import numpy as np
import math, cmath
from scipy.linalg import det

from simpler_single_qubit_rb_analysis import SimplerSingleQubitRBAnalysis
from rb_analysis_with_flipped_target_states import RBAnalysisWithFlippedTargetStates

class DirectRB(BaseExperiment):
    def __init__(self, physical_qubits, backend=None, **experiment_options):
        super().__init__(physical_qubits, analysis=None, backend=backend)
        self.set_experiment_options(**experiment_options)

    def _finalize(self):
        if self.experiment_options.flip_final_state:
            self.analysis=RBAnalysisWithFlippedTargetStates()  # has two models with opposite amplitudes
        else:
            self.analysis=SimplerSingleQubitRBAnalysis()  # has only one model
        return super()._finalize()
    
    @classmethod
    def _default_experiment_options(cls):
        options = super()._default_experiment_options()
        options.update_options(
            min_lengths=1,
            max_lengths=700,
            num_lengths=5,
            unitary_lengths=None,  # supply array
            rx_angles=[np.pi*1/3, np.pi*2/3],  # Note: excluding zero angle
            flip_final_state=True,
            num_samples=6,  # even number
            seed=1004,
        )
        return options

    def unitary_lengths(self):
        opt = self.experiment_options

        if opt.unitary_lengths is None:
            unitary_lengths = np.linspace(opt.min_lengths, opt.max_lengths, opt.num_lengths, dtype=int)
        else:
            unitary_lengths = np.asarray(opt.unitary_lengths, dtype=int)
        
        return unitary_lengths
    
    @classmethod
    def _default_transpile_options(cls) -> Options:
        """Default transpiler options for transpiling RB circuits."""
        return Options(optimization_level=0, target=None)
    
    def _euler_angles_to_gate_sequence(self, theta, phi, lam):
        """Single qubit Clifford decomposition that ignores global phase.
        """
        return [
            RZGate(lam),
            SXGate(),
            RZGate(theta + math.pi),
            SXGate(),
            RZGate(phi - math.pi),
        ]
    
    def euler_decomposition(self, mat:np.array): 
        su_mat = det(mat) ** (-0.5) * mat
        theta = 2 * math.atan2(abs(su_mat[1, 0]), abs(su_mat[0, 0]))
        phiplambda2 = cmath.phase(su_mat[1, 1])
        phimlambda2 = cmath.phase(su_mat[1, 0])
        phi = phiplambda2 + phimlambda2
        lam = phiplambda2 - phimlambda2

        return theta, phi, lam
    
    def circuits(self):
        circuits = []
        options = self.experiment_options
        rng = np.random.default_rng(seed=options["seed"])

        for idx in range(options["num_samples"]):
            for unitary_length in self.unitary_lengths():
                # create a quantum circuit with randomly sampled unitary gates
                qc = QuantumCircuit(1, 1)

                ### "C_{sp}" subcircuit ###
                psi_0 = random_unitary(2, seed=rng)
                theta, phi, lam = self.euler_decomposition(psi_0.to_matrix())
                csp_sequence = self._euler_angles_to_gate_sequence(theta=theta, phi=phi, lam=lam)
                for gate in csp_sequence:
                    qc.append(gate, qc.qubits, [])
                    # qc.barrier()  # safeguard against optimization
                qc.barrier()

                ### "core" subcircuit ###
                for _ in range(unitary_length):
                    rz_angle_1, rz_angle_2 = rng.uniform(low=0, high=np.pi, size=2)
                    qc.rz(rz_angle_1, 0)
                    qc.rx(rng.choice(options["rx_angles"]), 0)
                    qc.rz(rz_angle_2, 0)
                    qc.barrier()

                ### "C_{mp}" subcircuit ###
                # calculate the inverse and convert it to RZ-SX-RZ-SX-RZ sequence.
                first_two_subcircs = Operator(qc)
                inverse = np.linalg.inv(first_two_subcircs)

                if self.experiment_options.flip_final_state:
                    if idx % 2 == 1:  # target state = 1
                        inverse = np.matmul(np.array([[0,1],[1,0]]), inverse)
                        qc._metadata.update({"target_state": 1})
                    else:
                        qc._metadata.update({"target_state": 0})
                        
                theta, phi, lam = self.euler_decomposition(inverse)
                cmp_sequence = self._euler_angles_to_gate_sequence(theta=theta, phi=phi, lam=lam)
                for gate in cmp_sequence:
                    qc.append(gate, qc.qubits, [])
                    # qc.barrier()  # safeguard against optimization

                # a DRB circuit is now ready
                qc.measure(0, 0)
                qc._metadata.update({"xval": int(unitary_length)})
                circuits.append(qc)

        return circuits
