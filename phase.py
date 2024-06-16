from qiskit import QuantumRegister, ClassicalRegister, AncillaRegister, QuantumCircuit
from qiskit.quantum_info import Statevector, Operator
from qiskit_aer import Aer 
from qiskit import transpile, assemble
from qiskit.result.utils import marginal_distribution
from qiskit.visualization import plot_histogram, plot_bloch_multivector, plot_distribution, plot_state_qsphere
import matplotlib.pyplot as plt
import numpy as np
import math

_global_plots_config_ = "hide"          # "hide" / "show" / "save"

#___________________________________
# INPUT
def getInput(n = 4, verbose = False) -> tuple[np.ndarray, np.ndarray]:
    """Generate linearly-spaced input vector in range [0, 255] and interpolation in radians in range [0, pi] both of size 'n'. We assume square images ('n' should be a perfect square).

    Args:
        `n` (int, optional): size of input. Defaults to 4.
        `verbose` (bool, optional): log additional info and graphs. Graph plotting depends on additional global config (_global_plots_config_). Defaults to False.

    Returns:
        `tuple[np.ndarray, np.ndarray]`: (vector, angles)
    """
    input_vector = np.linspace(start=0, stop=255, num=n, dtype=int)
    input_angles = np.interp(input_vector, (0, 255), (0, np.pi))
    
    if verbose:
        print(input_vector,"\n", input_angles)
        plt.title('Input image')
        plt.imshow(input_vector.reshape(2,2), cmap='gray')

    return input_vector, input_angles

#___________________________________
# ENCODER
def phaseEncoder(qc: QuantumCircuit, angles: np.array, measure = False, verbose = False):
    """Add Phase encoding model to a blank circuit.

    Args:
        qc (QuantumCircuit): _description_
        angles (np.array): _description_
        measure (bool, optional): _description_. Defaults to False.
        verbose (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    q = QuantumRegister(len(angles), 'q')
    c = ClassicalRegister(len(angles), 'c')

    qc.add_register(q, c)

    for i, ang in enumerate(angles):
        qc.h(i)
        qc.p(ang, i)
    
    qc.barrier()

    for i in range(len(angles)):
        qc.h(i)

    # sv = Statevector(qc)
    if verbose: print(sv)
    
    if measure: qc.measure(range(len(angles)), range(len(angles)))
    else: qc.barrier()

    return None

#___________________________________
# INVERTER LOGIC
def invertPixels(qc: QuantumCircuit, verbose = False):
    for i in range(qc.num_qubits):
        qc.x(i)

    qc.barrier()

    if verbose: print(qc)
    return qc

#___________________________________
# Measurements
def addMeasurements(qc: QuantumCircuit, verbose = False):
    qc.measure(list(range(qc.num_qubits)), list(range(qc.cregs[0].size)))
    
    if verbose: print(f"Adding Measurments:\n{qc}")

    return qc

#___________________________________
# SIMULATE
def simulate(qc: QuantumCircuit, shots = 1000000, verbose = False):
    aer_sim = Aer.get_backend('aer_simulator')
    t_qc_image = transpile(qc, aer_sim)
    qobj = assemble(t_qc_image, shots=shots)
    job = aer_sim.run(qobj)
    result = job.result()
    counts = result.get_counts()

    if verbose:
        print(counts)
        # display(plot_histogram(counts))
    
    return counts

#___________________________________
# DECODE
def phaseDecoder(counts, n = 4, shots = 1000000, verbose = False):
    output_values = []

    for  iq in range(n):
        marCnt=marginal_distribution(counts,[iq])
        #print('iq:',iq,marCnt)
        if '1' in marCnt:
            prob=marCnt['1']/shots
        else:
            prob=0
        ev=1-2*prob
        output_values.append(np.arccos(ev))
    
    output_values = np.interp(output_values, (0, np.pi), (0, 255)).astype(int)

    # RECONSTRUCT
    if verbose:
        print("q0, q1, q2, q3 : ", output_values)

        for i, value in enumerate(output_values):
            print(f"\nFor pixel {i}:")
            print(f"\tvalue/shots = {value}/{shots} = {value/shots}")
            print(f"\tsqrt(value/shots) = {(value/shots)**(1/2)}")
            print(f"\tarccos = {np.arccos((value/shots)**(1/2))}")
            print(f"\t2 * arccos = {2*np.arccos((value/shots)**(1/2))}")

    return list(output_values)

#___________________________________
# Compare
def plot_to_compare(input_vector, output_vector, verbose = False):
    if verbose: print(f"\nOriginal values: {input_vector}\nReconstructed Inverted values: {output_vector}")

    plt.imshow(np.reshape(input_vector, (2, 2)), cmap = 'gray')
    plt.title('Input Image')
    plt.show()
    plt.imshow(np.reshape(output_vector, (2, 2)), cmap = 'gray')
    plt.title('Reconstructed Inverted Image')
    plt.show()

# TODO
#___________________________________
def testPhase(n=4):
    pass


if __name__ == "__main__":
    qc = QuantumCircuit()
    phaseEncoder(qc, [0, 0.5, 0.6, 0.7])
    addMeasurements(qc, True)
