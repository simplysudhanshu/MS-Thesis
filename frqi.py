from qiskit import QuantumRegister, ClassicalRegister, AncillaRegister, QuantumCircuit
from qiskit.quantum_info import Statevector, Operator
from qiskit_aer import Aer 
from qiskit import transpile, assemble
from qiskit.visualization import plot_histogram, plot_bloch_multivector, plot_distribution, plot_state_qsphere
import matplotlib.pyplot as plt
import numpy as np
import math

_global_plots_config_ = "hide"          # "hide" / "show" / "save"

#___________________________________
# INPUT
def getInput(n = 4, verbose = False) -> tuple[np.ndarray, np.ndarray]:
    """Generate linearly-spaced input vector in range [0, 255] and interpolation in radians in range [0, pi/2] both of size 'n'. We assume square images ('n' should be a perfect square).

    Args:
        n (int, optional): size of input. Defaults to 4.
        verbose (bool, optional): print additional info and graphs. Graph plotting depends on additional global config (_global_plots_config_). Defaults to False.

    Returns:
        tuple[np.ndarray, np.ndarray]: (vector, angles)
    """
    input_vector = np.linspace(start=0, stop=255, num=n, dtype=int)
    input_angles = np.interp(input_vector, (0, 255), (0, np.pi/2))
    
    if verbose:
        print(input_vector,"\n", input_angles)
        plt.title('Input image')
        plt.imshow(input_vector.reshape(math.sqrt(2), math.sqrt(2)), cmap='gray')

    return input_vector, input_angles

#___________________________________
# ENCODER
def frqiEncoder(qc: QuantumCircuit, angles: np.array, measure = False, verbose = False):
    coord_q_num = int(np.ceil(math.log(len(angles), 2)))

    q = QuantumRegister(1,'q')                          # gray value
    Q = QuantumRegister(coord_q_num, 'Q')               # coords
    c = ClassicalRegister(Q.size+q.size, "c")           # measurement

    qc.add_register(q, Q, c)

    qc.id(q)
    qc.h(Q)

    controls_ = []
    for i in Q:
        controls_.append(i)

    for i, theta in enumerate(angles):
        qubit_index_bin = "{0:b}".format(i).zfill(coord_q_num)
        
        for k, qub_ind in enumerate(qubit_index_bin):
            if int(qub_ind):
                qc.x(Q[k])
                
        qc.barrier()

        # for coord_or_intns in (0,1):
        qc.mcry(theta= 2*theta, q_controls=controls_, q_target=q[0])

        qc.barrier()
        
        for k, qub_ind in enumerate(qubit_index_bin):
            if int(qub_ind):
                qc.x(Q[k])

    sv = Statevector(qc)
    
    if measure: qc.measure(list(reversed(range(qc.num_qubits))), list(range(c.size)))
    else: qc.barrier()

    return sv

#___________________________________
# INVERTER LOGIC
def invertPixels(qc: QuantumCircuit, verbose = False):
    qc.x(qc.qregs[0])
    qc.barrier()

    if verbose: print(qc)
    return qc

#___________________________________
# Measurements
def addMeasurements(qc: QuantumCircuit, verbose = False):
    qc.measure(list(reversed(range(qc.num_qubits))), list(range(qc.cregs[0].size)))
    
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
def frqiDecoder(counts, n = 4, verbose = False):
    reconstruct = []

    for i in range(n):
        if verbose: print(f"\nFor pixel at coords {i}:")
        color_list = []                         # stores the values 

        # step 1
        for key, count in counts.items():
            int_coord = int(key[1:], 2)         # all qubits but 1st will store coordinates
            
            # if coordinates match with pixel in focus, store the gray value and count
            if int_coord == i:
                color_list.append((key[0], count))        
        if verbose: print(f"\tGray value of matching states and their counts: {color_list}")

        # step 2
        zero_count = 0                          # total count for gray value = 0
        for gvalue, count in color_list:
            if not int(gvalue):                 # if gvalue == 0, get total for P(j||0>)
                zero_count = zero_count + count
        if verbose: print(f"\tTotal count where gray value is 0 = P(j||0>) = {zero_count}")

        # step 3
        try:
            gvalue = np.arccos((zero_count/sum(n for _, n in color_list))**(1/2))
            reconstruct.append(gvalue)
            if verbose: print(f"\tarccos(sqrt(zero_count / total_count)): {gvalue}")
        except ZeroDivisionError:
            print("\tZeroDivisionError")

    # step 4 (readout is reversed as we used 1st qubit for gray value instead of the last qubit)
    reconstruct = list(reversed(np.interp(reconstruct, (0, np.pi/2), (0, 255)).astype(int)))

    return reconstruct

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
def testFRQI(n=4):
    pass


if __name__ == "__main__":
    qc = QuantumCircuit()
    frqiEncoder(qc, [0, 0.5, 0.6, 0.7])
    addMeasurements(qc, True)