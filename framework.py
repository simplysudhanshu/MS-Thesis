from qiskit import QuantumRegister, ClassicalRegister, AncillaRegister, QuantumCircuit, qpy
from qiskit.quantum_info import Statevector, Operator
from qiskit_aer import Aer 
from qiskit import transpile, assemble
from qiskit.visualization import plot_histogram, plot_bloch_multivector, plot_distribution, plot_state_qsphere
import matplotlib.pyplot as plt
import numpy as np
import math
import logging
import time
import statistics

import qubit_lattice
import frqi

# setup logging
logging.basicConfig(level=logging.DEBUG, filename=f"btq_{time.strftime('%Y-%m-%d')}.log", filemode="w", format='%(asctime)s - %(levelname)s - (%(funcName)s) = %(message)s')
logger = logging.getLogger("btq_logs")

logging.getLogger('stevedore.extension').setLevel(logging.CRITICAL)
logging.getLogger('qiskit.passmanager.base_tasks').setLevel(logging.CRITICAL)
logging.getLogger('qiskit.transpiler.passes.basis.basis_translator').setLevel(logging.CRITICAL)
logging.getLogger('qiskit.compiler.transpiler').setLevel(logging.CRITICAL)
logging.getLogger('stevedore._cache').setLevel(logging.CRITICAL)
logging.getLogger('qiskit.compiler.assembler').setLevel(logging.CRITICAL)

# Configuration:

# Qiskit backend
aer_sim = Aer.get_backend('aer_simulator')

# Input runs
inputs = [4, 16]

# Shots
shots = 1000000

# class btqFramework():
#     def __init__(self, verbose="Min") -> None:
#         self.circuit = QuantumCircuit()
        
#         self.verbose = verbose
#         self.logger = logging.getLogger("btq_logs")

#     #------------------------------------
#     # ENCODING
#     def setupEncoder(self, encoderFunc: function, **kwargs):
#         self.circuit = encoderFunc(self.circuit, **kwargs)
    
#     #------------------------------------
#     # ALGORITHM
#     def setupAlgorithm(self, algorithmFunc: function, **kwargs):
#         self.circuit = algorithmFunc(self.circuit, **kwargs)

#     #------------------------------------
#     # ENCODING
#     def setupDecoder(self, encoderFunc: function, **kwargs):
#         self.circuit = encoderFunc(self.circuit, **kwargs)

#___________________________________
# INPUT
def prepareInput(n=4, input_range=(0, 255), angle_range=(0, np.pi/2), verbose=1):
    input_vector = np.linspace(start=0, stop=255, num=n, dtype=int)
    input_angles = np.interp(input_vector, input_range, angle_range) 
    
    if verbose: logger.debug(f'Inputs: size({n}), Vector: {input_vector}, Angles: {input_angles}')
    
    return input_vector, input_angles

#___________________________________
# TRANSPILE CIRCUIT
def transpileCircuit(qc: QuantumCircuit):
    return transpile(qc, aer_sim)

#___________________________________
# SIMULATE CIRCUIT
def simulate(tqc: QuantumCircuit, shots, verbose=1):
    # qobj = assemble(tqc, shots=shots)
    job = aer_sim.run(tqc, shots=shots)
    result = job.result()
    counts = result.get_counts()

    if verbose:
        logger.debug(counts)
        # display(plot_histogram(counts))
    
    return counts

#___________________________________
# Compare results
def plot_to_compare(input_vector, output_vector, verbose=1):
    if verbose: logger.debug(f'Original values: {input_vector}\tReconstructed Inverted values: {output_vector}')
    size_sqrt = math.sqrt(len(input_vector))

    plt.imshow(np.reshape(input_vector, (size_sqrt, size_sqrt)), cmap = 'gray')
    plt.title('Input Image')
    plt.show()
    plt.imshow(np.reshape(output_vector, (size_sqrt, size_sqrt)), cmap = 'gray')
    plt.title('Reconstructed Inverted Image')
    plt.show()


#___________________________________
# QUBIT LATTICE EXPERIMENT
def qubitLatticeExperiment(n=4, shots=1000000, verbose=0, run_simulation=False):
    logger.debug(f'=> Qubit Lattice Experiment:: Image size: {math.sqrt(n)} x {math.sqrt(n)}\tShots: {shots}')

    # input
    input_vector, input_angles = prepareInput(n=n, input_range=(0, 255), angle_range=(0, np.pi), verbose=verbose)

    circuit = QuantumCircuit()

    # encoing
    init_time = time.process_time()
    qubit_lattice.qubitLatticeEncoder(qc=circuit, angles=input_angles, verbose=verbose)
    logger.info(f'{{"Profiler":"Encoder", "runtime":"{time.process_time() - init_time}", "depth":"{circuit.depth()}", "width":"{circuit.num_qubits}", "Exp":"Qubit Lattice,{n},{shots}"}}')
    
    # invert + measurements
    init_time = time.process_time()
    qubit_lattice.invertPixels(qc=circuit, verbose=verbose)
    qubit_lattice.addMeasurements(qc=circuit, verbose=verbose)
    logger.info(f'{{"Profiler":"Invert + Measurement", "runtime":"{time.process_time() - init_time}", "depth":"{circuit.depth()}", "width":"{circuit.num_qubits}", "Exp":"Qubit Lattice,{n},{shots}"}}')

    
    if verbose: logger.debug(f'Total Circuit depth: {circuit.depth}\tCircuit Width: {circuit.num_qubits}')

    # transpile
    init_time = time.process_time()
    tcircuit = transpileCircuit(qc=circuit)
    logger.info(f'{{"Profiler":"Transpile", "runtime":"{time.process_time() - init_time}", "depth":"{tcircuit.depth()}", "width":"{tcircuit.num_qubits}", "Exp":"Qubit Lattice,{n},{shots}"}}')

    # store transpiled circuit
    with open(f'ql_{n}x{n}_{circuit.num_qubits}.qpy', 'wb') as f:
        qpy.dump(tcircuit, f)
    
    # run experiment
    if run_simulation:

        # load transpiled circuit
        with open(f'ql_{n}x{n}_{circuit.num_qubits}.qpy', 'rb') as fd:
            stored_tcircuit = qpy.load(fd)[0]
        
        # simulate
        init_time = time.process_time()
        experiment_result_counts = simulate(tqc=stored_tcircuit, shots=shots, verbose=verbose)
        logger.info(f'{{"Profiler":"Simulate", "runtime":"{time.process_time() - init_time}", "depth":"{tcircuit.depth()}", "width":"{tcircuit.num_qubits}", "Exp":"Qubit Lattice,{n},{shots}"}}')

        # decodde  
        output_vector = qubit_lattice.qubitLatticeDecoder(counts=experiment_result_counts, n=n, shots=shots)
        logger.debug(f'Original values: {input_vector}\tReconstructed Inverted values: {output_vector}')

        # accuracy
        logger.info(f'{{"Profiler":"Accuracy", "value":"{statistics.fmean(1 - abs(255 - (input_vector[i] + output_vector[i]))/255 for i in range(n))}", "Exp":"Qubit Lattice,{n},{shots}"}}')

        if verbose: plot_to_compare(input_vector, output_vector)


#___________________________________
# FRQI EXPERIMENT
def frqiExperiment(n=4, shots=1000000, verbose=0, run_simulation=False):
    logger.debug(f'\n=> FRQI Experiment:: Image size: {math.sqrt(n)} x {math.sqrt(n)}\tShots: {shots}')

    # input
    input_vector, input_angles = prepareInput(n=n, input_range=(0, 255), angle_range=(0, np.pi/2), verbose=verbose)

    circuit = QuantumCircuit()

    # encode
    init_time = time.process_time()
    frqi.frqiEncoder(qc=circuit, angles=input_angles, verbose=verbose)
    logger.info(f'{{"Profiler":"Encoder", "runtime":"{time.process_time() - init_time}", "depth":"{circuit.depth()}", "width":"{circuit.num_qubits}", "Exp":"FRQI,{n},{shots}"}}')
    
    # invert + measurements
    init_time = time.process_time()
    frqi.invertPixels(qc=circuit, verbose=verbose)
    frqi.addMeasurements(qc=circuit, verbose=verbose)
    logger.info(f'{{"Profiler":"Invert + Measurement", "runtime":"{time.process_time() - init_time}", "depth":"{circuit.depth()}", "width":"{circuit.num_qubits}", "Exp":"FRQI,{n},{shots}"}}')
    
    if verbose: logger.debug(f'Circuit depth: {circuit.depth()}\tCircuit Width: {circuit.num_qubits}')

    # transpile
    init_time = time.process_time()
    tcircuit = transpileCircuit(qc=circuit)
    logger.info(f'{{"Profiler":"Transpile", "runtime":"{time.process_time() - init_time}", "depth":"{tcircuit.depth()}", "width":"{tcircuit.num_qubits}", "Exp":"FRQI,{n},{shots}"}}')

    # store transpiled circuit
    with open(f'frqi_{n}x{n}_{circuit.num_qubits}.qpy', 'wb') as f:
        qpy.dump(tcircuit, f)
    
    # run experiment
    if run_simulation:

        # load transpiled circuit
        with open(f'frqi_{n}x{n}_{circuit.num_qubits}.qpy', 'rb') as fd:
            stored_tcircuit = qpy.load(fd)[0]

        # simulate
        init_time = time.process_time()
        experiment_result_counts = simulate(tqc=stored_tcircuit, shots=shots, verbose=verbose)
        logger.info(f'{{"Profiler":"Simulate", "runtime":"{time.process_time() - init_time}", "depth":"{tcircuit.depth()}", "width":"{tcircuit.num_qubits}", "Exp":"FRQI,{n},{shots}"}}')
        
        # decode
        output_vector = frqi.frqiDecoder(counts=experiment_result_counts, n=n)
        logger.debug(f'Original values: {input_vector}\tReconstructed Inverted values: {output_vector}')
        
        # accuracy
        logger.info(f'{{"Profiler":"Accuracy", "value":"{statistics.fmean(1 - abs(255 - (input_vector[i] + output_vector[i]))/255 for i in range(n))}", "Exp":"FRQI,{n},{shots}"}}')

        if verbose: plot_to_compare(input_vector, output_vector)


#___________________________________
if __name__ == "__main__":
    print(f"BTQ - Trial runs\n")

    for input in inputs:
        print(f"Running for input: {input}")

        print("\t1/2", end='\r')
        try:
            init_time = time.process_time()
            qubitLatticeExperiment(n=input, run_simulation=True)
            logger.info(f'{{"Profiler":"Algorithm Runtime", "runtime":"{time.process_time() - init_time}","Exp":"Qubit Lattice,{input},{shots}"}}')

        except:
            logger.error(f'Error in Qubit Lattice Experiment (input: {input})', exc_info=True)

        print("\t2/2", end='\r')
        try:
            init_time = time.process_time()
            frqiExperiment(n=input, run_simulation=True)
            logger.info(f'{{"Profiler":"Algorithm Runtime", "runtime":"{time.process_time() - init_time}","Exp":"FRQI,{input},{shots}"}}')

        except:
            logger.error(f'Error in FRQI Experiment (input: {input})', exc_info=True)
        print("\tdone")
