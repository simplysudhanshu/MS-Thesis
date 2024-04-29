from qiskit import QuantumRegister, ClassicalRegister, AncillaRegister, QuantumCircuit, qpy
from qiskit.quantum_info import Statevector, Operator
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit import transpile, assemble
from qiskit.visualization import plot_histogram, plot_bloch_multivector, plot_distribution, plot_state_qsphere
from qiskit_ibm_runtime import QiskitRuntimeService, Batch, Sampler
from qiskit.result.utils import marginal_distribution
from qiskit_ibm_runtime.fake_provider import FakeMumbai

import matplotlib.pyplot as plt
import numpy as np
import math
import logging
import time
import statistics
import os
import pickle
import sys

import qubit_lattice
import phase
import frqi
import btq_plotter
import supermarq_metrics

# setup logging
os.makedirs("./experiment_data", exist_ok=True)

import logging.config
logging.config.dictConfig({
    'version': 1,
    # Other configs ...
    'disable_existing_loggers': True
})

logging.basicConfig(level=logging.DEBUG, filename=os.path.join("experiment_data", f"btq_{time.strftime('%Y-%m-%d')}.log"), filemode="w", format='%(asctime)s - %(levelname)s - (%(funcName)s) = %(message)s')
logger = logging.getLogger("btq_logs")

# logging.getLogger('stevedore.extension').setLevel(logging.CRITICAL)
# logging.getLogger('qiskit.passmanager.base_tasks').setLevel(logging.CRITICAL)
# logging.getLogger('qiskit.transpiler.passes.basis.basis_translator').setLevel(logging.CRITICAL)
# logging.getLogger('qiskit.compiler.transpiler').setLevel(logging.CRITICAL)
# logging.getLogger('stevedore._cache').setLevel(logging.CRITICAL)
# logging.getLogger('qiskit.compiler.assembler').setLevel(logging.CRITICAL)

# Configuration:

# Qiskit backend basics
qiskitService = None
pure_backend = AerSimulator()
noisy_backend = AerSimulator()
ibmq_backend = None

#___________________________________
# Qiskit Backends
''' Noisy backend'''
def setupNoisyBackend():
    qiskitService = QiskitRuntimeService(channel="ibm_quantum", token="b58f04be6f1a8412295c8b59c3e19af21d1c9501d28ca176687d5c3f49338f0211c08461348a978c07dcecc76de10675f84deb366634b56c3a8ac0514085be5c")

    ''' Noisy model from AER: https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.noise.NoiseModel.html '''
    # Get a fake backend from the fake provider
    # backend = FakeMumbai()
    # noise_model = NoiseModel.from_backend(backend)

    # # Get coupling map from backend
    # coupling_map = backend.configuration().coupling_map

    # # Get basis gates from noise model
    # basis_gates = noise_model.basis_gates

    # noisy_backend = AerSimulator(noise_model=noise_model,
    #                        coupling_map=coupling_map,
    #                        basis_gates=basis_gates)

    ''' Noisy model from QiskitRuntimeService: https://docs.quantum.ibm.com/api/qiskit-ibm-runtime/dev/fake_provider '''
    noisy_backend = qiskitService.get_backend('ibm_kyoto')
    noisy_backend = AerSimulator.from_backend(noisy_backend)

''' IBMQ Hardware '''
def setupIBMQBackend():
    qiskitService = QiskitRuntimeService(channel="ibm_quantum", token="b58f04be6f1a8412295c8b59c3e19af21d1c9501d28ca176687d5c3f49338f0211c08461348a978c07dcecc76de10675f84deb366634b56c3a8ac0514085be5c")
    # To run on hardware, select the backend with the fewest number of jobs in the queue
    ibmq_backend = qiskitService.least_busy(operational=True, simulator=False)

#___________________________________
# INPUT
def prepareInput(n=4, input_range=(0, 255), angle_range=(0, np.pi/2), dist="linear", verbose=1):
    side = int(math.sqrt(n))
    if dist.lower() == "random":
        input_vector = np.random.uniform(low=0, high=255, size=n, dtype=int)
    
    elif dist.lower() == "reversing":
        input_vector = []
        init_vector = np.linspace(start=0, stop=255, num=n, dtype=int)

        for i in range(side):
            input_vector.extend(init_vector[i*side:i*side+side] if not i%2 else init_vector[i*side+side-1:i*side-1:-1])
    else:
        input_vector = np.linspace(start=0, stop=255, num=n, dtype=int)

    input_angles = np.interp(input_vector, input_range, angle_range) 
    
    if verbose: logger.debug(f'Inputs: size({n}), Vector: {input_vector}, Angles: {input_angles}')
    
    return input_vector, input_angles

#___________________________________
# TRANSPILE CIRCUIT
def transpileCircuit(qc: QuantumCircuit, noisy=False, backend="simulator"):
    if backend == "simulator":
        if noisy:
            return transpile(qc, noisy_backend, basis_gates=['u', 'cx'])
        else:
            return transpile(qc, pure_backend, basis_gates=['u', 'cx'])
    
    elif backend == "ibmq":
        return transpile(qc, backend=ibmq_backend, optimization_level=3, seed_transpiler=0)

#___________________________________
# SIMULATE CIRCUIT
def simulate(tqc: QuantumCircuit, shots, noisy=False, verbose=1, backend="simulator"):
    if backend == "simulator":
        if noisy:
            job = noisy_backend.run(tqc, shots=shots)
        else:    
            job = pure_backend.run(tqc, shots=shots)

            result = job.result()

    # https://learning.quantum.ibm.com/tutorial/submit-transpiled-circuits#step-3-execute-using-qiskit-primitives
    # https://docs.quantum.ibm.com/api/migration-guides/v2-primitives#steps-to-migrate-to-sampler-v2
    elif backend == "ibmq":
        with Batch(service=qiskitService, backend=ibmq_backend):
            sampler = Sampler()
            job = sampler.run(
                circuits=tqc,
                skip_transpilation=True,
                shots=10000,
            )
            result = job.result()

            print(result)

            with open(os.path.join("experiment_data", f"exp_{time.strftime('%Y-%m-%d')}_{ibmq_backend.name}_{job.job_id()}.pkl"), 'wb') as f:
                pickle.dump(result, f)

    counts = result.get_counts()

    if verbose:
        logger.debug(counts)
        # display(plot_histogram(counts))
    
    return counts

#___________________________________
# SIMULATE CIRCUIT
def simulate_stateVec(qc: QuantumCircuit, verbose=1):
    job = noisy_backend.run(qc, shots=shots)

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
def qubitLatticeExperiment(n=4, shots=1000000, verbose=0, run_simulation=False, exp_dict=None, noisy=False, dist="linear"):
    logger.debug(f'> Qubit Lattice Experiment:: Image size: {math.sqrt(n)} x {math.sqrt(n)}\tShots: {shots} (noisy={noisy})')

    # input
    input_vector, input_angles = prepareInput(n=n, input_range=(0, 255), angle_range=(0, np.pi), verbose=verbose, dist=dist)

    circuit = QuantumCircuit()

    # encoding
    init_time = time.process_time()
    qubit_lattice.qubitLatticeEncoder(qc=circuit, angles=input_angles, verbose=verbose)
    
    logger.info(f'{{"Profiler":"Encoder", "runtime":"{time.process_time() - init_time}", "depth":"{circuit.depth()}", "width":"{circuit.num_qubits}", "Exp":"Qubit Lattice,{n},{shots}"}}')
    if exp_dict:
        if noisy: exp_dict["runtimes"]["Noisy Encoder"].append(time.process_time() - init_time)
        else: 
            exp_dict["runtimes"]["Encoder"].append(time.process_time() - init_time)
            exp_dict["depths"]["Encoder"].append(circuit.depth())
            exp_dict["widths"].append(circuit.num_qubits)

    # invert + measurements
    init_time = time.process_time()
    qubit_lattice.invertPixels(qc=circuit, verbose=verbose)
    qubit_lattice.addMeasurements(qc=circuit, verbose=verbose)
    
    logger.info(f'{{"Profiler":"Invert + Measurement", "runtime":"{time.process_time() - init_time}", "depth":"{circuit.depth()}", "width":"{circuit.num_qubits}", "Exp":"Qubit Lattice,{n},{shots}"}}')
    if exp_dict:
        if noisy: exp_dict["runtimes"]["Noisy Invert + Measurement"].append(time.process_time() - init_time)
        else: 
            exp_dict["runtimes"]["Invert + Measurement"].append(time.process_time() - init_time)
            exp_dict["depths"]["Invert + Measurement"].append(circuit.depth())
    
    if verbose: logger.debug(f'Total Circuit depth: {circuit.depth}\tCircuit Width: {circuit.num_qubits}')

    # transpile
    init_time = time.process_time()
    tcircuit = transpileCircuit(qc=circuit, noisy=noisy)

    logger.info(f'{{"Profiler":"Transpile", "runtime":"{time.process_time() - init_time}", "depth":"{tcircuit.depth()}", "width":"{tcircuit.num_qubits}", "count_ops":"{tcircuit.count_ops()}", "Exp":"Qubit Lattice,{n},{shots}"}}')
    if exp_dict:
        if noisy: exp_dict["runtimes"]["Noisy Transpile"].append(time.process_time() - init_time)
        else: 
            exp_dict["runtimes"]["Transpile"].append(time.process_time() - init_time)
            exp_dict["depths"]["Transpile"].append(circuit.depth())

    # store transpiled circuit
    with open(os.path.join('experiment_data', f'ql_{n}x{n}_{circuit.num_qubits}.qpy'), 'wb') as f:
        qpy.dump(tcircuit, f)
    
    # run experiment
    if run_simulation:

        # load transpiled circuit
        with open(os.path.join('experiment_data', f'ql_{n}x{n}_{circuit.num_qubits}.qpy'), 'rb') as f:
            stored_tcircuit = qpy.load(f)[0]
        
        # simulate
        init_time = time.process_time()
        experiment_result_counts = simulate(tqc=stored_tcircuit, shots=shots, verbose=verbose)

        logger.info(f'{{"Profiler":"Simulate", "runtime":"{time.process_time() - init_time}", "depth":"{tcircuit.depth()}", "width":"{tcircuit.num_qubits}", "Exp":"Qubit Lattice,{n},{shots}"}}')
        if exp_dict:
            if noisy: exp_dict["runtimes"]["Noisy Simulate"].append(time.process_time() - init_time)
            else: 
                exp_dict["runtimes"]["Simulate"].append(time.process_time() - init_time)
                exp_dict["depths"]["Simulate"].append(circuit.depth())

        # decode
        init_time = time.process_time()  
        output_vector = qubit_lattice.qubitLatticeDecoder(counts=experiment_result_counts, n=n, shots=shots)
        
        logger.info(f'{{"Profiler":"Decoder", "runtime":"{time.process_time() - init_time}", "Exp":"Qubit Lattice,{n},{shots}"}}')
        if exp_dict:
            if noisy: exp_dict["runtimes"]["Noisy Decoder"].append(time.process_time() - init_time)
            else: exp_dict["runtimes"]["Decoder"].append(time.process_time() - init_time)
        
        # data points
        logger.info(f'{{"Profiler":"Data Points", "original_values": {list(input_vector)}, "reconstructed_values": {output_vector}}}')
        if exp_dict:
            if noisy: exp_dict['noisy_data_points'].append([list(input_vector), list(output_vector)])
            else: exp_dict['data_points'].append([list(input_vector), list(output_vector)])

        # accuracy
        accuracy = statistics.fmean([1 - round(abs(output_vector[i] - (255 - input_vector[i]))/max((255 - input_vector[i]), output_vector[i]),4) if (255-input_vector[i]) != output_vector[i] else 1 for i in range(n)])
        logger.info(f'{{"Profiler":"Accuracy", "value":"{accuracy}", "Exp":"Qubit Lattice,{n},{shots}"}}')

        if exp_dict:
            if noisy: exp_dict['noisy_accuracy'].append(accuracy)
            else: exp_dict['accuracy'].append(accuracy)

        if verbose: plot_to_compare(input_vector, output_vector)
    
    return exp_dict, circuit, accuracy


#___________________________________
# PHASE EXPERIMENT
def phaseExperiment(n=4, shots=1000000, verbose=0, run_simulation=False, exp_dict=None, noisy=False, dist="linear"):
    logger.debug(f'> Qubit Lattice Experiment:: Image size: {math.sqrt(n)} x {math.sqrt(n)}\tShots: {shots} (noisy={noisy})')
    
    # input
    input_vector, input_angles = prepareInput(n=n, input_range=(0, 255), angle_range=(0, np.pi), verbose=verbose, dist=dist)

    circuit = QuantumCircuit()

    # encoding
    init_time = time.process_time()
    phase.phaseEncoder(qc=circuit, angles=input_angles, verbose=verbose)
    
    logger.info(f'{{"Profiler":"Encoder", "runtime":"{time.process_time() - init_time}", "depth":"{circuit.depth()}", "width":"{circuit.num_qubits}", "Exp":"Phase,{n},{shots}"}}')
    if noisy: exp_dict["runtimes"]["Noisy Encoder"].append(time.process_time() - init_time)
    else: 
        exp_dict["runtimes"]["Encoder"].append(time.process_time() - init_time)
        exp_dict["depths"]["Encoder"].append(circuit.depth())
        exp_dict["widths"].append(circuit.num_qubits)

    # invert + measurements
    init_time = time.process_time()
    phase.invertPixels(qc=circuit, verbose=verbose)
    phase.addMeasurements(qc=circuit, verbose=verbose)
    
    logger.info(f'{{"Profiler":"Invert + Measurement", "runtime":"{time.process_time() - init_time}", "depth":"{circuit.depth()}", "width":"{circuit.num_qubits}", "Exp":"Phase,{n},{shots}"}}')
    if exp_dict:
        if noisy: exp_dict["runtimes"]["Noisy Invert + Measurement"].append(time.process_time() - init_time)
        else: 
            exp_dict["runtimes"]["Invert + Measurement"].append(time.process_time() - init_time)
            exp_dict["depths"]["Invert + Measurement"].append(circuit.depth())
    
    if verbose: logger.debug(f'Total Circuit depth: {circuit.depth}\tCircuit Width: {circuit.num_qubits}')

    # transpile
    init_time = time.process_time()
    tcircuit = transpileCircuit(qc=circuit, noisy=noisy)

    logger.info(f'{{"Profiler":"Transpile", "runtime":"{time.process_time() - init_time}", "depth":"{tcircuit.depth()}", "width":"{tcircuit.num_qubits}", "count_ops":"{tcircuit.count_ops()}", "Exp":"Phase,{n},{shots}"}}')
    if exp_dict:
        if noisy: exp_dict["runtimes"]["Noisy Transpile"].append(time.process_time() - init_time)
        else: 
            exp_dict["runtimes"]["Transpile"].append(time.process_time() - init_time)
            exp_dict["depths"]["Transpile"].append(circuit.depth())

    # store transpiled circuit
    with open(os.path.join('experiment_data', f'ql_{n}x{n}_{circuit.num_qubits}.qpy'), 'wb') as f:
        qpy.dump(tcircuit, f)
    
    # run experiment
    if run_simulation:

        # load transpiled circuit
        with open(os.path.join('experiment_data', f'ql_{n}x{n}_{circuit.num_qubits}.qpy'), 'rb') as f:
            stored_tcircuit = qpy.load(f)[0]
        
        # simulate
        init_time = time.process_time()
        experiment_result_counts = simulate(tqc=stored_tcircuit, shots=shots, verbose=verbose)

        logger.info(f'{{"Profiler":"Simulate", "runtime":"{time.process_time() - init_time}", "depth":"{tcircuit.depth()}", "width":"{tcircuit.num_qubits}", "Exp":"Phase,{n},{shots}"}}')
        if exp_dict:
            if noisy: exp_dict["runtimes"]["Noisy Simulate"].append(time.process_time() - init_time)
            else: 
                exp_dict["runtimes"]["Simulate"].append(time.process_time() - init_time)
                exp_dict["depths"]["Simulate"].append(circuit.depth())

        # decode
        init_time = time.process_time()  
        output_vector = phase.phaseDecoder(counts=experiment_result_counts, n=n, shots=shots)
        
        logger.info(f'{{"Profiler":"Decoder", "runtime":"{time.process_time() - init_time}", "Exp":"Phase,{n},{shots}"}}')
        if exp_dict:
            if noisy: exp_dict["runtimes"]["Noisy Decoder"].append(time.process_time() - init_time)
            else: exp_dict["runtimes"]["Decoder"].append(time.process_time() - init_time)
        
        # data points
        logger.info(f'{{"Profiler":"Data Points", "original_values": {list(input_vector)}, "reconstructed_values": {output_vector}}}')
        if exp_dict:
            if noisy: exp_dict['noisy_data_points'].append([list(input_vector), list(output_vector)])
            else: exp_dict['data_points'].append([list(input_vector), list(output_vector)])

        # accuracy
        accuracy = statistics.fmean([1 - round(abs(output_vector[i] - (255 - input_vector[i]))/max((255 - input_vector[i]), output_vector[i]),4) if (255-input_vector[i]) != output_vector[i] else 1 for i in range(n)])
        logger.info(f'{{"Profiler":"Accuracy", "value":"{accuracy}", "Exp":"Phase,{n},{shots}"}}')

        if exp_dict:
            if noisy: exp_dict['noisy_accuracy'].append(accuracy)
            else: exp_dict['accuracy'].append(accuracy)

        if verbose: plot_to_compare(input_vector, output_vector)
    
    return exp_dict, circuit, accuracy

#___________________________________
# FRQI EXPERIMENT
def frqiExperiment(n=4, shots=1000000, verbose=0, run_simulation=False, exp_dict=None, noisy=False, dist="linear", backend="simulator"):
    """_summary_

    Args:
        n (int, optional): _description_. Defaults to 4.
        shots (int, optional): _description_. Defaults to 1000000.
        verbose (int, optional): _description_. Defaults to 0.
        run_simulation (bool, optional): _description_. Defaults to False.
        exp_dict (_type_, optional): _description_. Defaults to None.
        noisy (bool, optional): _description_. Defaults to False.
        dist (str, optional): _description_. Defaults to "linear".
        backend (str, optional): _description_. Defaults to "simulator".

    Returns:
        _type_: _description_
    """
    logger.debug(f'> FRQI Experiment:: Image size: {math.sqrt(n)} x {math.sqrt(n)}\tShots: {shots} (noisy={noisy}, backend={backend})')

    # input
    input_vector, input_angles = prepareInput(n=n, input_range=(0, 255), angle_range=(0, np.pi/2), dist=dist, verbose=verbose)

    # citcuit
    circuit = QuantumCircuit()

    # encode
    init_time = time.process_time()
    frqi.frqiEncoder(qc=circuit, angles=input_angles, verbose=verbose)
    
    logger.info(f'{{"Profiler":"Encoder", "runtime":"{time.process_time() - init_time}", "depth":"{circuit.depth()}", "width":"{circuit.num_qubits}", "Exp":"FRQI,{n},{shots}"}}')
    if exp_dict:
        if noisy: exp_dict["runtimes"]["Noisy Encoder"].append(time.process_time() - init_time)
        else: 
            exp_dict["runtimes"]["Encoder"].append(time.process_time() - init_time)
            exp_dict["depths"]["Encoder"].append(circuit.depth())
            exp_dict["widths"].append(circuit.num_qubits)

    # invert + measurements
    init_time = time.process_time()
    frqi.invertPixels(qc=circuit, verbose=verbose)
    frqi.addMeasurements(qc=circuit, verbose=verbose)
    
    logger.info(f'{{"Profiler":"Invert + Measurement", "runtime":"{time.process_time() - init_time}", "depth":"{circuit.depth()}", "width":"{circuit.num_qubits}", "Exp":"FRQI,{n},{shots}"}}')
    if exp_dict:
        if noisy: exp_dict["runtimes"]["Noisy Invert + Measurement"].append(time.process_time() - init_time)
        else: 
            exp_dict["runtimes"]["Invert + Measurement"].append(time.process_time() - init_time)
            exp_dict["depths"]["Invert + Measurement"].append(circuit.depth())

    if verbose: logger.debug(f'Circuit depth: {circuit.depth()}\tCircuit Width: {circuit.num_qubits}')

    # transpile
    init_time = time.process_time()
    tcircuit = transpileCircuit(qc=circuit, noisy=noisy, backend=backend)
    
    logger.info(f'{{"Profiler":"Transpile", "runtime":"{time.process_time() - init_time}", "depth":"{tcircuit.depth()}", "width":"{tcircuit.num_qubits}", "count_ops":"{tcircuit.count_ops()}", "Exp":"FRQI,{n},{shots}"}}')
    if exp_dict:
        if noisy: exp_dict["runtimes"]["Noisy Transpile"].append(time.process_time() - init_time)
        else: 
            exp_dict["runtimes"]["Transpile"].append(time.process_time() - init_time)
            exp_dict["depths"]["Transpile"].append(circuit.depth())
        
    # store transpiled circuit
    with open(os.path.join('experiment_data', f'frqi_{n}x{n}_{circuit.num_qubits}.qpy'), 'wb') as f:
        qpy.dump(tcircuit, f)
    
    # run experiment
    if run_simulation:

        # load transpiled circuit
        with open(os.path.join('experiment_data', f'frqi_{n}x{n}_{circuit.num_qubits}.qpy'), 'rb') as f:
            stored_tcircuit = qpy.load(f)[0]

        # simulate
        init_time = time.process_time()
        experiment_result_counts = simulate(tqc=tcircuit, shots=shots, verbose=verbose, backend=backend)
        
        logger.info(f'{{"Profiler":"Simulate", "runtime":"{time.process_time() - init_time}", "depth":"{tcircuit.depth()}", "width":"{tcircuit.num_qubits}", "Exp":"FRQI,{n},{shots}"}}')
        if exp_dict:
            if noisy: exp_dict["runtimes"]["Noisy Simulate"].append(time.process_time() - init_time)
            else: 
                exp_dict["runtimes"]["Simulate"].append(time.process_time() - init_time)
                exp_dict["depths"]["Simulate"].append(circuit.depth())

        # decode
        init_time = time.process_time()
        output_vector = frqi.frqiDecoder(counts=experiment_result_counts, n=n)
        
        logger.info(f'{{"Profiler":"Decoder", "runtime":"{time.process_time() - init_time}", "Exp":"FRQI,{n},{shots}"}}')
        if exp_dict:
            if noisy: exp_dict["runtimes"]["Noisy Decoder"].append(time.process_time() - init_time)
            else: exp_dict["runtimes"]["Decoder"].append(time.process_time() - init_time)
        
        # data points
        logger.info(f'{{"Profiler":"Data Points", "original_values": {list(input_vector)}, "reconstructed_values": {output_vector}}}')
        if exp_dict:
            if noisy: exp_dict['noisy_data_points'].append([list(input_vector), list(output_vector)])
            else: exp_dict['data_points'].append([list(input_vector), list(output_vector)])
        
        # accuracy
        accuracy = statistics.fmean([1 - round(abs(output_vector[i] - (255 - input_vector[i]))/max((255 - input_vector[i]), output_vector[i]),4) if (255-input_vector[i]) != output_vector[i] else 1 for i in range(n)])
        logger.info(f'{{"Profiler":"Accuracy", "value":"{accuracy}", "Exp":"FRQI,{n},{shots}"}}')

        if exp_dict:
            if noisy: exp_dict['noisy_accuracy'].append(accuracy)
            else: exp_dict['accuracy'].append(accuracy)
        
        if verbose: plot_to_compare(input_vector, output_vector)

    return exp_dict, circuit, accuracy

#___________________________________
# DEFAULTS:
# Input runs
power_inputs = lambda n: [(2**x)**2 for x in range(1, n+1)]
square_inputs = lambda n: [x**2 for x in range(2, n+1)]

# Shots
shots = 50000

# input distribution ["reversing", "random", "linear"]
dist = "reversing"

# experiments: ["all", "ql", "phase", "frqi", "ibmq", "shots", "backends"]
experiments = "all"

# THE MAIN
#___________________________________
if __name__ == "__main__":
    
    # cmd arguments
    if len(sys.argv) > 1 and sys.argv[1] in ["all", "ql", "phase", "frqi", "ibmq", "shots"]: 
        experiments = sys.argv[1]

    if len(sys.argv) > 2: 
        shots = sys.argv[2]

    if len(sys.argv) > 3 and sys.argv[3] in ["reversing", "random", "linear"]: 
        dist = sys.argv[2]

    if experiments in ["all", "ql", "ph", "frqi"]: 
        setupNoisyBackend()
        ql_ph_inputs = square_inputs(5)
        frqi_inputs = power_inputs(4)
        exp_list = []

    print(f"\n- BTQ - Trial runs\t[{shots if experiments != "shots" else '[5000, ..., 100000]'} shots - {dist} input - {experiments} experiments]\n")

    if experiments in ["all", "ql"]:
        #----------------------------------
        print(f"Qubit Lattice Experiments")

        exp = btq_plotter.get_dict("exp")
        exp['name'] = "Qubit Lattice"

        for i, input in enumerate(ql_ph_inputs):        
            print("\033[K", f"\t{i+1}/{len(ql_ph_inputs)} - {input}", end='\r')

            try:
                exp['shots'].append(shots)
                exp['size'].append(input)

                # Pure
                init_time = time.process_time()
                exp, circuit, accuracy = qubitLatticeExperiment(n=input, run_simulation=True, exp_dict=exp, noisy=False, dist=dist, shots=shots)

                logger.info(f'{{"Profiler":"Algorithm Runtime", "runtime":"{time.process_time() - init_time}","Exp":"Qubit Lattice,{input},{shots}"}}')
                exp["runtimes"]["Algorithm Runtime"].append(time.process_time() - init_time)
                
                supermarq_list = supermarq_metrics.compute_all(qc=circuit)
                logger.info(f'{{"Profiler":"SupermarQ", "metrics":"{supermarq_list}","Exp":"Qubit Lattics,{input},{shots}"}}')
                exp['supermarq_metrics'].append(supermarq_list)

                # Noisy
                init_time = time.process_time()
                exp, circuit, accuracy = qubitLatticeExperiment(n=input, run_simulation=True, exp_dict=exp, noisy=True, dist=dist, shots=shots)
                
                logger.info(f'{{"Profiler":"Algorithm Runtime", "runtime":"{time.process_time() - init_time}","Exp":"Qubit Lattice,{input},{shots}"}}')
                exp["runtimes"]["Noisy Algorithm Runtime"].append(time.process_time() - init_time)

            except:
                logger.error(f'Error in Qubit Lattice Experiment (input: {input})', exc_info=True)
        
        # save experiments dict
        with open(os.path.join("experiment_data", f"ql_{time.strftime('%Y-%m-%d')}.pkl"), 'wb') as f:
            pickle.dump(exp, f)
        
        exp_list.append(exp)
        
        # save plots
        btq_plotter.plot(exp=exp, save=True)
    
    if experiments in ["all", "ph"]:
        #----------------------------------
        print(f"Phase Experiments")

        exp = btq_plotter.get_dict("exp")
        exp['name'] = "Phase"

        for i, input in enumerate(ql_ph_inputs):        
            print("\033[K", f"\t{i+1}/{len(ql_ph_inputs)} - {input}", end='\r')

            try:
                exp['shots'].append(shots)
                exp['size'].append(input)

                # Pure
                init_time = time.process_time()
                exp, circuit, accuracy = phaseExperiment(n=input, run_simulation=True, exp_dict=exp, noisy=False, dist=dist, shots=shots)

                logger.info(f'{{"Profiler":"Algorithm Runtime", "runtime":"{time.process_time() - init_time}","Exp":"Phase,{input},{shots}"}}')
                exp["runtimes"]["Algorithm Runtime"].append(time.process_time() - init_time)
                
                supermarq_list = supermarq_metrics.compute_all(qc=circuit)
                logger.info(f'{{"Profiler":"SupermarQ", "metrics":"{supermarq_list}","Exp":"Phase,{input},{shots}"}}')
                exp['supermarq_metrics'].append(supermarq_list)
                
                # Noisy
                init_time = time.process_time()
                exp, circuit, accuracy = phaseExperiment(n=input, run_simulation=True, exp_dict=exp, noisy=True, dist=dist, shots=shots)
                
                logger.info(f'{{"Profiler":"Algorithm Runtime", "runtime":"{time.process_time() - init_time}","Exp":"Phase,{input},{shots}"}}')
                exp["runtimes"]["Noisy Algorithm Runtime"].append(time.process_time() - init_time)

            except:
                logger.error(f'Error in Phase Experiment (input: {input})', exc_info=True)
        
        # save experiments dict
        with open(os.path.join("experiment_data", f"ph_{time.strftime('%Y-%m-%d')}.pkl"), 'wb') as f:
            pickle.dump(exp, f)
        
        exp_list.append(exp)
        
        # save plots
        btq_plotter.plot(exp=exp, save=True)
    
    if experiments in ["all", "frqi"]:
        #----------------------------------
        print(f"FRQI Experiments")

        exp = btq_plotter.get_dict("exp")
        exp['name'] = "FRQI"

        for i, input in enumerate(frqi_inputs):
            
            print("\033[K", f"\t{i+1}/{len(frqi_inputs)} - {input}", end='\r')

            try:
                exp['shots'].append(shots)
                exp['size'].append(input)

                # Pure
                init_time = time.process_time()
                exp, circuit, accuracy = frqiExperiment(n=input, run_simulation=True, exp_dict=exp, noisy=False, dist=dist, shots=shots)

                logger.info(f'{{"Profiler":"Algorithm Runtime", "runtime":"{time.process_time() - init_time}","Exp":"FRQI,{input},{shots}"}}')
                exp["runtimes"]["Algorithm Runtime"].append(time.process_time() - init_time)

                supermarq_list = supermarq_metrics.compute_all(qc=circuit)
                logger.info(f'{{"Profiler":"SupermarQ", "metrics":"{supermarq_list}","Exp":"FRQI,{input},{shots}"}}')
                exp['supermarq_metrics'].append(supermarq_list)

                # Noisy
                init_time = time.process_time()
                exp, circuit, accuracy = frqiExperiment(n=input, run_simulation=True, exp_dict=exp, noisy=True, dist=dist, shots=shots)
                
                logger.info(f'{{"Profiler":"Algorithm Runtime", "runtime":"{time.process_time() - init_time}","Exp":"FRQI,{input},{shots}"}}')
                exp["runtimes"]["Noisy Algorithm Runtime"].append(time.process_time() - init_time)

                # IBMQ
                # exp = frqiExperiment(n=input, run_simulation=True, exp_dict=exp, noisy=False, dist=dist, backend="ibmq")

            except:
                logger.error(f'Error in FRQI Experiment (input: {input})', exc_info=True)

        # save experiments dict
        with open(os.path.join("experiment_data", f"frqi_{time.strftime('%Y-%m-%d')}.pkl"), 'wb') as f:
            pickle.dump(exp, f)

        exp_list.append(exp)
        
        # save plots
        btq_plotter.plot(exp=exp, save=True)

        with open(os.path.join("experiment_data", f"exp_{time.strftime('%Y-%m-%d')}.pkl"), 'wb') as f:
            pickle.dump(exp_list, f)

    if experiments in ["all", "shots"]:
        #----------------------------------
        print(f"FRQI - Shots Experiments")
        
        shots_dict = btq_plotter.get_dict("shots")

        for i, shot in enumerate(sorted(set([5000, 10000, 25000, 50000, 75000, 100000] + [shots]))):
            
            print("\033[K", f"\t{i+1}/{len([5000, 10000, 25000, 50000, 75000, 100000] + [shots])} - {shot}", end='\r')

            try:
                shots_dict['shots'].append(shot)

                init_time = time.process_time()
                _, __, accuracy = frqiExperiment(n=256, shots=shot, run_simulation=True, noisy=False, dist=dist)
                shots_dict["accuracy"].append(accuracy)
                shots_dict["runtimes"].append(time.process_time() - init_time)
                
                logger.info(f'{{"Profiler":"Algorithm Runtime", "runtime":"{time.process_time() - init_time}","Exp":"FRQI_shots,256,{shot}"}}')

            except:
                logger.error(f'Error in FRQI Shots Experiment (shot: {shot})', exc_info=True)
        
        # save experiments dict
        with open(os.path.join("experiment_data", f"frqi_shots_{time.strftime('%Y-%m-%d')}.pkl"), 'wb') as f:
            pickle.dump(shots_dict, f)
        
        # save plots
        btq_plotter.plot(shots_dict=shots_dict)

    # =========================
    elif experiments == "ibmq":
        setupIBMQBackend()
    