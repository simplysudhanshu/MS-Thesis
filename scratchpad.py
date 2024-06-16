from qiskit_ibm_runtime import QiskitRuntimeService, Batch, Sampler
import os
import sys
import pickle
import time
import math
import numpy as np

exp_list = []

for pickled_exp in sys.argv[1:]:   
    print(f"Added {pickled_exp}")
    
    with open(os.path.join("experiment_data", pickled_exp), 'rb') as f:
        exp_list.append(pickle.load(f))

with open(os.path.join("experiment_data", f"exp_{time.strftime('%Y-%m-%d')}.pkl"), 'wb') as f:
    pickle.dump(exp_list, f)



# job_id = 'crvmwxvy7jt000807jgg'

# with open('ibmq.token', 'r') as f:
#     ibmqt = str(f.read())

# qiskitService = QiskitRuntimeService(channel="ibm_quantum", token=ibmqt)
# backend = qiskitService.least_busy(operational=True, simulator=False)

# retrieved_job = qiskitService.job(job_id)
# result = retrieved_job.result()

# print(f"\nJOB: \n{retrieved_job.__dict__}")
# print(f"\nRESULT: \n{result}")
# print(f"\nRESULT COUNTS: \n{result.get_counts()}")


# with open(os.path.join("experiment_data", f"{sys.argv[1]}"), 'rb') as f:
#     exp = pickle.load(f)
    
# print(exp)

# 'Simulate': [6.20547453, 14.331503102000003, 35.958899162], 'Decoder': [0.0, 0.0, 0.0], 'Algorithm Runtime': []} [0.594425, 0.5724812499999999, 0.59219375]
# 'Algorithm Runtime': [0.0424649715423584, 1.2623794078826904, 4.3320677280426025, 30.2229106426239]
# 'accuracy': [1.0, 0.97703125, 0.9013390625, 0.820845703125]