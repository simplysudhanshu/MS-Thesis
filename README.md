# From Bits to Qubits: Challenges in Classical-Quantum Integration

Masters Thesis for Master of Science degree in Computer Science at San Francisco State University.

### Abstract
Quantum computing has the potential to revolutionize the technology landscape by tackling complex problems that are presently intractable for classical computers. Before these advanced computational resources can be employed to solve real-world problems, it is important to understand the interplay between classical and quantum computing. Quantum encoding is crucial in this phase because it allows classical information to be transformed into a quantum state, enabling it to be processed within a quantum computing system. This study closely examines a few such quantum encoding models- the Phase Encoding algorithm, the Qubit Lattice model, and the Flexible Representation of Quantum Images (FRQI). The aim of quantifying their different characteristics is to analyze their impact on quantum processing workflows. The comparative analysis may provide insights into the limitations, performance, and resource requirements of these models.

> [$\rightarrow$ Access to full thesis at ScholarWorks.](http://hdl.handle.net/20.500.12680/b8515w61h)

---

### Project Structure
[The Notebooks folder](Notebooks) contains a jupyter notebook for each of the encoding techniques discussed in the study and a notebook summarizing all three with an example of 2x2 grayscale image.

In the root folder, the main driver code rests in [framework.py](framework.py). It takes command line arguments to control the experiment like the encoding technique, number of shots, input image and invoking actual IBMQ backend. Logs and visualizations are handled by [btq_plotter.py](btq_plotter.py) which will create two folders in the root directory - one for storing logs, circuits and results, another one for graphs.