'''
Adapted from:
https://github.com/Infleqtion/client-superstaq/blob/main/supermarq-benchmarks/supermarq/converters.py

SuperMarQ paper:
https://arxiv.org/pdf/2202.11045.pdf
'''
import qiskit
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
from typing import Any
import numpy as np
import numpy.typing as npt
from matplotlib.patches import Circle
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes

def compute_communication_with_qiskit(circuit: qiskit.QuantumCircuit) -> float:
    """Compute the program communication of the given quantum circuit.

    Program communication = circuit's average qubit degree / degree of a complete graph.

    Args:
        circuit: A quantum circuit.

    Returns:
        The value of the communication feature for this circuit.
    """
    num_qubits = circuit.num_qubits
    dag = qiskit.converters.circuit_to_dag(circuit)
    dag.remove_all_ops_named("barrier")

    graph = nx.Graph()
    for op in dag.two_qubit_ops():
        q1, q2 = op.qargs
        graph.add_edge(circuit.find_bit(q1).index, circuit.find_bit(q2).index)

    degree_sum = sum([graph.degree(n) for n in graph.nodes])

    return degree_sum / (num_qubits * (num_qubits - 1))


def compute_liveness_with_qiskit(circuit: qiskit.QuantumCircuit) -> float:
    """Compute the liveness of the given quantum circuit.

    Liveness feature = sum of all entries in the liveness matrix / (num_qubits * depth).

    Args:
        circuit: A quantum circuit.

    Returns:
        The value of the liveness feature for this circuit.
    """

    num_qubits = circuit.num_qubits
    dag = qiskit.converters.circuit_to_dag(circuit)
    dag.remove_all_ops_named("barrier")

    activity_matrix = np.zeros((num_qubits, dag.depth()))

    for i, layer in enumerate(dag.layers()):
        for op in layer["partition"]:
            for qubit in op:
                activity_matrix[circuit.find_bit(qubit).index, i] = 1

    return np.sum(activity_matrix) / (num_qubits * dag.depth())


def compute_parallelism_with_qiskit(circuit: qiskit.QuantumCircuit) -> float:
    """Compute the parallelism of the given quantum circuit.

    Parallelism feature = max((((# of gates / depth) - 1) /(# of qubits - 1)), 0).

    Args:
        circuit: A quantum circuit.

    Returns:
        The value of the parallelism feature for this circuit.
    """
    dag = qiskit.converters.circuit_to_dag(circuit)
    dag.remove_all_ops_named("barrier")
    if circuit.num_qubits <= 1:
        return 0
    depth = dag.depth()
    if depth == 0:
        return 0
    return max(((len(dag.gate_nodes()) / depth) - 1) / (circuit.num_qubits - 1), 0)


def compute_measurement_with_qiskit(circuit: qiskit.QuantumCircuit) -> float:
    """Compute the measurement feature of the given quantum circuit.

    Measurement feature = # of layers of mid-circuit measurement / circuit depth.

    Args:
        circuit: A quantum circuit.

    Returns:
        The value of the measurement feature for this circuit.
    """
    circuit.remove_final_measurements()
    dag = qiskit.converters.circuit_to_dag(circuit)
    dag.remove_all_ops_named("barrier")

    reset_moments = 0
    gate_depth = dag.depth()

    for layer in dag.layers():
        reset_present = False
        for op in layer["graph"].op_nodes():
            if op.name == "reset":
                reset_present = True
        if reset_present:
            reset_moments += 1

    return reset_moments / gate_depth


def compute_entanglement_with_qiskit(circuit: qiskit.QuantumCircuit) -> float:
    """Compute the entanglement-ratio of the given quantum circuit.

    Entanglement-ratio = ratio between # of 2-qubit gates and total number of gates in the
    circuit.

    Args:
        circuit: A quantum circuit.

    Returns:
        The value of the entanglement feature for this circuit.
    """
    dag = qiskit.converters.circuit_to_dag(circuit)
    dag.remove_all_ops_named("barrier")

    return len(dag.two_qubit_ops()) / len(dag.gate_nodes())


def compute_depth_with_qiskit(circuit: qiskit.QuantumCircuit) -> float:
    """Compute the critical depth of the given quantum circuit.

    Critical depth = # of 2-qubit gates along the critical path / total # of 2-qubit gates.

    Args:
        circuit: A quantum circuit.

    Returns:
        The value of the depth feature for this circuit.
    """
    dag = qiskit.converters.circuit_to_dag(circuit)
    dag.remove_all_ops_named("barrier")
    longest_paths = dag.count_ops_longest_path()
    n_ed = sum(
        [
            longest_paths[name]
            for name in {op.name for op in dag.two_qubit_ops()}
            if name in longest_paths
        ]
    )
    n_e = len(dag.two_qubit_ops())

    if n_ed == 0:
        return 0

    return n_ed / n_e


def compute_all(qc: qiskit.QuantumCircuit):
    """Get the whole of supermarq

    Args:
        qc (qiskit.QuantumCircuit): quantum circquit

    Returns:
        list: list of all values
    """
    return [
        compute_communication_with_qiskit(qc),
        compute_liveness_with_qiskit(qc),
        compute_parallelism_with_qiskit(qc),
        compute_measurement_with_qiskit(qc),
        compute_entanglement_with_qiskit(qc),
        compute_depth_with_qiskit(qc)
    ]

def plot_benchmark(
    data: list[str | list[str] | list[list[float]]],
    show: bool = True,
    savefn: str | None = None,
    spoke_labels: list[str] | None = None,
    legend_loc: tuple[float, float] = (0.75, 0.85),
) -> None:
    """Create a radar plot showing the feature vectors of the given benchmarks.

    Args:
        data: Contains the title, feature data, and labels in the format:
            [[benchmark labels], [[features_1], [features_2], ...]].
        show: Display the plot using `plt.show`.
        savefn: Path to save the plot, if `None`, the plot is not saved.
        spoke_labels: Optional labels for the feature vector dimensions.
        legend_loc: Optional argument to fine tune the legend placement.
    """
    if spoke_labels is None:
        spoke_labels = ["Connectivity", "Liveness", "Parallelism", "Measurement", "Entanglement", "Depth"]

    num_spokes = len(spoke_labels)
    theta = radar_factory(num_spokes)

    _, ax = plt.subplots(dpi=150, subplot_kw=dict(projection="radar"))

    labels, case_data = data
    ax.set_rgrids([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # ax.set_title(
    #     title,
    #     weight="bold",
    #     size="medium",
    #     position=(0.5, 1.1),
    #     horizontalalignment="center",
    #     verticalalignment="center",
    # )
    for d, label in zip(case_data, labels):
        ax.plot(theta, d, label=label)
        ax.fill(theta, d, alpha=0.25)
    ax.set_varlabels(spoke_labels)

    ax.legend(loc=legend_loc, labelspacing=0.1, fontsize=11)
    plt.tight_layout()

    if savefn is not None:
        # Don't want to save figures when running tests
        plt.savefig(savefn)  # pragma: no cover

    if show:
        # Tests will hang if we show figures during tests
        plt.show()  # pragma: no cover

    plt.close()
    
def radar_factory(num_vars: int) -> np.ndarray[np.float_]:
    """Create a radar chart with `num_vars` axes.

    (https://matplotlib.org/stable/gallery/specialty_plots/radar_chart.html) This function
    creates a `RadarAxes` projection and registers it.

    Args:
        num_vars: Number of variables for radar chart.

    Returns:
        A list of evenly spaced angles.
    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarAxes(RadarAxesMeta):
        """A helper class that sets the shape of the feature plot"""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """Initializes the helper `RadarAxes` class."""
            self.frame = "circle"
            self.theta = theta
            self.num_vars = num_vars
            super().__init__(*args, **kwargs)

    register_projection(RadarAxes)
    return theta


class RadarAxesMeta(PolarAxes):
    """A helper class to generate feature-vector plots."""

    name = "radar"
    # use 1 line segment to connect specified points
    RESOLUTION = 1

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the `RadarAxesMeta` class."""
        super().__init__(*args, **kwargs)
        # rotate plot such that the first axis is at the top
        self.set_theta_zero_location("N")

    def fill(
        self, *args: Any, closed: bool = True, **kwargs: Any
    ) -> list[matplotlib.patches.Polygon]:
        """Method to override fill so that line is closed by default.

        Args:
            args: Arguments to be passed to fill.
            closed: Optional parameter to close fill or not. Defaults to True.
            kwargs: Other desired keyworded arguments to be passed to fill.

        Returns:
            A list of `matplotlib.patches.Polygon`.
        """
        return super().fill(closed=closed, *args, **kwargs)

    def plot(self, *args: Any, **kwargs: Any) -> None:
        """Overrides plot so that line is closed by default.

        Args:
            args: Desired arguments for plotting.
            kwargs: Other desired keyword arguments for plotting.
        """
        lines = super().plot(*args, **kwargs)
        for line in lines:
            self._close_line(line)

    def _close_line(self, line: matplotlib.lines.Line2D) -> None:
        """A method to close the input line.

        Args:
            line: The line to close.
        """
        x, y = line.get_data()
        # FIXME: markers at x[0], y[0] get doubled-up.
        # See issue https://github.com/Infleqtion/client-superstaq/issues/27
        if x[0] != x[-1]:
            x = np.append(x, x[0])
            y = np.append(y, y[0])
            line.set_data(x, y)

    def set_varlabels(self, labels: list[str]) -> None:
        """Sets the spoke labels at the appropriate points on the radar plot.

        Args:
            labels: The list of labels to apply.
        """
        self.set_thetagrids(np.degrees(self.theta), labels, fontsize=10)

    def _gen_axes_patch(self) -> matplotlib.patches.Circle:
        # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
        # in axes coordinates.
        return Circle((0.5, 0.5), 0.5)

    def _gen_axes_spines(self) -> matplotlib.spines.Spine:
        return super()._gen_axes_spines()
