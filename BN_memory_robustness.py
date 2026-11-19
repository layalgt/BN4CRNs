"""
(1) This code simulates a small boolean network of N = 3 nodes. 
(2) The states are updated synchronously and deterministically based on the topology of the network and the boolean functions assigned to each node.
(3) The probabilities of the global states are computed exactly using a master equation update.
(4) Memory is defined as a system-level property and is quantified as I(X_t; X_{t+lag}) where X_t is the global state of the system at time t and X_{t+lag} is the global state of the system at time t+lag.
(5) Optimization is carried out for memory using the DE algorithm from the nevergrad package. The optimization is performed for a fixed topology, and the best boolean functions are selected based on the memory score. 
(6) Robustness is computed as (1 - D_JS), where D_JS is the Jensen-Shannon divergence between the baseline and perturbed distributions of the global states. 
    Each node is perturbed one at a time, individual robustness scores are computed by measuring the difference in their corresponding probability distributions.
    The system-level robustness is computed as the average of the individual robustness scores.
(7) The results are saved in two CSV files inside a time-stamped folder.
    BN_results_summ_rob_mem.csv, the main columns are:
        - topology_id                       : index of the topology
        - stationary_memory                 : memory of the optimized rules (the quantity that is maximized)
        - mean_robustness_over_time_avg3    : system-level robustness, 
        - mean_worst_case_robustness_avg3   : min over time of R for each perturbed node, then averaged over the three nodes
        - mean_synergy_over_time_avg3       : whole-minus-sum synergy of the perturbed runs, averaged over the three perturbations and over t = 1,...,t
        - adjacency_bits, adjacency_json    : the topology
        - lookup_tables_json                : the optimized boolean functions
        - structural columns                : n_edges, self loops, unidirectional edges, reciprocal pairs, cycle_lengths_json
   

(*) Note 1: The code also includes functions for computing single-node memory and robustness, but these are only for curiosity purposes.
    Relevant values are scored in indivdual CSV file: node_memory_timeseries.csv
(*) Note 2: The plots are generated in a different script.
"""



# =============================================================================
# PACKAGES 
# =============================================================================
from __future__ import annotations

from ast import main
from datetime import datetime
from pathlib import Path
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import nevergrad as ng
import networkx as nx
import pandas as pd
import itertools
import json
import math
import random
import time
import os

# =============================================================================
# CREATE FOLDER TO SAVE THE CSV FILE IN 
# =============================================================================
folder_dir = Path(__file__).resolve().parent
folder_name = datetime.now().strftime("%Y%m%d-%H%M%S")
folder_path = folder_dir / folder_name


# =============================================================================
# PARAMETERS
# =============================================================================
N = 3            # number of nodes in Boolean network
SEED = 42        # random seed for reproducibility 
t = 100          # total simulation time
tau = 1          # time step size/ lag with which the information-theoretical measures are computed
budget = 200     # number of iterations for optimization
n_restarts = 5   # number of restarts for optimization
baseline_p = 0.5 # probability for node x being in state 1 before perturbation
perturb_p = 0.6  # probability for node x being in state 1 after perturbation
node_memory_auc_end_time = 4  # node-memory area is integrated over t=0,...,4


State = Tuple[int, ...]
Pair = Tuple[State, State]
np.random.seed(SEED)  # set random seed for reproducibility


# ===============================================================================================================================
# DEFINING FUNCTIONS RELATED TO THE NETWORK INITIALIZATION AND REPRESENTATION
# ===============================================================================================================================
'''
For N = 3 case, there exists a total of 2**N = 8 global states, namely:
[000], 
[001], 
[010], 
[011], 
[100], 
[101], 
[110], 
[111]
'''
# ---------------------------------------------------------------------------------------------------------------------------
# FUNCTION TO LIST ALL POSSIBLE GLOBAL STATES FOR N=3 (We need this function because we are computing probabilities exactly)
# ---------------------------------------------------------------------------------------------------------------------------
def all_possible_states(N):
    return list(itertools.product([0,1], repeat=N))  #total of 2^N states

# -------------------------------------------------------------------------------------------------------------------------------------------
# FUNCTION TO GET INPUT NODES TO EACH NODE AND ALL THEIR POSSIBLE STATES (We need this because each node updates its state based on a function that depends on that node's inputs)
# -------------------------------------------------------------------------------------------------------------------------------------------
def get_input_nodes_and_states(A, node):
    input_nodes = [i for i in range(A.shape[0]) if A[i, node] == 1]
    input_states = list(itertools.product([0,1], repeat=len(input_nodes)))  #total of 2^k states
    return input_nodes, input_states

# ---------------------------------------------------------------------------------------------------------------------------
# FUNCTION TO UPDATE THE STATE OF EACH NODE
# ---------------------------------------------------------------------------------------------------------------------------
def next_state(
    state: State,
    A: np.ndarray,
    lookup_tables: Dict[int, Dict[str, object]],
) -> State:
    updated = []

    for node in range(A.shape[0]):
        input_nodes = lookup_tables[node]["inputs"]
        table = lookup_tables[node]["table"]
        input_state = tuple(state[j] for j in input_nodes)
        updated.append(table[input_state])

    return tuple(updated)

# ===============================================================================================================================
# CALCULATING EXACT PROBABILITIES OF GLOBAL STATES
# ===============================================================================================================================
'''
For N = 3 case, the code implements a master equation update where the 
probability of each global state at time 't' is computed exactly.
Thus, for each time step, we end up with a probability vector of length 8, 
where each entry corresponds to the probability 
of being in one of the 8 global states.
'''
@dataclass(frozen=True)
class AttractorResult:
    initial_state: State
    transient: Tuple[State, ...]
    cycle: Tuple[State, ...]

    @property
    def transient_length(self) -> int:
        return len(self.transient)

    @property
    def cycle_length(self) -> int:
        return len(self.cycle)


def find_attractor(
    initial_state: State,
    A: np.ndarray,
    lookup_tables: Dict[int, Dict[str, object]],
) -> AttractorResult:
    visited_at: Dict[State, int] = {}
    path: List[State] = []
    state = initial_state

    while state not in visited_at:
        visited_at[state] = len(path)
        path.append(state)
        state = next_state(state, A, lookup_tables)

    cycle_start = visited_at[state]

    return AttractorResult(
        initial_state=initial_state,
        transient=tuple(path[:cycle_start]),
        cycle=tuple(path[cycle_start:]),
    )


'''
for every conceivable starting condition, which attractor cycle does it end up in
and how long does it take to get there. This is exhaustive rather than sampled —
since N is small (3), it's cheap to just try all starting states rather than 
randomly sampling a few.
'''
def analyze_all_initial_states(
    A: np.ndarray,
    lookup_tables: Dict[int, Dict[str, object]],
) -> List[AttractorResult]:
    return [
        find_attractor(state, A, lookup_tables)
        for state in all_possible_states(A.shape[0])
    ]

'''
each of the 2^N initial states are given equal weight, then we look at where 
each one lands on its attractor cycle to build the exact stationary
joint distribution P_π(X_t, X_{t+lag}) — which is in turn what memory uses to 
compute the mutual-information "memory"
'''
def stationary_joint_distribution(
    attractor_results: Sequence[AttractorResult],
    lag: int,
) -> Dict[Pair, float]:
    """
    Exact stationary joint distribution P_pi(X_t, X_{t+lag}),
    with each initial state receiving equal prior weight.
    """
    joint = Counter()
    n_initial_states = len(attractor_results)

    for result in attractor_results:
        cycle = result.cycle
        L = len(cycle)
        phase_weight = 1.0 / (n_initial_states * L)

        for phase, x in enumerate(cycle):
            y = cycle[(phase + lag) % L]
            joint[(x, y)] += phase_weight

    return dict(joint)

# =============================================================================
# EXACT INITIAL DISTRIBUTIONS AND PROPAGATION
# =============================================================================
def independent_initial_distribution(
    p_one_per_node: Sequence[float],
) -> Dict[State, float]:
    """
    Exact product distribution over all Boolean states.

    [0.5,0.5,0.5] -> each of 8 states has probability 1/8.
    [0.6,0.5,0.5] -> node 0 is perturbed.
    """
    if len(p_one_per_node) != N:
        raise ValueError(f"Expected {N} node probabilities.")

    distribution: Dict[State, float] = {}

    for state in all_possible_states(N):
        p = 1.0

        for bit, p_one in zip(state, p_one_per_node):
            p *= p_one if bit == 1 else (1.0 - p_one)

        distribution[state] = p

    total = sum(distribution.values())

    if not np.isclose(total, 1.0):
        raise RuntimeError(f"Initial distribution not normalized: {total}")

    return distribution


def propagate_distribution_one_step(
    distribution: Dict[State, float],
    A: np.ndarray,
    lookup_tables: Dict[int, Dict[str, object]],
) -> Dict[State, float]:
    new_distribution = Counter()

    for state, probability in distribution.items():
        if probability == 0.0:
            continue

        nxt = next_state(state, A, lookup_tables)
        new_distribution[nxt] += probability

    return {
        state: float(new_distribution.get(state, 0.0))
        for state in all_possible_states(N)
    }


def distribution_history(
    initial_distribution: Dict[State, float],
    A: np.ndarray,
    lookup_tables: Dict[int, Dict[str, object]],
    n_steps: int,
) -> List[Dict[State, float]]:
    """Return [P_0, P_1, ..., P_n_steps]."""
    history = [dict(initial_distribution)]

    for _ in range(n_steps):
        history.append(
            propagate_distribution_one_step(
                history[-1],
                A,
                lookup_tables,
            )
        )

    return history

# ============================================================================================================================================================================================================================================================================
# FUNCTION TO DEFINE MUTUAL INFORMATION USING ITS FORMAL DEFINITON  (here our variables are the global state of the system at time 't' and time 't+1'. We need this because we define memory as the mutual information of the state of the system between past and future.)
# ============================================================================================================================================================================================================================================================================
def mutual_info(joint: Dict[Tuple[object, object], float],) -> float:
    p_x = Counter()  #builds probability of variable x through counting the number of times x appears
    p_y = Counter()  #builds probability of variable y through counting the number of times y appears

    for (x,y), p_xy in joint.items():
        p_x[x] += p_xy
        p_y[y] += p_xy

    mi = 0.0

    for (x, y), p_xy in joint.items():
        if p_xy <= 0.0:
            mi += 0.0

        denom = p_x[x] * p_y[y]

        if denom > 0.0:
            mi += p_xy * math.log2(p_xy / denom)

    return float(mi)


# -------------------------------------------------------------------------------------------------------------------------------------------
# FUNCTION THAT USES THE DEF OF MUTUAL INFORMATION TO CALCULATE MEMORY FROM THE STATIONARY DISTRIBUTIONS
# -------------------------------------------------------------------------------------------------------------------------------------------
'''
computed under the stationary distribution of the dynamics 
(i.e., averaged over the attractor cycles as if the system has run forever), 
not from a single simulated trajectory
'''
def stationary_memory(
    A: np.ndarray,
    lookup_tables: Dict[int, Dict[str, object]],
    lag: int = tau,
) -> float:
    analysis = analyze_all_initial_states(A, lookup_tables)
    joint = stationary_joint_distribution(analysis, lag)
    return mutual_info(joint)

# -------------------------------------------------------------------------------------------------------------------------------------------
# EXTRA: NODE-LEVEL MEMORY
# FUNCTION THAT USES THE DEF OF MUTUAL INFORMATION TO CALCULATE MEMORY AT THE LEVEL OF A SINGLE NODE
# -------------------------------------------------------------------------------------------------------------------------------------------
def node_memory_series(
    distribution_hist: Sequence[Dict[State, float]],
    node: int,
    A: np.ndarray,
    lookup_tables: Dict[int, Dict[str, object]],
) -> np.ndarray:
    """Return M_i(0), ..., M_i(T) for one node."""
    return np.array([
        node_memory_from_distribution(
            P_t,
            node,
            A,
            lookup_tables,
        )
        for P_t in distribution_hist
    ], dtype=float)


def node_memory_from_distribution(
    P_t: Dict[State, float],
    node: int,
    A: np.ndarray,
    lookup_tables: Dict[int, Dict[str, object]],
) -> float:
    """Compute node-specific finite-time memory from a global distribution."""
    joint = Counter()

    for current_state, p_state in P_t.items():
        if p_state <= 0.0:
            continue

        future_state = next_state(current_state, A, lookup_tables)
        joint[(current_state[node], future_state[node])] += p_state

    return mutual_info(dict(joint))

# -------------------------------------------------------------------------------------------------------------------------------------------
# FUNCTION TO INTEGRATE THE NODE-MEMORY CURVE OVER THE FIRST FEW TIME STEPS (TRAPEZOIDAL RULE) TO USE IN THE ROBUSTNESS PLOTS
# -------------------------------------------------------------------------------------------------------------------------------------------
def node_memory_auc_first_steps(
    memory_curve: np.ndarray,
    end_time: int = node_memory_auc_end_time,
) -> float:
    """Area under the node-memory curve from t=0 to t=end_time (trapezoidal)."""
    if end_time >= len(memory_curve):
        raise ValueError(
            "node_memory_auc_end_time must be smaller than the "
            "number of available memory samples."
        )

    y = memory_curve[: end_time + 1]
    x = np.arange(end_time + 1, dtype=float)

    return float(np.trapezoid(y, x))

# ===============================================================================================================================
# BUILDING GENOME FOR THE OPTIMIZATION
# ===============================================================================================================================
def compute_table_offsets(A: np.ndarray) -> List[int]:
    input_counts = [len(get_input_nodes_and_states(A, node)[0]) for node in range(A.shape[0])]
    table_sizes = [2**k for k in input_counts]
    return [0] + [int(x) for x in np.cumsum(table_sizes)]

def decode_lookup_tables(
    A: np.ndarray,
    table_genome: Sequence[float],
) -> Dict[int, Dict[str, object]]:
    """Convert a real-valued genome into one Boolean truth table per node."""
    table_genome = np.asarray(table_genome)
    offsets = compute_table_offsets(A)

    lookup_tables: Dict[int, Dict[str, object]] = {}

    for node in range(A.shape[0]):
        input_nodes, input_states = get_input_nodes_and_states(A, node)
        start = offsets[node]
        end = offsets[node + 1]
        node_raw = table_genome[start:end]

        table = {
            input_state: int(raw > 0.5)
            for input_state, raw in zip(input_states, node_raw)
        }

        lookup_tables[node] = {
            "inputs": input_nodes,
            "table": table,
        }

    return lookup_tables


def next_state(
    state: State,
    A: np.ndarray,
    lookup_tables: Dict[int, Dict[str, object]],
) -> State:
    """One synchronous deterministic Boolean update."""
    updated = []

    for node in range(A.shape[0]):
        input_nodes = lookup_tables[node]["inputs"]
        table = lookup_tables[node]["table"]
        input_state = tuple(state[j] for j in input_nodes)
        updated.append(table[input_state])

    return tuple(updated)


# =============================================================================
# OPTIMIZE RULES FOR MEMORY FOR A FIXED TOPOLOGY
# =============================================================================
def inner_objective(table_genome: Sequence[float], A: np.ndarray) -> float:
    lookup_tables = decode_lookup_tables(A, table_genome)
    return -stationary_memory(
        A,
        lookup_tables,
        lag=tau,
    )


def optimize_lookup_tables(
    A: np.ndarray,
    budget: int = budget,
    n_restarts: int = n_restarts,
) -> Tuple[np.ndarray, float]:
    exact_size = int(compute_table_offsets(A)[-1])
    restart_budget = max(1, budget // n_restarts)

    best_table_genome = None
    best_mi = -np.inf

    for restart in range(n_restarts):
        parametrization = ng.p.Array(
            shape=(exact_size,),
            lower=0.0,
            upper=1.0,
        )
        parametrization.random_state.seed(SEED + restart)

        optimizer = ng.optimizers.DE(
            parametrization=parametrization,
            budget=restart_budget,
        )

        recommendation = optimizer.minimize(
            lambda genome: inner_objective(genome, A)
        )

        table_genome = np.asarray(recommendation.value)
        mi = -inner_objective(table_genome, A)

        if mi > best_mi:
            best_mi = mi
            best_table_genome = table_genome.copy()

    if best_table_genome is None:
        raise RuntimeError("Optimizer failed to produce a candidate.")

    return best_table_genome, float(best_mi)


# =============================================================================
# TOPOLOGIES + STRUCTURAL FEATURES
# =============================================================================
def generate_all_structures(
    n_nodes: int,
    allow_self_loops: bool = True,
) -> List[np.ndarray]:
    if allow_self_loops:
        return [
            np.array(bits, dtype=int).reshape(n_nodes, n_nodes)
            for bits in itertools.product([0, 1], repeat=n_nodes * n_nodes)
        ]

    off_diagonal = [
        (i, j)
        for i in range(n_nodes)
        for j in range(n_nodes)
        if i != j
    ]

    structures = []

    for bits in itertools.product([0, 1], repeat=len(off_diagonal)):
        A = np.zeros((n_nodes, n_nodes), dtype=int)

        for bit, (i, j) in zip(bits, off_diagonal):
            A[i, j] = bit

        structures.append(A)

    return structures

# -------------------------------------------------------------------------------------------------------------------------------------------
# FUNCTION TO DRAW THE NETWORKS BASED ON THEIR ADJACENCY MATRICES (FOR VISUALIZATION PURPOSES)
# -------------------------------------------------------------------------------------------------------------------------------------------
def adjacency_to_graph(A: np.ndarray) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_nodes_from(range(A.shape[0]))

    for target in range(A.shape[0]):
        for source in range(A.shape[1]):
            if A[target, source] == 1:
                G.add_edge(source, target)

    return G

# -------------------------------------------------------------------------------------------------------------------------------------------
# FUNCTION TO COUNT THE NUMBER OF SIMPLE DIRECTED CYCLES IN A GIVEN TOPOLOGY (TO USE IN SCATTER PLOTS LATER)
# -------------------------------------------------------------------------------------------------------------------------------------------
def cycle_lengths_in_topology(A: np.ndarray) -> List[int]:
    """Lengths of all simple directed cycles; self-loops have length 1."""
    G = adjacency_to_graph(A)
    return sorted(len(cycle) for cycle in nx.simple_cycles(G))

# -------------------------------------------------------------------------------------------------------------------------------------------
# FUNCTION TO COUNT RECIPROCAL PAIRS IN A GIVEN TOPOLOGY
# -------------------------------------------------------------------------------------------------------------------------------------------
def reciprocal_pair_count(A: np.ndarray) -> int:
    """
    Number of unordered pairs {i,j} with both directions present:
        i -> j AND j -> i

    For N=3, possible values are 0,1,2,3.
    Self-loops do not count here.
    """
    count = 0

    for i in range(N):
        for j in range(i + 1, N):
            if A[i, j] == 1 and A[j, i] == 1:
                count += 1

    return count


def nodes_in_reciprocal_pairs(A: np.ndarray) -> int:
    """Number of distinct nodes participating in at least one reciprocal pair."""
    involved = set()

    for i in range(N):
        for j in range(i + 1, N):
            if A[i, j] == 1 and A[j, i] == 1:
                involved.add(i)
                involved.add(j)

    return len(involved)

# -------------------------------------------------------------------------------------------------------------------------------------------
# FUNCTION TO COUNT NODES INVILVED IN ONE-WAY CONNECTIONS IN A GIVEN TOPOLOGY
# -------------------------------------------------------------------------------------------------------------------------------------------
def one_way_nodes(A: np.ndarray) -> int:
    """
    Number of distinct nodes involved in at least one one-way inter-node link.

    For an unordered pair i,j:
      exactly one of i->j or j->i must be present.
    """
    involved = set()

    for i in range(N):
        for j in range(i + 1, N):
            if bool(A[i, j]) ^ bool(A[j, i]):
                involved.add(i)
                involved.add(j)

    return len(involved)



def node_label(node: int) -> str:
    return f"X{node + 1}"


def self_loop_node_names(A: np.ndarray) -> List[str]:
    return [node_label(i) for i in range(N) if A[i, i] == 1]


def unidirectional_edge_names(A: np.ndarray) -> List[str]:
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            i_to_j = bool(A[j, i])
            j_to_i = bool(A[i, j])
            if i_to_j and not j_to_i:
                edges.append(f"{node_label(i)}->{node_label(j)}")
            elif j_to_i and not i_to_j:
                edges.append(f"{node_label(j)}->{node_label(i)}")
    return edges


def unidirectional_node_names(A: np.ndarray) -> List[str]:
    """Distinct nodes participating in at least one unidirectional connection."""
    involved = set()
    for i in range(N):
        for j in range(i + 1, N):
            if bool(A[i, j]) ^ bool(A[j, i]):
                involved.add(i)
                involved.add(j)
    return [node_label(i) for i in sorted(involved)]


def reciprocal_pair_names(A: np.ndarray) -> List[str]:
    pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            if A[i, j] == 1 and A[j, i] == 1:
                pairs.append(f"{node_label(i)}<->{node_label(j)}")
    return pairs


def reciprocal_node_names(A: np.ndarray) -> List[str]:
    involved = set()
    for i in range(N):
        for j in range(i + 1, N):
            if A[i, j] == 1 and A[j, i] == 1:
                involved.add(i)
                involved.add(j)
    return [node_label(i) for i in sorted(involved)]


def memory_class(memory: float) -> int:
    """
    Exploratory memory bins:
      MI < 1.5       -> class 1
      1.5 <= MI<2.5 -> class 2
      MI >= 2.5      -> class 3

    These are bins, not claims that every value is exactly 1, 2, or 3 bits.
    """
    if memory < 1.5:
        return 1
    elif memory < 2.5:
        return 2
    else:
        return 3


# =============================================================================
# EXACT INITIAL DISTRIBUTIONS AND PROPAGATION
# =============================================================================
def independent_initial_distribution(
    p_one_per_node: Sequence[float],
) -> Dict[State, float]:
    """
    Exact product distribution over all Boolean states.

    [0.5,0.5,0.5] -> each of 8 states has probability 1/8.
    [0.6,0.5,0.5] -> node 0 is perturbed.
    """
    if len(p_one_per_node) != N:
        raise ValueError(f"Expected {N} node probabilities.")

    distribution: Dict[State, float] = {}

    for state in all_possible_states(N):
        p = 1.0

        for bit, p_one in zip(state, p_one_per_node):
            p *= p_one if bit == 1 else (1.0 - p_one)

        distribution[state] = p

    total = sum(distribution.values())

    if not np.isclose(total, 1.0):
        raise RuntimeError(f"Initial distribution not normalized: {total}")

    return distribution


def propagate_distribution_one_step(
    distribution: Dict[State, float],
    A: np.ndarray,
    lookup_tables: Dict[int, Dict[str, object]],
) -> Dict[State, float]:
    new_distribution = Counter()

    for state, probability in distribution.items():
        if probability == 0.0:
            continue

        nxt = next_state(state, A, lookup_tables)
        new_distribution[nxt] += probability

    return {
        state: float(new_distribution.get(state, 0.0))
        for state in all_possible_states(N)
    }


def distribution_history(
    initial_distribution: Dict[State, float],
    A: np.ndarray,
    lookup_tables: Dict[int, Dict[str, object]],
    n_steps: int,
) -> List[Dict[State, float]]:
    """Return [P_0, P_1, ..., P_n_steps]."""
    history = [dict(initial_distribution)]

    for _ in range(n_steps):
        history.append(
            propagate_distribution_one_step(
                history[-1],
                A,
                lookup_tables,
            )
        )

    return history


# =============================================================================
# JENSEN-SHANNON DIVERGENCE
# =============================================================================
def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    mask = p > 0.0
    return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))


def jensen_shannon_divergence(
    P: Dict[State, float],
    Q: Dict[State, float],
) -> float:
    """
    Base-2 Jensen-Shannon divergence.
    For normalized distributions: 0 <= JSD <= 1.
    """
    state_space = all_possible_states(N)

    p = np.array([P.get(state, 0.0) for state in state_space], dtype=float)
    q = np.array([Q.get(state, 0.0) for state in state_space], dtype=float)

    p = p / p.sum()
    q = q / q.sum()

    m = 0.5 * (p + q)

    jsd = 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

    return float(np.clip(jsd, 0.0, 1.0))


# =============================================================================
# WHOLE-MINUS-SUM SYNERGY PROXY
# =============================================================================
def synergy_proxy_from_distribution(
    P_t: Dict[State, float],
    A: np.ndarray,
    lookup_tables: Dict[int, Dict[str, object]],
) -> Tuple[float, List[float]]:
    r"""
    For target node i:

        S_i(t) =
            I( X_i(t+1) ; X(t) )
            - sum_j I( X_i(t+1) ; X_j(t) )

    Network value:
        S(t) = (1/N) * sum_i S_i(t)

    This is a whole-minus-sum synergy proxy, not a full PID decomposition.
    """
    per_target = []

    for target in range(N):
        joint_whole = Counter()
        joint_parts = [Counter() for _ in range(N)]

        for current_state, p_state in P_t.items():
            if p_state <= 0.0:
                continue

            future_state = next_state(current_state, A, lookup_tables)
            future_target = future_state[target]

            joint_whole[(current_state, future_target)] += p_state

            for source in range(N):
                joint_parts[source][
                    (current_state[source], future_target)
                ] += p_state

        whole_mi = mutual_info(dict(joint_whole))

        part_sum = sum(
            mutual_info(dict(joint_part))
            for joint_part in joint_parts
        )

        per_target.append(float(whole_mi - part_sum))

    return float(np.mean(per_target)), per_target


def synergy_time_series(
    distribution_hist: Sequence[Dict[State, float]],
    A: np.ndarray,
    lookup_tables: Dict[int, Dict[str, object]],
) -> np.ndarray:
    return np.array([
        synergy_proxy_from_distribution(P_t, A, lookup_tables)[0]
        for P_t in distribution_hist
    ])


# =============================================================================
# BASELINE + PERTURBATION ANALYSIS
# =============================================================================
def finite_time_memory_from_distribution(
    P_t: Dict[State, float],
    A: np.ndarray,
    lookup_tables: Dict[int, Dict[str, object]],
) -> float:
    """Compute M(t) = I(X_t ; X_{t+1}) from the exact distribution P_t."""
    joint = Counter()

    for current_state, p_state in P_t.items():
        if p_state <= 0.0:
            continue

        future_state = next_state(current_state, A, lookup_tables)
        joint[(current_state, future_state)] += p_state

    return mutual_info(dict(joint))


def finite_time_memory_series(
    distribution_hist: Sequence[Dict[State, float]],
    A: np.ndarray,
    lookup_tables: Dict[int, Dict[str, object]],
) -> np.ndarray:
    """Return M(0), ..., M(T), with M(t)=I(X_t;X_{t+1})."""
    return np.array([
        finite_time_memory_from_distribution(P_t, A, lookup_tables)
        for P_t in distribution_hist
    ], dtype=float)


def evaluate_dynamics(
    A: np.ndarray,
    lookup_tables: Dict[int, Dict[str, object]],
    n_steps: int = t,
) -> Dict[str, object]:
    """
    Baseline: [0.5, 0.5, 0.5]
    Perturbations: [0.6,0.5,0.5], [0.5,0.6,0.5], [0.5,0.5,0.6]

    Returns finite-time baseline memory M(t), baseline synergy,
    perturbed synergy for each node, mean perturbed synergy,
    JSD for each perturbation, and mean JSD.
    """
    baseline_probs = [baseline_p] * N
    P0_baseline = independent_initial_distribution(baseline_probs)

    baseline_history = distribution_history(
        P0_baseline, A, lookup_tables, n_steps
    )

    baseline_memory = finite_time_memory_series(
        baseline_history, A, lookup_tables
    )

    # Node-specific memory curves I(X_i(t); X_i(t+1))
    baseline_node_memory = np.vstack([
        node_memory_series(
            baseline_history,
            node,
            A,
            lookup_tables,
        )
        for node in range(N)
    ])

    baseline_synergy = synergy_time_series(
        baseline_history, A, lookup_tables
    )

    jsd_by_perturbed_node = []
    synergy_by_perturbed_node = []

    for perturbed_node in range(N):
        probs = [baseline_p] * N
        probs[perturbed_node] = perturb_p

        P0_perturbed = independent_initial_distribution(probs)
        perturbed_history = distribution_history(
            P0_perturbed, A, lookup_tables, n_steps
        )

        jsd_curve = np.array([
            jensen_shannon_divergence(P_base, P_pert)
            for P_base, P_pert in zip(baseline_history, perturbed_history)
        ], dtype=float)

        perturbed_synergy_curve = synergy_time_series(
            perturbed_history, A, lookup_tables
        )

        jsd_by_perturbed_node.append(jsd_curve)
        synergy_by_perturbed_node.append(perturbed_synergy_curve)

    jsd_by_perturbed_node = np.vstack(jsd_by_perturbed_node)
    synergy_by_perturbed_node = np.vstack(synergy_by_perturbed_node)

    mean_jsd = np.mean(jsd_by_perturbed_node, axis=0)
    mean_perturbed_synergy = np.mean(synergy_by_perturbed_node, axis=0)

    # Robustness is defined so that larger = more robust.
    robustness_by_node = 1.0 - jsd_by_perturbed_node
    mean_robustness = 1.0 - mean_jsd

    return {
        "baseline_memory": baseline_memory,
        "baseline_node_memory": baseline_node_memory,
        "baseline_synergy": baseline_synergy,
        "perturbed_synergy_by_node": synergy_by_perturbed_node,
        "mean_perturbed_synergy": mean_perturbed_synergy,
        "jsd_by_node": jsd_by_perturbed_node,
        "mean_jsd": mean_jsd,
        "robustness_by_node": robustness_by_node,
        "mean_robustness": mean_robustness,
    }


def evaluate_dynamics_perturb_node0(
    A: np.ndarray,
    lookup_tables: Dict[int, Dict[str, object]],
    n_steps: int = t,
) -> Dict[str, object]:
    """
    Same as evaluate_dynamics, but only node 0 (X1) is perturbed instead of
    perturbing each of the three nodes separately and averaging the results.

    Baseline:     [baseline_p, baseline_p, baseline_p]
    Perturbation: [perturb_p,  baseline_p, baseline_p]
    """
    baseline_probs = [baseline_p] * N
    P0_baseline = independent_initial_distribution(baseline_probs)

    baseline_history = distribution_history(
        P0_baseline, A, lookup_tables, n_steps
    )

    baseline_memory = finite_time_memory_series(
        baseline_history, A, lookup_tables
    )

    baseline_synergy = synergy_time_series(
        baseline_history, A, lookup_tables
    )

    perturbed_probs = [baseline_p] * N
    perturbed_probs[0] = perturb_p  # perturb only node 0 (X1)

    P0_perturbed = independent_initial_distribution(perturbed_probs)
    perturbed_history = distribution_history(
        P0_perturbed, A, lookup_tables, n_steps
    )

    jsd_curve = np.array([
        jensen_shannon_divergence(P_base, P_pert)
        for P_base, P_pert in zip(baseline_history, perturbed_history)
    ], dtype=float)

    perturbed_synergy_curve = synergy_time_series(
        perturbed_history, A, lookup_tables
    )

    # Robustness is defined so that larger = more robust.
    robustness_curve = 1.0 - jsd_curve

    return {
        "baseline_memory": baseline_memory,
        "baseline_synergy": baseline_synergy,
        "perturbed_synergy": perturbed_synergy_curve,
        "jsd": jsd_curve,
        "robustness": robustness_curve,
    }


# =============================================================================
# SERIALIZATION HELPERS
# =============================================================================

def adjacency_bitstring(A: np.ndarray) -> str:
    return "".join(str(int(x)) for x in A.flatten())


def lookup_tables_to_json(
    lookup_tables: Dict[int, Dict[str, object]],
) -> str:
    serializable = {}

    for node, rule in lookup_tables.items():
        serializable[str(node)] = {
            "inputs": [int(x) for x in rule["inputs"]],
            "table": {
                "".join(map(str, input_state)): int(output)
                for input_state, output in rule["table"].items()
            },
        }

    return json.dumps(serializable, sort_keys=True)

# =============================================================================
# SAVING TOPOLOGIES INTO CSV
# =============================================================================

def main() -> None:
    folder_path.mkdir(parents=True, exist_ok=True)
    csv_path = folder_path / "BN_results_summ_rob_mem.csv"
    node_memory_csv_path = folder_path / "node_memory_timeseries.csv"

    all_As = generate_all_structures(N, allow_self_loops=True)

    print("============================================================")
    print("BOOLEAN NETWORK: MEMORY + SYNERGY + ROBUSTNESS")
    print("============================================================")
    print(f"N                         : {N}")
    print(f"Number of topologies      : {len(all_As)}")
    print(f"Stationary optimization lag: {tau}")
    print(f"Rule-table budget/topology: {budget}")
    print(f"Restarts/topology          : {n_restarts}")
    print(f"Finite-time horizon        : 0 ... {t}")
    print(f"Baseline P(1)              : {baseline_p:.2f}")
    print(f"Perturbed-node P(1)        : {perturb_p:.2f}")
    print()

    topology_rows = []
    node_rows = []

    for topology_index, A in enumerate(all_As):
        print(
            f"Topology {topology_index + 1:3d}/{len(all_As)}",
            end="\r",
        )

        topology_id = topology_index + 1

        table_genome, stationary_mi = optimize_lookup_tables(A)
        lookup_tables = decode_lookup_tables(A, table_genome)

        dyn = evaluate_dynamics(A, lookup_tables, n_steps=t)

        cycle_lengths = cycle_lengths_in_topology(A)

    
        row = {
            "topology_id": topology_id,

            # Stationary memory used for the optimization
            "stationary_memory": float(stationary_mi),

            # Adjacency / optimized rules
            "adjacency_bits": adjacency_bitstring(A),
            "adjacency_json": json.dumps(A.astype(int).tolist()),
            "lookup_tables_json": lookup_tables_to_json(lookup_tables),

            # Structural information
            "n_edges": int(A.sum()),
            "n_self_loop_nodes": int(np.trace(A)),
            "self_loop_nodes": ",".join(self_loop_node_names(A)),
            "n_unidirectional_nodes": int(one_way_nodes(A)),
            "unidirectional_nodes": ",".join(unidirectional_node_names(A)),
            "unidirectional_edges": "; ".join(unidirectional_edge_names(A)),
            "n_reciprocal_pairs": int(reciprocal_pair_count(A)),
            "reciprocal_pairs": "; ".join(reciprocal_pair_names(A)),
            "n_nodes_in_reciprocal_pairs": int(nodes_in_reciprocal_pairs(A)),
            "nodes_in_reciprocal_pairs": ",".join(reciprocal_node_names(A)),
            "cycle_lengths_json": json.dumps(cycle_lengths),

            # One scalar synergy and robustness value per topology, both
            # computed from perturbing ONLY node 0 (X1) - not averaged
            # across all three single-node perturbations.
            # These are time averages over t=1,...,t.
            "mean_synergy_over_time": float(
                np.mean(dyn["perturbed_synergy_by_node"][0][1:])
            ),
            "mean_robustness_over_time": float(
                np.mean(dyn["robustness_by_node"][0][1:])
            ),

            # Worst-case robustness: R_worst = 1 - max_t D_JS(t) = min_t R(t).
            # Captures the worst deviation the perturbation ever causes,
            # instead of averaging it away over the whole horizon.
            "worst_case_robustness": float(
                np.min(dyn["robustness_by_node"][0][1:])
            ),

            # Same two scalars, but averaged across all three single-node
            # perturbations (the original definition, kept alongside the
            # node-0-only one above).
            "mean_synergy_over_time_avg3": float(
                np.mean(dyn["mean_perturbed_synergy"][1:])
            ),
            "mean_robustness_over_time_avg3": float(
                np.mean(dyn["mean_robustness"][1:])
            ),

            # Worst-case robustness averaged across all three single-node
            # perturbations: take each node's own worst instant first, then
            # average across nodes (NOT the worst instant of the averaged
            # curve - that would blur nodes that peak at different times).
            "mean_worst_case_robustness_avg3": float(
                np.mean([
                    np.min(dyn["robustness_by_node"][node][1:])
                    for node in range(N)
                ])
            ),
        }

        # ---------------------------------------------------------------------
        # Node-level SCALAR metrics (not the full time series - those go
        # into node_memory_timeseries.csv).
        #
        # For node i:
        #   x = area under I(X_i(t);X_i(t+1)) from t=0 to node_memory_auc_end_time
        #   y = mean robustness when node i is the perturbed node
        #       averaged over t=1,...,t.
        # ---------------------------------------------------------------------
        for node in range(N):
            node_name = f"X{node + 1}"

            row[f"{node_name}_memory_auc_t0_t{node_memory_auc_end_time}"] = (
                node_memory_auc_first_steps(
                    dyn["baseline_node_memory"][node],
                    end_time=node_memory_auc_end_time,
                )
            )

            row[f"{node_name}_mean_robustness_over_time"] = float(
                np.mean(
                    dyn["robustness_by_node"][node, 1:]
                )
            )

            # Worst-case robustness for this node: R_worst = min_t R(t).
            row[f"{node_name}_worst_case_robustness"] = float(
                np.min(
                    dyn["robustness_by_node"][node, 1:]
                )
            )

        for step in range(t + 1):
            row[f"memory_t{step}"] = float(
                dyn["baseline_memory"][step]
            )

            row[f"synergy_t{step}"] = float(
                dyn["perturbed_synergy_by_node"][0][step]
            )

            # Keep raw JSD for reference.
            row[f"jsd_t{step}"] = float(
                dyn["jsd_by_node"][0][step]
            )

            # Robustness used in the analysis:
            # R(t) = 1 - D_JS(t), so larger = more robust.
            row[f"robustness_t{step}"] = float(
                dyn["robustness_by_node"][0][step]
            )

        topology_rows.append(row)

        # ---------------------------------------------------------------------
        # Node-only MI time series: one row per (topology, t), with one
        # column per node (X1, X2, X3) holding I(X_i(t); X_i(t+1)).
        # ---------------------------------------------------------------------
        for step in range(t + 1):
            node_rows.append({
                "topology_id": topology_id,
                "t": step,
                "X1": float(dyn["baseline_node_memory"][0][step]),
                "X2": float(dyn["baseline_node_memory"][1][step]),
                "X3": float(dyn["baseline_node_memory"][2][step]),
            })

    print()

    # Save the two CSVs: topology-level summary, and node-only MI time series.
    topology_df = pd.DataFrame(topology_rows)
    topology_df.to_csv(csv_path, index=False)

    node_df = pd.DataFrame(node_rows)
    node_df.to_csv(node_memory_csv_path, index=False)

    print("\nSaved topology summary CSV to:")
    print(csv_path.resolve())
    print("\nSaved node-level MI time series CSV to:")
    print(node_memory_csv_path.resolve())
    print(
        f"\nSaved M(t), S(t), D_JS(t), and robustness R(t)=1-D_JS(t) "
        f"for t=0,...,{t} for every topology, plus per-node M_i(t) series."
    )



if __name__ == "__main__":
    main()


