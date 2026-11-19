"""
This script reads teh CSV generated in BN_memory_robustness.py and 
produces the following figures:

1) Stationary memory vs connection structure.
2) Average robustness (1-D_JS, averaged over perturbing each of the three
   nodes) vs connection structure.
3) Time-resolved synergy vs robustness:
       (S(t), 1-D_JS(t)) for every topology and every t.
4) Time-resolved memory vs robustness:
       (M(t), 1-D_JS(t)) for every topology and every t.
"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


# =============================================================================
# SETTINGS
# =============================================================================


csv_files_path = Path(r"C:\Users\layal\Downloads\UvA\Project 1\Formal_code_shared_on_github\20261116-165846")

prob_xlsx_path = csv_files_path / "global_state_probability_evolution.xlsx"

OUTPUT_DIR = csv_files_path / "visualizations"

SEED = 42

# Structural memory plots
JITTER_STD = 0.055
SCATTER_SIZE = 12
SCATTER_ALPHA = 0.50
VIOLIN_WIDTH = 0.72
VIOLIN_ALPHA = 0.14

# Time-resolved trajectory plots
TRAJECTORY_ALPHA = 0.08
TRAJECTORY_LINEWIDTH = 0.7
TIME_POINT_SIZE = 4

# Scatter colors for "number of nodes involved"
scatter_cmap = plt.colormaps["plasma"]

NODE_COUNT_COLORS = {
    1: scatter_cmap(0.15),
    2: scatter_cmap(0.50),
    3: scatter_cmap(0.85),
}

# Network drawings: all nodes use one color
network_cmap = plt.colormaps["GnBu"]
NETWORK_NODE_COLOR = network_cmap(0.60)

# One distinct color per global state (S000...S111) for the
# probability-vs-time plots.
state_cmap = plt.colormaps["tab10"]
STATE_COLORS = {
    f"S{i:03b}": state_cmap(i / 8)
    for i in range(8)
}

# Memory threshold used for the highlighted region is defined
# directly inside plot_time_resolved_memory_vs_jsd().


# =============================================================================
# TIME-COLUMN HELPERS
# =============================================================================

def available_times(df, prefix):
    """
    Return sorted times for columns such as:
        memory_t0, memory_t1, ...
        synergy_t0, ...
        jsd_t0, ...
    """
    pattern = re.compile(rf"^{re.escape(prefix)}_t(\d+)$")

    times = []

    for col in df.columns:
        match = pattern.match(col)
        if match:
            times.append(int(match.group(1)))

    return sorted(times)


def common_analysis_times(df):
    """
    Return times for which memory_t, synergy_t, and jsd_t all exist.
    """
    memory_times = set(available_times(df, "memory"))
    synergy_times = set(available_times(df, "synergy"))
    jsd_times = set(available_times(df, "jsd"))

    common = sorted(
        memory_times
        & synergy_times
        & jsd_times
    )

    if not common:
        raise ValueError(
            "No common memory_t*, synergy_t*, and jsd_t* columns were found."
        )

    return common


def build_time_resolved_dataframe(df, times):
    """
    Long table with one row per:
        topology × time

    Columns:
        topology_id
        t
        memory
        synergy
        jsd
        adjacency_json
    """
    records = []

    for _, row in df.iterrows():

        topology_id = int(row["topology_id"])

        for t in times:

            records.append({
                "topology_id": topology_id,
                "t": t,
                "memory": float(row[f"memory_t{t}"]),
                "synergy": float(row[f"synergy_t{t}"]),
                "jsd": float(row[f"jsd_t{t}"]),
                "robustness": 1.0 - float(row[f"jsd_t{t}"]),
                "adjacency_json": row["adjacency_json"],
            })

    return pd.DataFrame(records)


# =============================================================================
# GRAPH HELPERS
# =============================================================================

def adjacency_to_graph(A):

    G = nx.DiGraph()
    G.add_nodes_from(range(A.shape[0]))

    for target in range(A.shape[0]):
        for source in range(A.shape[1]):

            if A[target, source] == 1:
                G.add_edge(source, target)

    return G


def count_unidirectional_edges(A):

    count = 0

    for i in range(A.shape[0]):
        for j in range(i + 1, A.shape[0]):

            if bool(A[i, j]) ^ bool(A[j, i]):
                count += 1

    return count


def is_pure_unidirectional(A):
    """
    True only for the pure unidirectional structural class used in the plot:

        - exactly ONE one-way inter-node connection,
        - NO reciprocal inter-node pair anywhere,
        - NO self-loops.
    """
    n_one_way = 0
    n_reciprocal = 0

    for i in range(A.shape[0]):
        for j in range(i + 1, A.shape[0]):
            a = bool(A[i, j])
            b = bool(A[j, i])
            if a and b:
                n_reciprocal += 1
            elif a ^ b:
                n_one_way += 1

    n_self_loops = int(np.trace(A))

    return (
        n_one_way == 1
        and n_reciprocal == 0
        and n_self_loops == 0
    )


# =============================================================================
# STRUCTURAL INSET SELECTION
# =============================================================================

def choose_inset_topology(
    df,
    category,
    memory_col,
):
    """
    Choose one actual topology from the CSV for each structural class.

    Preference:
        1) belongs to requested class
        2) minimal unrelated structure
        3) highest requested memory among ties
    """

    work = df.copy()

    work["_A"] = work[
        "adjacency_json"
    ].apply(
        lambda s:
        np.array(
            json.loads(s),
            dtype=int,
        )
    )

    work[
        "_n_unidir_edges"
    ] = work[
        "_A"
    ].apply(
        count_unidirectional_edges
    )

    if category == "self_loop":

        candidates = work[
            work["n_self_loop_nodes"] > 0
        ].copy()

        candidates["_penalty"] = (
            abs(
                candidates["n_self_loop_nodes"]
                - 1
            )
            + candidates["_n_unidir_edges"]
            + 2
            * candidates["n_reciprocal_pairs"]
        )

    elif category == "unidirectional":

        candidates = work[
            work["_A"].apply(
                is_pure_unidirectional
            )
        ].copy()

        candidates["_penalty"] = 0

    elif category == "reciprocal_1":

        candidates = work[
            work["n_reciprocal_pairs"] == 1
        ].copy()

        candidates["_penalty"] = (
            candidates["n_self_loop_nodes"]
            + candidates["_n_unidir_edges"]
        )

    elif category == "reciprocal_2":

        candidates = work[
            work["n_reciprocal_pairs"] == 2
        ].copy()

        candidates["_penalty"] = (
            candidates["n_self_loop_nodes"]
            + candidates["_n_unidir_edges"]
        )

    elif category == "reciprocal_3":

        candidates = work[
            work["n_reciprocal_pairs"] == 3
        ].copy()

        candidates["_penalty"] = (
            candidates["n_self_loop_nodes"]
            + candidates["_n_unidir_edges"]
        )

    else:
        raise ValueError(
            f"Unknown category: {category}"
        )

    if candidates.empty:
        return None

    return candidates.sort_values(
        [
            "_penalty",
            memory_col,
        ],
        ascending=[
            True,
            False,
        ],
    ).iloc[0]


# =============================================================================
# STRUCTURAL INSET DRAWING
# =============================================================================

def add_network_inset(
    fig,
    row,
    rect,
):

    if row is None:
        return

    A = np.array(
        json.loads(
            row["adjacency_json"]
        ),
        dtype=int,
    )

    G = adjacency_to_graph(A)

    ax = fig.add_axes(rect)

    pos = {
        0: (-0.8, -0.5),
        1: (0.8, -0.5),
        2: (0.0, 0.8),
    }

    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_size=380,
        node_color=[
            NETWORK_NODE_COLOR
        ] * 3,
        edgecolors="black",
        linewidths=1.0,
    )

    nx.draw_networkx_labels(
        G,
        pos,
        ax=ax,
        labels={
            0: "X1",
            1: "X2",
            2: "X3",
        },
        font_size=8,
    )

    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        arrows=True,
        arrowsize=14,
        node_size=380,
        width=1.4,
        edge_color="black",
        connectionstyle="arc3,rad=0.12",
    )

    ax.set_xlim(
        -1.25,
        1.25,
    )

    ax.set_ylim(
        -1.0,
        1.2,
    )

    ax.axis("off")


# =============================================================================
# STRUCTURAL LONG-FORM DATA
# =============================================================================

def build_feature_dataframe(
    df,
    memory_col,
):

    records = []

    for _, row in df.iterrows():

        topology_id = int(
            row["topology_id"]
        )

        memory = float(
            row[memory_col]
        )

        # ---------------------------------------------------------
        # Self-loops
        # ---------------------------------------------------------

        n_self = int(
            row["n_self_loop_nodes"]
        )

        if n_self > 0:

            records.append({
                "topology_id":
                    topology_id,

                "category_x":
                    0,

                "memory":
                    memory,

                "n_nodes":
                    n_self,
            })

        # ---------------------------------------------------------
        # Pure unidirectional connection
        # ---------------------------------------------------------

        A = np.array(
            json.loads(
                row["adjacency_json"]
            ),
            dtype=int,
        )

        if is_pure_unidirectional(A):

            records.append({
                "topology_id":
                    topology_id,

                "category_x":
                    1,

                "memory":
                    memory,

                "n_nodes":
                    2,
            })

        # ---------------------------------------------------------
        # Reciprocal feedback
        # ---------------------------------------------------------

        n_pairs = int(
            row["n_reciprocal_pairs"]
        )

        if n_pairs > 0:

            records.append({
                "topology_id":
                    topology_id,

                "category_x":
                    n_pairs + 1,

                "memory":
                    memory,

                "n_nodes":
                    int(
                        row[
                            "n_nodes_in_reciprocal_pairs"
                        ]
                    ),
            })

    return pd.DataFrame(
        records
    )


# =============================================================================
# MEMORY VS STRUCTURE
# =============================================================================

def plot_memory_vs_feedback_structure(
    df,
    memory_col,
    ylabel,
    title,
    filename,
):

    feature_df = (
        build_feature_dataframe(
            df,
            memory_col,
        )
    )

    fig, ax = plt.subplots(
        figsize=(12, 7.5)
    )

    # -------------------------------------------------------------
    # Violins
    # -------------------------------------------------------------

    violin_data = []
    violin_positions = []

    for category_x in [
        0,
        1,
        2,
        3,
        4,
    ]:

        values = feature_df.loc[
            feature_df[
                "category_x"
            ] == category_x,
            "memory",
        ].to_numpy()

        if len(values) > 1:

            violin_data.append(
                values
            )

            violin_positions.append(
                category_x
            )

    if violin_data:

        violin = ax.violinplot(
            violin_data,
            positions=violin_positions,
            widths=VIOLIN_WIDTH,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )

        for body in violin[
            "bodies"
        ]:

            body.set_facecolor(
                "lightgray"
            )

            body.set_edgecolor(
                "gray"
            )

            body.set_alpha(
                VIOLIN_ALPHA
            )

    # -------------------------------------------------------------
    # Jittered scatter
    # -------------------------------------------------------------

    local_rng = np.random.default_rng(
        SEED
    )

    for n_nodes in [
        1,
        2,
        3,
    ]:

        subset = feature_df[
            feature_df[
                "n_nodes"
            ] == n_nodes
        ]

        if subset.empty:
            continue

        jitter = local_rng.normal(
            0.0,
            JITTER_STD,
            size=len(subset),
        )

        ax.scatter(
            subset[
                "category_x"
            ].to_numpy()
            + jitter,

            subset[
                "memory"
            ],

            s=SCATTER_SIZE,
            alpha=SCATTER_ALPHA,
            color=NODE_COUNT_COLORS[
                n_nodes
            ],
            edgecolors="none",

            label=(
                f"{n_nodes} node"
                if n_nodes == 1
                else f"{n_nodes} nodes"
            ),

            zorder=3,
        )

    ax.set_xticks([
        0,
        1,
        2,
        3,
        4,
    ])

    ax.set_xticklabels([
        "Self-loop(s)",
        "Unidirectional",
        "1 reciprocal\npair",
        "2 reciprocal\npairs",
        "3 reciprocal\npairs",
    ])

    ax.set_xlabel(
        "Connection structure"
    )

    ax.set_ylabel(
        ylabel
    )

    ax.set_title(
        title
    )

    ax.grid(
        axis="y",
        alpha=0.20,
        linewidth=0.8,
    )

    ax.set_axisbelow(
        True
    )

    ax.legend(
        title="Nodes involved",
        loc="lower left",
        frameon=True,
    )

    # -------------------------------------------------------------
    # Actual topology insets
    # -------------------------------------------------------------

    inset_categories = [
        "self_loop",
        "unidirectional",
        "reciprocal_1",
        "reciprocal_2",
        "reciprocal_3",
    ]

    inset_lefts = [
        0.105,
        0.285,
        0.465,
        0.645,
        0.825,
    ]

    for category, left in zip(
        inset_categories,
        inset_lefts,
    ):

        row = choose_inset_topology(
            df,
            category,
            memory_col,
        )

        add_network_inset(
            fig,
            row,
            rect=[
                left,
                0.015,
                0.09,
                0.15,
            ],
        )

    fig.subplots_adjust(
        bottom=0.28,
        left=0.09,
        right=0.97,
        top=0.92,
    )

    fig.savefig(
        OUTPUT_DIR
        / filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


# =============================================================================
# TIME-RESOLVED SYNERGY VS JSD
# =============================================================================

def plot_time_resolved_synergy_vs_jsd(
    df_long
):
    """
    Each topology contributes a trajectory:
        (S(0),D_JS(0)) -> ... -> (S(T),D_JS(T))

    All 512 topology trajectories are shown faintly.
    """

    fig, ax = plt.subplots(
        figsize=(8.5, 7)
    )

    # One trajectory per topology
    for _, group in df_long.groupby(
        "topology_id"
    ):

        group = group.sort_values(
            "t"
        )

        # ax.plot(
        #     group["synergy"],
        #     group["jsd"],
        #     linewidth=TRAJECTORY_LINEWIDTH,
        #     alpha=TRAJECTORY_ALPHA,
        # )

        ax.scatter(
            group["synergy"],
            group["robustness"],
            s=TIME_POINT_SIZE,
            alpha=TRAJECTORY_ALPHA,
            edgecolors="none",
        )

    ax.set_xlabel(
        "Synergy"
    )

    ax.set_ylabel(
        r"Robustness $(1-D_{JS})$"
    )

    ax.set_title(
        "Robustness vs Synergy"
    )

    ax.set_xlim(
        left=-1.0
    )

    ax.grid(
        alpha=0.20
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "time_resolved_synergy_vs_jsd.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


# =============================================================================
# TIME-RESOLVED MEMORY VS JSD
# =============================================================================

def plot_time_resolved_memory_vs_jsd(
    df_long
):
    """
    Plot the complete time-resolved relationship (M(t), 1-D_JS(t)).

    The x-axis ends at 3.01.
    """

    fig, ax = plt.subplots(
        figsize=(9.5, 7.5)
    )

    ax.scatter(
        df_long["memory"],
        df_long["robustness"],
        s=8,
        alpha=0.25,
        edgecolors="none",
        color="gray",
        zorder=2,
    )

    ax.set_xlabel(
        "Memory"
    )

    ax.set_ylabel(
        r"Robustness $(1-D_{JS})$"
    )

    ax.set_title(
        "Robustness vs Memory"
    )

    current_xlim = ax.get_xlim()
    ax.set_xlim(
        current_xlim[0],
        3.01,
    )

    ax.grid(
        alpha=0.20
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "time_resolved_memory_vs_jsd.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


# =============================================================================
# WORST-CASE ROBUSTNESS VS SYNERGY / VS MEMORY
# =============================================================================
#
# One point per topology. y-axis in both figures:
#     mean_worst_case_robustness_avg3 = (1/3) sum_i min_t R_i(t),
#     i.e. each node's own worst instant, averaged over the three nodes.

WORST_CASE_YLABEL = (
    r"Worst-case robustness $\frac{1}{3}\sum_i \min_t (1-D_{JS}^{(i)}(t))$"
)


def plot_worst_case_robustness_vs_synergy(df):

    fig, ax = plt.subplots(figsize=(8.5, 7))

    ax.scatter(
        df["mean_synergy_over_time_avg3"],
        df["mean_worst_case_robustness_avg3"],
        s=14,
        alpha=0.55,
        color=scatter_cmap(0.50),
        edgecolors="none",
    )

    ax.set_xlabel("Mean synergy over time (avg. over 3 perturbations)")
    ax.set_ylabel(WORST_CASE_YLABEL)
    ax.set_title("Worst-case robustness vs synergy")
    ax.grid(alpha=0.20)

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "worst_case_robustness_vs_synergy.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


def plot_worst_case_robustness_vs_memory(df):

    fig, ax = plt.subplots(figsize=(8.5, 7))

    ax.scatter(
        df["Memory"],
        df["mean_worst_case_robustness_avg3"],
        s=14,
        alpha=0.55,
        color=scatter_cmap(0.50),
        edgecolors="none",
    )

    ax.set_xlabel("Memory")
    ax.set_ylabel(WORST_CASE_YLABEL)
    ax.set_title("Worst-case robustness vs memory")
    ax.grid(alpha=0.20)

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "worst_case_robustness_vs_memory.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


# =============================================================================
# SYSTEM-LEVEL D_JS OVER TIME FOR SELECTED TOPOLOGIES
# =============================================================================

def plot_system_jsd_for_topologies(df, selected):
    """
    Plot the system-level D_JS(t) (the jsd_t* columns) for a few topologies,
    one subplot each.

    Note: jsd_t* in the summary CSV is for the perturbation of X1 only,
    not averaged over the three nodes.

    selected: list of (label, topology_id) pairs.
    """

    times = available_times(df, "jsd")

    fig, axes = plt.subplots(
        1, len(selected),
        figsize=(5 * len(selected), 4.2),
        sharey=True,
        squeeze=False,
    )

    for ax, (label, topology_id) in zip(axes[0], selected):

        row = df[df["topology_id"] == topology_id]

        if row.empty:
            ax.set_title(f"{label} (topology {topology_id} not found)")
            continue

        row = row.iloc[0]
        jsd = [row[f"jsd_t{t}"] for t in times]

        ax.plot(
            times,
            jsd,
            color=scatter_cmap(0.50),
            linewidth=1.6,
        )

        ax.set_title(f"{label}\n(topology {topology_id})")
        ax.set_xlabel("Time, $t$")
        ax.grid(alpha=0.20)

    axes[0][0].set_ylabel(r"System-level $D_{JS}(t)$")

    fig.suptitle(r"System-level $D_{JS}$ over time (perturbation of X1)")
    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "system_jsd_selected_topologies.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


# =============================================================================
# NODE MEMORY TIME SERIES FOR SELECTED TOPOLOGIES
# =============================================================================

def plot_node_memory_timeseries(node_df, topology_ids):
    """
    Read from node_memory_timeseries.csv (topology_id, t, X1, X2, X3) and
    plot I(X_i(t); X_i(t+1)) vs t for each node, one subplot per topology.
    """

    node_colors = {
        "X1": scatter_cmap(0.15),
        "X2": scatter_cmap(0.50),
        "X3": scatter_cmap(0.85),
    }

    # Different line styles so nodes with identical curves stay visible.
    node_linestyles = {
        "X1": "-",
        "X2": "--",
        "X3": ":",
    }

    n_cols = 3
    n_rows = -(-len(topology_ids) // n_cols)  # ceiling division

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 3.8 * n_rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for i, topology_id in enumerate(topology_ids):

        ax = axes[i // n_cols][i % n_cols]

        table = node_df[node_df["topology_id"] == topology_id].sort_values("t")

        if table.empty:
            ax.set_title(f"Topology {topology_id} (not found)")
            continue

        for node, color in node_colors.items():
            ax.plot(
                table["t"],
                table[node],
                color=color,
                linestyle=node_linestyles[node],
                linewidth=1.8,
                label=node,
            )

        ax.set_title(f"Topology {topology_id}")
        ax.grid(alpha=0.20)

        if i // n_cols == n_rows - 1:
            ax.set_xlabel("Time, $t$")
        if i % n_cols == 0:
            ax.set_ylabel(r"$I(X_i(t);X_i(t+1))$")

    # Hide any unused subplot slots.
    for j in range(len(topology_ids), n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Node",
        loc="lower center",
        ncol=3,
        frameon=True,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle("Node memory over time")
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])

    fig.savefig(
        OUTPUT_DIR / "node_memory_timeseries.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


# =============================================================================
# STATE PROBABILITIES OVER TIME (BASELINE VS PERTURBED, PER TOPOLOGY)
# =============================================================================

def plot_state_probabilities_over_time(prob_xlsx_path):
    """
    Read global_state_probability_evolution.xlsx (one sheet per selected
    topology, long format: state, t, baseline_prob, perturbed_prob) and
    plot P(state) vs t for baseline (solid) and perturbed (dashed), with
    one subplot per topology.
    """

    if not prob_xlsx_path.exists():
        raise FileNotFoundError(
            f"Could not find:\n{prob_xlsx_path}\n\n"
            "Run global_state_probability_evolution.py and point "
            "prob_xlsx_path at its output, or copy the .xlsx here."
        )

    workbook = pd.ExcelFile(prob_xlsx_path)
    topology_sheets = [name for name in workbook.sheet_names if name != "legend"]

    n_topologies = len(topology_sheets)
    n_cols = 2
    n_rows = -(-n_topologies // n_cols)  # ceiling division

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(6.5 * n_cols, 4.2 * n_rows),
        squeeze=False,
    )

    for i, sheet_name in enumerate(topology_sheets):

        ax = axes[i // n_cols][i % n_cols]

        table = workbook.parse(sheet_name)

        for state in sorted(table["state"].unique()):

            state_df = table[table["state"] == state].sort_values("t")
            color = STATE_COLORS[state]

            ax.plot(
                state_df["t"],
                state_df["baseline_prob"],
                color=color,
                linestyle="-",
                linewidth=1.4,
            )

            ax.plot(
                state_df["t"],
                state_df["perturbed_prob"],
                color=color,
                linestyle="--",
                linewidth=1.4,
            )

        ax.set_title(sheet_name.replace("_", " ").title())
        ax.set_xlabel("Time, $t$")
        ax.set_ylabel("$P(\\mathrm{state})$")
        ax.grid(alpha=0.20)

    # Hide any unused subplot slots (e.g. 3 topologies in a 2x2 grid).
    for j in range(n_topologies, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")

    # One shared legend: state colors, plus what solid/dashed mean.
    state_handles = [
        plt.Line2D([0], [0], color=color, linewidth=1.4, label=state)
        for state, color in STATE_COLORS.items()
    ]
    style_handles = [
        plt.Line2D([0], [0], color="black", linestyle="-", label="Baseline"),
        plt.Line2D([0], [0], color="black", linestyle="--", label="Perturbed"),
    ]

    fig.legend(
        handles=state_handles + style_handles,
        loc="lower center",
        ncol=5,
        frameon=True,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle("Global state probability over time: baseline vs perturbed")
    fig.tight_layout(rect=[0, 0.08, 1, 0.97])

    fig.savefig(
        OUTPUT_DIR / "state_probabilities_over_time.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


# =============================================================================
# VALIDATION
# =============================================================================

def validate_columns(
    df,
):

    static_required = [
        "topology_id",
        "stationary_memory",
        "adjacency_json",
        "n_self_loop_nodes",
        "n_unidirectional_nodes",
        "n_reciprocal_pairs",
        "n_nodes_in_reciprocal_pairs",
        "mean_robustness_over_time_avg3",
        "X1_memory_auc_t0_t4",
        "X1_mean_robustness_over_time",
        "X2_memory_auc_t0_t4",
        "X2_mean_robustness_over_time",
        "X3_memory_auc_t0_t4",
        "X3_mean_robustness_over_time",
    ]

    missing = [
        column
        for column in static_required
        if column
        not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing required structural columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing
            )
        )

    times = common_analysis_times(
        df
    )

    return times


# =============================================================================
# MAIN
# =============================================================================

def main():

    if not csv_files_path.exists():

        raise FileNotFoundError(
            f"Could not find:\n"
            f"{csv_files_path}\n\n"
            "Place topology_structure_summary.csv "
            "in the same folder as this script "
            "or edit csv_files_path."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(
        csv_files_path/'BN_results_summ_rob_mem.csv'
    )

    times = validate_columns(
        df
    )

    print(
        f"Loaded {len(df)} topologies."
    )

    print(
        f"Time-resolved data available for "
        f"t={min(times)} ... {max(times)}."
    )

    # -------------------------------------------------------------
    # Long form for fully time-resolved plots
    # -------------------------------------------------------------

    df_long = (
        build_time_resolved_dataframe(
            df,
            times,
        )
    )

    # -------------------------------------------------------------
    # Figure 1:
    # Stationary memory vs structural class
    # -------------------------------------------------------------

    plot_memory_vs_feedback_structure(
        df=df,

        memory_col=
            "stationary_memory",

        ylabel=
            "Memory",

        title=
            "Memory vs connection structure",

        filename=
            "memory_vs_structure.png",
    )

    # -------------------------------------------------------------
    # Figure 3:
    # Average robustness (1-D_JS, averaged over perturbing each of the
    # three nodes) vs structural class
    # -------------------------------------------------------------

    plot_memory_vs_feedback_structure(
        df=df,

        memory_col=
            "mean_robustness_over_time_avg3",

        ylabel=
            r"Average robustness $(1-D_{JS})$",

        title=
            "Average robustness vs connection structure",

        filename=
            "avg_robustness_vs_structure.png",
    )

    # -------------------------------------------------------------
    # Figure 4:
    # FULL time-resolved synergy vs robustness
    # -------------------------------------------------------------

    plot_time_resolved_synergy_vs_jsd(
        df_long
    )

    # -------------------------------------------------------------
    # Figure 5:
    # FULL time-resolved memory vs robustness
    # -------------------------------------------------------------

    plot_time_resolved_memory_vs_jsd(
        df_long
    )

    # -------------------------------------------------------------
    # Figure 6:
    # NODE ROBUSTNESS VS NODE MEMORY AREA OVER t=0,...,4
    # -------------------------------------------------------------



    # -------------------------------------------------------------
    # Figure 8:
    # WORST-CASE ROBUSTNESS (avg. over 3 nodes) VS SYNERGY
    # -------------------------------------------------------------

    plot_worst_case_robustness_vs_synergy(
        df
    )

    # -------------------------------------------------------------
    # Figure 8b:
    # WORST-CASE ROBUSTNESS (avg. over 3 nodes) VS MEMORY
    # -------------------------------------------------------------

    plot_worst_case_robustness_vs_memory(
        df
    )

    # -------------------------------------------------------------
    # Figure 9:
    # NODE MEMORY TIME SERIES FOR SELECTED TOPOLOGIES
    # (reads node_memory_timeseries.csv from the same run)
    # -------------------------------------------------------------

    node_df = pd.read_csv(
        csv_files_path / "node_memory_timeseries.csv"
    )

    plot_node_memory_timeseries(
        node_df,
        topology_ids=[1, 100, 200, 300, 400, 500],
    )

    # -------------------------------------------------------------
    # Figure 10:
    # SYSTEM-LEVEL D_JS OVER TIME FOR SELECTED TOPOLOGIES
    #   1   = fully isolated (no edges, no self-loops)
    #   99  = directed ring X1 -> X2 -> X3 -> X1 (intermediate, 3 edges)
    #   512 = fully connected (all 9 edges incl. self-loops)
    # -------------------------------------------------------------

    plot_system_jsd_for_topologies(
        df,
        selected=[
            ("Fully isolated", 1),
            ("Intermediate: ring X1→X2→X3→X1", 99),
            ("Fully connected", 512),
        ],
    )

    # -------------------------------------------------------------
    # Figure 7:
    # STATE PROBABILITIES OVER TIME, BASELINE VS PERTURBED
    # (one subplot per topology in global_state_probability_evolution.xlsx)
    # -------------------------------------------------------------

    plot_state_probabilities_over_time(
        prob_xlsx_path
    )

    print(
        "\nSaved figures to:"
    )

    print(
        OUTPUT_DIR.resolve()
    )


if __name__ == "__main__":
    main()



