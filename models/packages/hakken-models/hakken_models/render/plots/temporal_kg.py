# mypy: ignore-errors

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import FancyArrowPatch
from torch import Tensor


def plot_temporal_kg(
    facts: Tensor | None,
    show_timestamps: bool = True,
    separate_by_time: bool = False,
    figsize: tuple = (12, 8),
    node_color: str = "lightblue",
    edge_colormap: str = "viridis",
    save_path: str | None = None,
) -> None:
    """
    Plots a temporal knowledge graph from a tensor of facts.

    Args:
        facts: Tensor of shape [n_facts, 4] where each row is
            [subject, relation, object, time]
        show_timestamps: If True, include timestamp in edge labels
        separate_by_time: If True, create separate plots for each timestamp
        figsize: Figure size tuple (width, height)
        node_color: Color for nodes
        edge_colormap: Colormap name for edges
            (e.g., 'viridis', 'plasma', 'tab10')
        save_path: Optional path to save the figure (e.g., 'plot.png')
    """
    if facts is None or len(facts) == 0:
        print("No facts to plot.")
        return

    # Convert tensor to numpy
    facts_np = facts.cpu().numpy() if facts.is_cuda else facts.numpy()
    timestamps = np.unique(facts_np[:, 3])
    cmap = plt.cm.get_cmap(edge_colormap)

    # Helper function to format label
    def format_label(rel, time, direction="forward"):
        label = f"R{rel}"
        if show_timestamps:
            label += f"(t={int(time)})"
        if direction == "forward":
            return label
        return label

    # Helper function to get label position
    def get_label_position(pos1, pos2, offset=0):
        mid_x = (pos1[0] + pos2[0]) / 2
        mid_y = (pos1[1] + pos2[1]) / 2
        # Calculate perpendicular offset
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            perp_x = -dy / length * offset
            perp_y = dx / length * offset
            mid_x += perp_x
            mid_y += perp_y
        return (mid_x, mid_y)

    # Helper function to calculate offset positions for parallel edges
    def get_offset_positions(pos1, pos2, offset_distance):
        """Calculate offset positions perpendicular to the edge direction."""
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        length = np.sqrt(dx**2 + dy**2)
        if length == 0:
            return pos1, pos2
        # Perpendicular unit vector
        perp_x = -dy / length
        perp_y = dx / length
        # Apply offset
        offset1 = (
            pos1[0] + perp_x * offset_distance,
            pos1[1] + perp_y * offset_distance,
        )
        offset2 = (
            pos2[0] + perp_x * offset_distance,
            pos2[1] + perp_y * offset_distance,
        )
        return offset1, offset2

    # Helper function to draw edge
    def draw_edge(ax, pos1, pos2, color, width=2.5, arrowsize=20, offset=0):
        pos1 = tuple(float(x) for x in pos1)
        pos2 = tuple(float(x) for x in pos2)
        if offset != 0:
            pos1, pos2 = get_offset_positions(pos1, pos2, offset)
        arrow = FancyArrowPatch(
            pos1,
            pos2,
            arrowstyle="->",
            mutation_scale=arrowsize,
            color=color,
            linewidth=width,
            zorder=2,
            shrinkA=10,
            shrinkB=10,
        )
        ax.add_patch(arrow)
    
    graph: nx.DiGraph
    if separate_by_time:
        n_times = len(timestamps)
        cols = min(3, n_times)
        rows = (n_times + cols - 1) // cols
        figsize_mult = (figsize[0], figsize[1] * rows / 2)
        _fig, axes = plt.subplots(rows, cols, figsize=figsize_mult)
        axes = np.atleast_1d(axes).flatten()
        for idx, timestamp in enumerate(timestamps):
            ax = axes[idx]
            mask = facts_np[:, 3] == timestamp
            time_facts = facts_np[mask]

            # Build graph
            graph = nx.DiGraph()
            edge_groups: dict[tuple, list[tuple]] = {}  # (subj, obj) -> list of (rel, time, direction)

            for fact in time_facts:
                subj, rel, obj, t = (
                    int(fact[0]),
                    int(fact[1]),
                    int(fact[2]),
                    float(fact[3]),
                )
                graph.add_edge(subj, obj)
                edge_key = (subj, obj)
                if edge_key not in edge_groups:
                    edge_groups[edge_key] = []
                edge_groups[edge_key].append((rel, t, "forward"))

            # Check for reverse edges
            for fact in time_facts:
                subj, rel, obj, t = (
                    int(fact[0]),
                    int(fact[1]),
                    int(fact[2]),
                    float(fact[3]),
                )
                reverse_key = (obj, subj)
                if reverse_key in edge_groups:
                    edge_groups[reverse_key].append((rel, t, "reverse"))

            pos = nx.spring_layout(graph, seed=42, k=2)

            # Draw nodes
            nx.draw_networkx_nodes(
                graph,
                pos,
                ax=ax,
                node_color=node_color,
                node_size=600,
                edgecolors="#333",
                linewidths=1.5,
            )
            nx.draw_networkx_labels(graph, pos, ax=ax, font_size=10, font_weight="bold")

            # Draw edges separately with offsets
            color = cmap(idx / max(1, n_times - 1))
            edge_offset_distance = 0.25  # Distance between parallel edges

            for (subj, obj), edge_list in edge_groups.items():
                if subj not in pos or obj not in pos:
                    continue

                pos1, pos2 = pos[subj], pos[obj]
                num_edges = len(edge_list)

                # Separate forward and reverse edges
                forward_edges = [
                    (i, edge) for i, edge in enumerate(edge_list) if edge[2] == "forward"
                ]
                reverse_edges = [
                    (i, edge) for i, edge in enumerate(edge_list) if edge[2] == "reverse"
                ]

                # Calculate offsets: forward edges on one side, reverse on the other
                if num_edges == 1:
                    offsets = [0]
                # Forward edges get positive offsets, reverse get negative
                elif forward_edges and reverse_edges:
                    # Bidirectional: separate them clearly
                    n_forward = len(forward_edges)
                    n_reverse = len(reverse_edges)
                    forward_offsets = (
                        np.linspace(
                            edge_offset_distance * 0.3,
                            edge_offset_distance * (0.3 + n_forward - 1),
                            n_forward,
                        )
                        if n_forward > 0
                        else []
                    )
                    reverse_offsets = (
                        np.linspace(
                            -edge_offset_distance * (0.3 + n_reverse - 1),
                            -edge_offset_distance * 0.3,
                            n_reverse,
                        )
                        if n_reverse > 0
                        else []
                    )
                    # Create offset mapping
                    offset_map = {}
                    for idx, (orig_idx, _) in enumerate(forward_edges):
                        offset_map[orig_idx] = forward_offsets[idx]
                    for idx, (orig_idx, _) in enumerate(reverse_edges):
                        offset_map[orig_idx] = reverse_offsets[idx]
                    offsets = [offset_map[i] for i in range(num_edges)]
                else:
                    # All same direction: center them
                    offsets = np.linspace(
                        -edge_offset_distance * (num_edges - 1) / 2,
                        edge_offset_distance * (num_edges - 1) / 2,
                        num_edges,
                    )

                # Draw each edge separately
                for i, (rel, t, direction) in enumerate(edge_list):
                    offset = offsets[i]
                    # For reverse edges, swap positions to draw in correct direction
                    if direction == "reverse":
                        draw_edge(
                            ax,
                            pos2,  # Swapped
                            pos1,  # Swapped
                            color,
                            width=2,
                            arrowsize=15,
                            offset=-offset,  # Invert offset for reverse direction
                        )
                        # Label position also needs to account for swapped positions
                        label_pos = get_label_position(pos2, pos1, -offset)
                    else:
                        draw_edge(
                            ax,
                            pos1,
                            pos2,
                            color,
                            width=2,
                            arrowsize=15,
                            offset=offset,
                        )
                        label_pos = get_label_position(pos1, pos2, offset)

                    # Add label for this edge
                    label = format_label(rel, t, direction)
                    bbox = dict(
                        boxstyle="round,pad=0.35",
                        facecolor="white",
                        alpha=0.95,
                        edgecolor="gray",
                        linewidth=0.8,
                    )
                    ax.text(
                        label_pos[0],
                        label_pos[1],
                        label,
                        fontsize=8,
                        ha="center",
                        va="center",
                        bbox=bbox,
                        zorder=100,
                    )

            ax.set_title(f"t = {int(timestamp)}", fontsize=12, fontweight="bold")
            ax.axis("off")

        for idx in range(n_times, len(axes)):
            axes[idx].axis("off")

        plt.tight_layout()
    else:
        fig, ax = plt.subplots(figsize=figsize)

        # Build graph with all timestamps
        graph = nx.DiGraph()
        # (subj, obj) -> list of (rel, time, color, direction)
        edge_groups = {}
        t_min, t_max = timestamps.min(), timestamps.max()
        t_range = t_max - t_min if t_max != t_min else 1

        for fact in facts_np:
            subj = int(fact[0])
            rel = int(fact[1])
            obj = int(fact[2])
            t = float(fact[3])
            graph.add_edge(subj, obj)
            edge_key = (subj, obj)
            if edge_key not in edge_groups:
                edge_groups[edge_key] = []
            edge_groups[edge_key].append((rel, t, cmap((t - t_min) / t_range), "forward"))

        # Check for reverse edges
        for fact in facts_np:
            subj = int(fact[0])
            rel = int(fact[1])
            obj = int(fact[2])
            t = float(fact[3])
            reverse_key = (obj, subj)
            if reverse_key in edge_groups:
                edge_groups[reverse_key].append((rel, t, cmap((t - t_min) / t_range), "reverse"))

        pos = nx.spring_layout(graph, seed=42, k=2)

        # Draw nodes
        nx.draw_networkx_nodes(
            graph,
            pos,
            ax=ax,
            node_color=node_color,
            node_size=800,
            edgecolors="#333",
            linewidths=2,
        )
        nx.draw_networkx_labels(graph, pos, ax=ax, font_size=11, font_weight="bold")

        # Draw edges separately with offsets
        # Distance between parallel edges
        edge_offset_distance = 0.25

        for (subj, obj), edge_list in edge_groups.items():
            if subj not in pos or obj not in pos:
                continue

            pos1, pos2 = pos[subj], pos[obj]
            num_edges = len(edge_list)

            # Separate forward and reverse edges
            forward_edges = [(i, edge) for i, edge in enumerate(edge_list) if edge[3] == "forward"]
            reverse_edges = [(i, edge) for i, edge in enumerate(edge_list) if edge[3] == "reverse"]

            # Calculate offsets: forward edges on one side, reverse on the other
            if num_edges == 1:
                offsets = [0]
            # Forward edges get positive offsets, reverse get negative
            elif forward_edges and reverse_edges:
                # Bidirectional: separate them clearly
                n_forward = len(forward_edges)
                n_reverse = len(reverse_edges)
                forward_offsets = (
                    np.linspace(
                        edge_offset_distance * 0.3,
                        edge_offset_distance * (0.3 + n_forward - 1),
                        n_forward,
                    )
                    if n_forward > 0
                    else []
                )
                reverse_offsets = (
                    np.linspace(
                        -edge_offset_distance * (0.3 + n_reverse - 1),
                        -edge_offset_distance * 0.3,
                        n_reverse,
                    )
                    if n_reverse > 0
                    else []
                )
                # Create offset mapping
                offset_map = {}
                for idx, (orig_idx, _) in enumerate(forward_edges):
                    offset_map[orig_idx] = forward_offsets[idx]
                for idx, (orig_idx, _) in enumerate(reverse_edges):
                    offset_map[orig_idx] = reverse_offsets[idx]
                offsets = [offset_map[i] for i in range(num_edges)]
            else:
                # All same direction: center them
                offsets = np.linspace(
                    -edge_offset_distance * (num_edges - 1) / 2,
                    edge_offset_distance * (num_edges - 1) / 2,
                    num_edges,
                )

            # Draw each edge separately with its own color and label
            for i, (rel, t, color, direction) in enumerate(edge_list):
                offset = offsets[i]
                # Use the color from the edge data
                edge_color = color[:3] if len(color) > 3 else color
                # For reverse edges, swap positions to draw in correct direction
                if direction == "reverse":
                    draw_edge(
                        ax,
                        pos2,  # Swapped
                        pos1,  # Swapped
                        edge_color,
                        width=2.5,
                        arrowsize=20,
                        offset=-offset,  # Invert offset for reverse direction
                    )
                    # Label position also needs to account for swapped positions
                    label_pos = get_label_position(pos2, pos1, -offset)
                else:
                    draw_edge(
                        ax,
                        pos1,
                        pos2,
                        edge_color,
                        width=2.5,
                        arrowsize=20,
                        offset=offset,
                    )
                    label_pos = get_label_position(pos1, pos2, offset)

                # Add label for this edge
                label = format_label(rel, t, direction)
                bbox = dict(
                    boxstyle="round,pad=0.35",
                    facecolor="white",
                    alpha=0.95,
                    edgecolor="gray",
                    linewidth=0.8,
                )
                ax.text(
                    label_pos[0],
                    label_pos[1],
                    label,
                    fontsize=9,
                    ha="center",
                    va="center",
                    bbox=bbox,
                    zorder=100,
                )

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(t_min, t_max))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label("Timestamp", fontsize=11)

        ax.set_title(
            "Temporal Knowledge Graph",
            fontsize=14,
            fontweight="bold",
            pad=15,
        )
        ax.axis("off")
        plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Figure saved to {save_path}")

    plt.show()
