### Packages ###
import json
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter

### Graph Tree ###
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import networkx as nx

def add_nodes_and_edges(graph, data, df_columns, parent=None, edge_label=""):
    """
    Recursively add nodes and edges to the graph from the JSON data, using column names from the DataFrame
    and replacing underscores with equals signs in the output.

    Parameters:
        graph (networkx.DiGraph): The graph to add nodes and edges to.
        data (dict): The tree JSON data.
        df_columns (list): List of column names from the DataFrame.
        parent (int): Parent node ID.
        edge_label (str): Label for the edge ("True" or "False").
    """
    if isinstance(data, dict):
        # Check if the node is a leaf
        if "prediction" in data:
            node_label = str(data["prediction"])  # Use only the prediction value
            is_leaf = True
        else:
            # Map the feature index to the column name and replace "_" with "="
            feature_index = data["feature"]
            node_label = df_columns[feature_index].replace("_", "=")  # Replace underscores with equals signs
            is_leaf = False
        
        # Add the current node to the graph
        current_node = len(graph)
        graph.add_node(current_node, label=node_label, is_leaf=is_leaf)  # Add is_leaf attribute
        
        if parent is not None:
            graph.add_edge(parent, current_node, label=edge_label)
        
        # Add child nodes recursively
        if "true" in data:
            add_nodes_and_edges(graph, data["true"], df_columns, current_node, "True")
        if "false" in data:
            add_nodes_and_edges(graph, data["false"], df_columns, current_node, "False")
    else:
        raise ValueError("Unsupported JSON structure.")
def compute_positions(graph, node, pos, x=0, y=0, layer_width=1.0, depth=1.0):
    """
    Compute positions for nodes in a tree structure.
    layer_width: Controls horizontal spacing (smaller = less spread out).
    depth: Controls vertical spacing (smaller = more compact vertically).
    """
    pos[node] = (x, y)
    children = list(graph.successors(node))
    if children:
        # Spread children evenly across the x-axis
        step = layer_width / len(children)
        next_x = x - layer_width / 2 + step / 2
        for child in children:
            compute_positions(graph, child, pos, next_x, y - depth, layer_width / len(children), depth)
            next_x += step
def draw_decision_tree(graph, groups, group_colors, tree_index, filename=None):
    """
    Draw the decision tree using networkx and matplotlib. Optionally save to a file.
    """
    pos = {}
    compute_positions(graph, 0, pos, layer_width=2.0, depth=1.5)  # Increased spacing for better layout
    labels = nx.get_node_attributes(graph, "label")
    edge_labels = nx.get_edge_attributes(graph, "label")
    
    plt.figure(figsize=(10, 12))  # Adjust figure size
    ax = plt.gca()  # Get current axes for custom drawing

    # Separate leaf and feature nodes
    leaf_nodes = [n for n, attr in graph.nodes(data=True) if attr["is_leaf"]]
    feature_nodes = [n for n, attr in graph.nodes(data=True) if not attr["is_leaf"]]

    # Draw edges
    nx.draw_networkx_edges(
        graph,
        pos,
        width=2,  # Thicker edge lines
        edge_color="gray",
    )

    # Draw edge labels
    nx.draw_networkx_edge_labels(
        graph,
        pos,
        edge_labels=edge_labels,
        font_size=15,
        font_weight="bold",
        rotate=False  # Keep labels horizontal
    )

    # Draw leaf nodes (circles with red/green)
    leaf_colors = ["red" if labels[n] == "0" else "green" for n in leaf_nodes]  # Red for 0, Green for 1
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=leaf_nodes,
        node_shape="o",  # Circle shape
        node_size=800,  # Reduced size for better fit
        node_color=leaf_colors,
        edgecolors="black",
    )
    # Add labels ("0" or "1") inside the circles
    for node in leaf_nodes:
        x, y = pos[node]
        ax.text(
            x, y, labels[node],  # Label corresponds to the prediction ("0" or "1")
            ha="center", va="center",
            fontsize=20, fontweight="bold", color="white", zorder=5  # White text for visibility
        )

    # Draw feature nodes as rectangles
    for node in feature_nodes:
        x, y = pos[node]
        label = labels[node]
        
        # Calculate rectangle dimensions based on the label length
        rect_width = 0.04 * len(label)  # Width scales with label length
        rect_height = 0.2  # Fixed height
        
        # Draw the rectangle
        rect = patches.Rectangle(
            (x - rect_width / 2, y - rect_height / 2),  # Bottom-left corner
            rect_width,
            rect_height,
            edgecolor="black",
            facecolor="lightcyan",
            zorder=3,
        )
        ax.add_patch(rect)

        # Add the label inside the rectangle
        ax.text(
            x, y, label,
            ha="center", va="center",
            fontsize=20, fontweight="bold",
            zorder=4,
        )

    # Assign group based on tree index
    group_name = ""
    for group, trees in groups.items():
        if tree_index in trees:
            group_name = group
            break

    # Get the color for the group's bbox
    group_color = group_colors.get(group_name, 'lightgray')  # Default to lightgray if not found

    # Add title in the center of the plot with a gray box outlined with black
    plt.text(
        0.5, 0.85,  # x=0.5 for centering, y=0.95 for slightly near the top
        f"Decision Tree: {tree_index} ({group_name})",
        ha='center',  # Horizontal alignment in the center
        va='center',  # Vertical alignment in the center
        transform=plt.gcf().transFigure,  # Use the figure coordinates for positioning
        fontsize=20,
        fontweight='bold',
        bbox=dict(facecolor=group_color, edgecolor='black', boxstyle='round,pad=0.5')  # Use group color for bbox
    )

    # Dynamically adjust axis limits based on node positions
    x_values = [p[0] for p in pos.values()]
    y_values = [p[1] for p in pos.values()]
    ax.set_xlim(min(x_values) - 0.15, max(x_values) + 0.15)  # Add a bit of padding, but keep it tight
    ax.set_ylim(min(y_values) - 0.15, max(y_values) + 0.5)  # Add a bit of padding, but keep it tight

    ax.axis("off")  # Turn off axis lines and ticks

    # Save or show the plot
    if filename:
        plt.savefig(filename, format="png", bbox_inches="tight")
    else:
        plt.show()
    
    plt.close()  # Close the plot to free memory and avoid overlap

### UNREAL vs. DUREAL Plot ###
def PlotTreeFarmsDecisionTreeErrorsWithGroups(AllErrors, order_errors=True, epsilon=0.01):
    """
    Plot misclassification errors with TreeIndex values and group labels.
    Uses consistent epsilon spacing for both group labels and orange ticks.

    Parameters:
        AllErrors (list): List of classification errors.
        order_errors (bool): Whether to sort errors by value (default True).
        epsilon (float): Vertical spacing for group labels and ticks.
    """
    # Create the DataFrame
    pdAllErrors = pd.DataFrame(AllErrors, columns=["ClassificationError"])
    pdAllErrors["TreeIndex"] = range(0, len(AllErrors))

    # Round errors to three digits
    pdAllErrors["RoundedError"] = pdAllErrors["ClassificationError"].round(3)

    # Sort the DataFrame if order_errors is True
    if order_errors:
        pdAllErrors = pdAllErrors.sort_values(by="ClassificationError").reset_index(drop=True)

    # Create figure with a reasonable size
    fig, ax = plt.subplots(figsize=(5, 5))

    # Calculate the y-range for better spacing
    y_min = min(pdAllErrors["ClassificationError"]) - 0.02
    y_max = max(pdAllErrors["ClassificationError"]) + 0.025  # Increased to accommodate elevated ticks
    y_range = y_max - y_min

    # Scatter plot for points with equal size
    ax.scatter(
        pdAllErrors.index,
        pdAllErrors["ClassificationError"],
        s=200,
        color="white",
        edgecolor="black",
        zorder=2,
    )

    # Add text labels for TreeIndex
    for i, row in pdAllErrors.iterrows():
        ax.text(
            i, row["ClassificationError"],
            str(int(row["TreeIndex"])),
            color="black", fontsize=10, ha="center", va="center"
        )

    # Define groups based on unique values
    unique_values = sorted(set(pdAllErrors["ClassificationError"]))
    value_to_group = {val: f"Group {i+1}" for i, val in enumerate(unique_values)}

    # Create groups based on unique values
    groups = {}
    for i, row in pdAllErrors.iterrows():
        value = row["ClassificationError"]
        group = value_to_group[value]
        if group not in groups:
            groups[group] = {"indices": [], "value": value}
        groups[group]["indices"].append(i)

    # Calculate positions
    unique_y = min(pdAllErrors["ClassificationError"]) - 0.015

    # Draw groups and their connections
    for group_name, group_data in groups.items():
        indices = group_data["indices"]
        value = group_data["value"]
        group_x = sum(indices) / len(indices)
        
        # Place group labels epsilon below their corresponding values
        ax.text(
            group_x, value - 0.003,
            group_name,
            color="black",
            fontsize=10, ha="center", va="center",
            bbox=dict(facecolor="cyan", alpha=0.5, edgecolor="black"),
            fontweight='bold'
        )

    # Unique Ensemble label
    ax.annotate(
        "Unique Ensemble (UNREAL)",
        xy=(4, unique_y), xytext=(len(AllErrors)/2, unique_y+epsilon/2),
        fontsize=10, color="black", ha="center", va="center",
        bbox=dict(facecolor="cyan", edgecolor="black"),
        fontweight='bold'
    )

    # Connect groups to Unique Ensemble
    for group_name, group_data in groups.items():
        indices = group_data["indices"]
        value = group_data["value"]
        group_x = sum(indices) / len(indices)
        ax.annotate(
            "", 
            xy=(group_x, value - 0.005), 
            xytext=(len(AllErrors)/2, unique_y + epsilon + 0.001),
            arrowprops=dict(facecolor="cyan", edgecolor="cyan", arrowstyle="<-", lw=2)
        )

    # Draw ticks above each point at their corresponding value + epsilon
    tick_height = 0.0005  # Small tick height
    for group_name, group_data in groups.items():
        indices = group_data["indices"]
        value = group_data["value"]
        tick_y = value + epsilon  # Place ticks epsilon above their corresponding values
        
        # Draw ticks for each point in the group
        for idx in indices:
            ax.plot([idx, idx], [tick_y-tick_height, tick_y+tick_height], 
                   color="orange", linewidth=2)
        
        # Connect ticks in the same group
        ax.plot([min(indices), max(indices)], [tick_y, tick_y], 
                color="orange", linewidth=2)
        
        # Add count label
        ax.text(sum(indices) / len(indices), tick_y + 0.002, 
                f"{len(indices)} trees", 
                color="orange", ha="center", va="center", 
                fontsize=10, fontweight='bold')

    # Duplicate Ensemble annotation - positioned above the highest tick
    max_tick_y = max(pdAllErrors["ClassificationError"]) + epsilon
    duplicate_y = max_tick_y + 0.008
    ax.annotate(
        "Duplicate Ensemble (DUREAL)",
        xy=(len(AllErrors)/2+1, duplicate_y), xytext=(len(AllErrors)/2, duplicate_y+epsilon/1.5),
        fontsize=10, color="black", ha="center", va="center",
        bbox=dict(facecolor="orange", edgecolor="black"),
        fontweight='bold'
    )

    # Connect tick groups to Duplicate Ensemble
    for group_name, group_data in groups.items():
        indices = group_data["indices"]
        value = group_data["value"]
        group_x = sum(indices) / len(indices)
        tick_y = value + epsilon  # Connect from the tick level
        ax.annotate(
            "", 
            xy=(group_x, tick_y+0.003), 
            xytext=(len(AllErrors)/2, duplicate_y),
            arrowprops=dict(facecolor="orange", edgecolor="orange", arrowstyle="<-", lw=2)
        )

    # Add error counts
    error_counts = Counter(pdAllErrors["RoundedError"])
    caption_header = "Error    Count"
    caption_rows = "\n".join([f"{error:.3f}    {count}" for error, count in error_counts.items()])
    caption = caption_header + "\n" + caption_rows

    ax.text(
        0.05, 0.95,
        caption,
        ha="left", va="top",
        transform=ax.transAxes,
        fontsize=10, family="monospace",
        bbox=dict(facecolor="white", alpha=0.8)
    )

    # Customize plot appearance
    ax.set_xlim(-1, len(pdAllErrors)+0)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Index of Trees in TREEFarms")
    ax.set_ylabel("Misclassification Error")
    ax.set_xticks([])

    plt.tight_layout()
    plt.show()