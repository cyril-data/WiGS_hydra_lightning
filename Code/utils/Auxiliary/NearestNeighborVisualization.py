### LIBRARIES ###
import matplotlib.pyplot as plt
import numpy as np
import argparse 

### PLOT FUNCTION ###
def main(output_file_path):
    """
    Main function to generate and save the plot.
    """
    
    ### 1. Define Point Coordinates with natural variation ###
    labeled = {
        'L1': (1.0, 4.5),   
        'L2': (0.4, 2.8),   
        'L3': (2.3, 0.9)    
    }
    unlabeled = {
        'U1': (4.6, 3.9),  
        'U2': (5.1, 1.5)   
    }

    # Get lists for plotting
    l_pos = list(labeled.values())
    u_pos = list(unlabeled.values())
    l_names = list(labeled.keys())
    u_names = list(unlabeled.keys())

    ### 2. Setup the Plot ###
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    # Plot labeled and unlabeled points
    ax.scatter([p[0] for p in l_pos], [p[1] for p in l_pos], 
               marker='s', s=250, c='royalblue', edgecolors='darkblue', 
               linewidth=1.5, zorder=5)
    ax.scatter([p[0] for p in u_pos], [p[1] for p in u_pos], 
               marker='o', s=250, c='red', edgecolors='darkred',
               linewidth=1.5, zorder=5)

    # Add text labels for each point
    for name, pos in labeled.items():
        ax.text(pos[0] - 0.25, pos[1], name, fontsize=13, ha='center', va='center', weight='bold')
    for name, pos in unlabeled.items():
        ax.text(pos[0] + 0.25, pos[1], name, fontsize=13, ha='center', va='center', weight='bold')

    ### 3. Calculate and Plot Distances ###
    nearest_neighbors = {} 
    nn_info = {}  

    # Find nearest neighbors
    for n, (uname, upos) in enumerate(unlabeled.items()):
        min_dist = float('inf')
        nn_pos = None
        nn_label_idx = None
        nn_label_name = None
        
        for m, (lname, lpos) in enumerate(labeled.items()):
            dist = np.linalg.norm(np.array(upos) - np.array(lpos))
            if dist < min_dist:
                min_dist = dist
                nn_pos = lpos
                nn_label_idx = m
                nn_label_name = lname
        
        nn_info[uname] = (nn_pos, nn_label_idx, nn_label_name)
        nearest_neighbors[uname] = min_dist

    # Second pass: plot all distances with proper layering
    for n, (uname, upos) in enumerate(unlabeled.items()):
        nn_pos, nn_idx, nn_name = nn_info[uname]
        
        # First draw all dashed lines
        for m, (lname, lpos) in enumerate(labeled.items()):
            if m != nn_idx:  # Skip the nearest neighbor for now
                ax.plot([lpos[0], upos[0]], [lpos[1], upos[1]], 
                       color='gray', linestyle='--', alpha=0.5, linewidth=1.2, zorder=1)
        
        # Then draw the nearest neighbor line on top
        for m, (lname, lpos) in enumerate(labeled.items()):
            if m == nn_idx:
                ax.plot([lpos[0], upos[0]], [lpos[1], upos[1]], 
                       color='gold', linewidth=3.5, zorder=2)

    # Third pass: add distance labels along the lines
    label_positions = { 'L1': [], 'L2': [], 'L3': [] }

    for n, (uname, upos) in enumerate(unlabeled.items()):
        nn_pos, nn_idx, nn_name = nn_info[uname]
        
        for m, (lname, lpos) in enumerate(labeled.items()):
            t = 0.25  
            label_x = lpos[0] + t * (upos[0] - lpos[0])
            label_y = lpos[1] + t * (upos[1] - lpos[1])
            
            is_nearest = (m == nn_idx)
            label = f'$d_{{{n+1}{m+1}}}$' 
            
            label_positions[lname].append({
                'x': label_x, 'y': label_y, 'label': label,
                'is_nearest': is_nearest, 'uname': uname,
                'angle': np.arctan2(upos[1] - lpos[1], upos[0] - lpos[0])
            })

    # Now place labels with smart positioning
    for lname, positions in label_positions.items():
        if len(positions) == 1:
            pos = positions[0]
            offset = 0.15
            offset_x = -offset * np.sin(pos['angle'])
            offset_y = offset * np.cos(pos['angle'])
            
            if pos['is_nearest']:
                ax.text(pos['x'] + offset_x, pos['y'] + offset_y, pos['label'],
                       fontsize=12, ha='center', va='center', weight='bold',
                       color='black', backgroundcolor='yellow', alpha=0.8)
            else:
                ax.text(pos['x'] + offset_x, pos['y'] + offset_y, pos['label'],
                       fontsize=11, ha='center', va='center', style='italic', color='gray')
        else:
            positions.sort(key=lambda p: p['angle'])
            for i, pos in enumerate(positions):
                offset = 0.15
                if i == 0:  
                    offset_x = -offset * np.sin(pos['angle'])
                    offset_y = offset * np.cos(pos['angle'])
                else:  
                    offset_x = offset * np.sin(pos['angle'])
                    offset_y = -offset * np.cos(pos['angle'])
                
                if pos['is_nearest']:
                    ax.text(pos['x'] + offset_x, pos['y'] + offset_y, pos['label'],
                           fontsize=12, ha='center', va='center', weight='bold',
                           color='black', backgroundcolor='yellow', alpha=0.8)
                else:
                    ax.text(pos['x'] + offset_x, pos['y'] + offset_y, pos['label'],
                           fontsize=11, ha='center', va='center', style='italic', color='gray')

    ### 4. Finalize and Save Plot ###
    # ax.set_title(r'Pairwise Distances $d_{nm}$ Between Labeled and Unlabeled Points', fontsize=16, pad=20)
    ax.set_xlabel('Dimension 1 (e.g., Feature or Predicted Output)', fontsize=14)
    ax.set_ylabel('Dimension 2 (e.g., Feature or Predicted Output)', fontsize=14)

    # Reduce padding #
    all_x = [p[0] for p in l_pos] + [p[0] for p in u_pos]
    all_y = [p[1] for p in l_pos] + [p[1] for p in u_pos]
    padding = 1.0
    ax.set_xlim(min(all_x) - padding, max(all_x) + padding)
    ax.set_ylim(min(all_y) - padding, max(all_y) + padding)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True, linestyle=':', alpha=0.3, linewidth=0.5)

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color('black')

    ### 5. Save the file ###
    plt.savefig(output_file_path, dpi=300, bbox_inches='tight')
    plt.close(fig) 
    # print(f"Nearest Neighbor plot saved to: {output_file_path}")

### MAIN ###
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the 'furthest nearest neighbor' visualization for the WiGS paper.")
    parser.add_argument('--output_file', type=str, required=True,
                        help='Full path to save the final .png image (e.g., Results/images/manuscript/NearestNeighborVisualization.png)')
    
    args = parser.parse_args()
    
    main(args.output_file)