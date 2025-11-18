import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import imageio.v3 as iio
from tqdm import tqdm
import tarfile
import shutil
import subprocess
import imageio_ffmpeg
import sys

def create_visualization(dgp_name, selector, seed, video_length, output_dir, cleanup_frames, num_iterations_total):
    """
    Generates a video and frame archive visualizing the active learning
    selection process for a given DGP, selector, and seed.
    """

    print(f"--- Starting visualization for: {dgp_name}, {selector}, Seed: {seed} ---")

    ## 1. Setup Paths and Parameters ##
    try:
        video_length_sec = float(video_length)
        if video_length_sec <= 0:
            raise ValueError
    except ValueError:
        print(f"Warning: Invalid video length '{video_length}'. Defaulting to 30 seconds.")
        video_length_sec = 30.0

    ## Define Project Root Dynamically ##
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root_local = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
    except NameError:
        project_root_local = os.getcwd()
    print(f"Project Root detected (within function) as: {project_root_local}")

    # Define all required file paths relative to project root
    base_results_path = os.path.join(project_root_local, "Results", "simulation_results", "aggregated", dgp_name)
    data_path = os.path.join(project_root_local, "Data", "processed", f"{dgp_name}.pkl")

    # Corrected path for InitialIndices.csv
    initial_path = os.path.join(base_results_path, 'InitialIndices.csv')
    selection_path = os.path.join(base_results_path, "selection_history", f"{selector}_SelectionHistory.csv")
    weight_path = os.path.join(base_results_path, "weight_history", f"{selector}_WeightHistory.csv")

    # Define output directories relative to specified *absolute* output_dir passed in
    safe_selector_name = selector.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').replace('=', '').replace(':', '')
    base_output_dir = os.path.join(output_dir, dgp_name, safe_selector_name, f"seed_{seed}") 

    # Create separate folders for SVG and PNG frames
    svg_frame_dir = os.path.join(base_output_dir, "frames_svg")
    png_frame_dir = os.path.join(base_output_dir, "frames_png_for_video")

    video_output_path = os.path.join(base_output_dir, f"{safe_selector_name}_seed_{seed}.mp4")
    tar_output_path = os.path.join(base_output_dir, f"{safe_selector_name}_seed_{seed}_svg_frames.tar.gz")

    # Create directories if they don't exist
    os.makedirs(svg_frame_dir, exist_ok=True)
    os.makedirs(png_frame_dir, exist_ok=True)

    ## 2. Load Data ##
    print("Loading data files...")
    try:
        df_data = pd.read_pickle(data_path)
        df_initial = pd.read_csv(initial_path)
        df_selection = pd.read_csv(selection_path, index_col='Iteration')
        df_weight = None
        if os.path.exists(weight_path):
            df_weight = pd.read_csv(weight_path, index_col='Iteration')
        else:
            print(f"Note: Weight history file not found at {weight_path}. Weights will be 'N/A'.")

    except FileNotFoundError as e:
        print(f"Error: Could not find data file. {e}")
        print(f"Missing file: {e.filename}")
        return
    except Exception as e:
        print(f"Error loading data files: {e}")
        return

    ## 3. Extract Data for the Specific Seed ##
    seed_str = f"Sim_{seed}"

    if seed_str not in df_initial.columns:
        print(f"Error: Column '{seed_str}' not found in {initial_path}")
        return
    if seed_str not in df_selection.columns:
        print(f"Error: Column '{seed_str}' not found in {selection_path}")
        return
    # Check weight DataFrame only if it was loaded successfully
    if df_weight is not None and seed_str not in df_weight.columns:
        print(f"Warning: Column '{seed_str}' not found in {weight_path}. Weights will be NaN.")

    ## Extract initial indices with cleaning ##
    try:
        initial_indices = (
            df_initial[seed_str]
            .dropna()
            .astype(str)      
            .str.strip('[]')  
            .astype(float)    
            .astype(int)     
            .tolist()
        )
    except Exception as e:
        print(f"Error cleaning initial indices for seed {seed}: {e}")
        return

    ## Extract selection indices with cleaning ##
    try:
        selection_indices = (
            df_selection[seed_str]
            .dropna()
            .astype(str)     
            .str.strip('[]')  
            .astype(float)    
            .astype(int)      
            .tolist()
        )
    except Exception as e:
        print(f"Error cleaning selection indices for seed {seed}: {e}")
        return


    ## Extract weights ##
    weights = []
    if df_weight is not None and seed_str in df_weight.columns:
        cleaned_weights = df_weight[seed_str].replace(r'^\s*$', np.nan, regex=True)
        weights = pd.to_numeric(cleaned_weights.astype(str).str.strip('[]'), errors='coerce').fillna(np.nan).tolist()
    else:
        weights = [np.nan] * len(selection_indices) 

    ## Determine frame count ##
    num_frames = num_iterations_total
    print(f"Targeting {num_frames} frames based on input.")
    if len(selection_indices) > num_frames:
        selection_indices = selection_indices[:num_frames]
        print(f"Warning: Selection history ({len(selection_indices)}) truncated to {num_frames} iterations.")
    elif len(selection_indices) < num_frames:
        print(f"Warning: Selection history ({len(selection_indices)}) is shorter than target iterations ({num_frames}). Video will be shorter.")
        num_frames = len(selection_indices) 

    if len(weights) > num_frames:
        weights = weights[:num_frames]
    elif len(weights) < num_frames:
        weights.extend([np.nan] * (num_frames - len(weights))) # Pad if needed

    if num_frames == 0:
        print("Error: No selection data to plot after adjustments.")
        return

    ## Calculate FPS based on FINAL frame count ##
    fps = num_frames / video_length_sec 
    if fps <= 0:
        print(f"Warning: Invalid fps calculation ({num_frames} / {video_length_sec}). Defaulting to 10 FPS.")
        fps = 10.0

    print(f"Found data for {num_frames} frames. Creating video with {fps:.2f} FPS.")

    ## 4. Prepare for Plotting ##
    if 'X1' not in df_data.columns:
        print(f"Error: 'X1' column not found in {data_path}. Cannot plot.")
        return
    if 'Y' not in df_data.columns:
         print(f"Error: 'Y' column not found in {data_path}. Cannot plot.")
         return

    x1_all = df_data['X1']
    y_all = df_data['Y']

    x_min, x_max = x1_all.min(), x1_all.max()
    y_min, y_max = y_all.min(), y_all.max()
    x_pad = (x_max - x_min) * 0.05
    y_pad = (y_max - y_min) * 0.05
    plot_xlim = (x_min - x_pad, x_max + x_pad)
    plot_ylim = (y_min - y_pad, y_max + y_pad)

    all_indices_set = set(df_data.index)
    initial_set = set(initial_indices)

    is_continuous_weight = "SAC" in selector

    print(f"Generating {num_frames} frames...")
    svg_files = []
    png_files = []

    ## 5. Generate Each Frame ##

    for i in tqdm(range(num_frames), desc="Generating Frames"):

        current_selection = selection_indices[i]
        current_weight = weights[i]

        valid_initial_indices = [idx for idx in initial_indices if idx in df_data.index]
        valid_selection_so_far = [idx for idx in selection_indices[:i] if idx in df_data.index]

        labeled_so_far_set = set(valid_initial_indices).union(set(valid_selection_so_far))
        if current_selection not in df_data.index:
             print(f"Warning: Skipping frame {i}, invalid index {current_selection}")
             continue

        unlabeled_remaining_set = all_indices_set - labeled_so_far_set - {current_selection}
        valid_unlabeled_remaining = [idx for idx in unlabeled_remaining_set if idx in df_data.index]

        ## Create the plot for this frame ##
        fig, ax = plt.subplots(figsize=(14, 10))

        ax.set_xlim(plot_xlim)
        ax.set_ylim(plot_ylim)

        # 1. Plot Unlabeled observations (gray) #
        if valid_unlabeled_remaining:
             ax.scatter(x1_all.loc[valid_unlabeled_remaining], y_all.loc[valid_unlabeled_remaining],
                        color='gray', alpha=0.25, label='Unlabeled', s=25)

        # 2. Plot Previously Labeled observations (red) #
        valid_labeled_so_far = list(labeled_so_far_set)
        if valid_labeled_so_far:
             ax.scatter(x1_all.loc[valid_labeled_so_far], y_all.loc[valid_labeled_so_far],
                        color='red', alpha=0.6, label='Previously Labeled', s=25)

        # 3. Plot the Newly Selected observation #
        ax.scatter(x1_all.loc[current_selection], y_all.loc[current_selection],
                   color='blue', s=105, label='Newly Selected', zorder=5)
        ax.scatter(x1_all.loc[current_selection], y_all.loc[current_selection],
                   facecolors='none', edgecolors='blue', s=210, linewidth=2.0, zorder=5)

        ## Add Titles and Labels ##
        ax.set_title(f"Selector: {selector}\nSeed: {seed} - Iteration: {i}/{num_frames-1}", fontsize=16) 
        ax.set_xlabel("X1", fontsize=12)
        ax.set_ylabel("Y", fontsize=12)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)

        ## Add DGP-specific annotations ##
        if dgp_name == "dgp_two_regime":
            for xpos in [0.5, 0.8, 0.9]:
                ax.axvline(xpos, linestyle="--", linewidth=1.5, color='dimgray', alpha=0.7)
            ax.text(0.18, 1.6, "Exploration-heavy", fontsize=10, alpha=0.8)
            ax.text(0.63, -1.6, "Investigation-heavy", fontsize=10, alpha=0.8)
            ax.text(0.805, 1.8, "High-noise trap", fontsize=9, alpha=0.8)
        elif dgp_name == "dgp_three_regime":
            for xpos in [0.4, 0.7, 0.6, 0.65]: # Added 0.65
                ax.axvline(xpos, linestyle="--", linewidth=1.5, color='dimgray', alpha=0.7)
            ax.text(0.12, 2.2, "Exploration", fontsize=10, alpha=0.8)
            ax.text(0.47, -2.4, "Investigation", fontsize=10, alpha=0.8)
            ax.text(0.75, 2.2, "Exploration", fontsize=10, alpha=0.8)
            ax.text(0.602, 2.6, "High-noise trap", fontsize=9, alpha=0.8)

        ## Add Weight Information ##
        if pd.notna(current_weight):
            weight_text = f"Weight ($w_x$): {current_weight:.4f}"
        else:
            weight_text = f"Weight ($w_x$): N/A"

        ax.text(0.98, 0.98, weight_text,
                transform=ax.transAxes,
                ha='right', va='top', fontsize=12,
                bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))

        ## Add Continuous Weight Progress Bar (if applicable) ##
        if is_continuous_weight and pd.notna(current_weight):
            # Ensure weight is within [0, 1] for bar display
            bar_weight = np.clip(current_weight, 0, 1)
            bar_ax = fig.add_axes([0.15, 0.02, 0.7, 0.03])
            bar_ax.barh([0], [1.0], height=1, color='lightgray', alpha=0.7)
            bar_ax.barh([0], [bar_weight], height=1, color='green', alpha=0.8)
            bar_ax.set_xlim(0, 1)
            bar_ax.set_yticks([])
            bar_ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            bar_ax.set_xticklabels(['0.0 (GSy)', '0.25', '0.5', '0.75', '1.0 (GSx)'])
            bar_ax.set_title("Continuous Weight ($w_x$)", fontsize=10)

        # Adjust layout slightly tighter if the bar is present
        tight_layout_rect = [0, 0.05, 1, 0.95] if (is_continuous_weight and pd.notna(current_weight)) else [0, 0, 1, 0.95]
        try:
            fig.tight_layout(rect=tight_layout_rect)
        except ValueError:
             try:
                  fig.tight_layout()
             except Exception as layout_e:
                  print(f"Warning: tight_layout failed on frame {i}: {layout_e}")


        ## Save the Frames ##
        frame_basename = f"{safe_selector_name}_seed_{seed}_frame_{i:04d}"
        svg_path = os.path.join(svg_frame_dir, f"{frame_basename}.svg")
        png_path = os.path.join(png_frame_dir, f"{frame_basename}.png")

        try:
            fig.savefig(svg_path, format='svg', bbox_inches='tight') # Added bbox_inches
            fig.savefig(png_path, dpi=100, bbox_inches='tight') # Added bbox_inches

            svg_files.append(svg_path)
            png_files.append(png_path)
        except Exception as e:
            print(f"Warning: Failed to save frame {i}. Error: {e}")

        plt.close(fig)

    ### 6. Compile Video (from PNGs) ###
    if not png_files:
        print("No PNG frames were generated. Skipping video.")
    else:
        print(f"\nCompiling video... saving to {video_output_path}")

        try:
            # Get the path to the ffmpeg executable
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

            # Define the input file pattern
            frame_pattern_base = f"{safe_selector_name}_seed_{seed}_frame_"
            input_pattern = os.path.join(png_frame_dir, f"{frame_pattern_base}%04d.png")

            # Build the full command for subprocess
            command = [
                ffmpeg_exe,
                '-r', str(fps),          
                '-i', input_pattern,     
                '-c:v', 'libx264',      
                '-pix_fmt', 'yuv420p',   
                '-y',                   
                video_output_path
            ]

            # Run the command
            result = subprocess.run(command, check=False, capture_output=True, text=True) # Changed check to False

            # Check result manually
            if result.returncode != 0:
                print(f"\n!!!!!!!! ERROR: ffmpeg execution failed! !!!!!!!!!")
                print("--- ffmpeg command: ---")
                print(" ".join(command)) # Print the command that failed
                print("--- ffmpeg error output: ---")
                print(result.stderr)
                print("------------------------------")
            else:
                print(f"Video saved successfully to {video_output_path}")

        except Exception as e:
            print(f"\n!!!!!!!! ERROR: Video compilation failed! !!!!!!!!!")
            print(f"Error was: {e}")

    ### 7. Archive Frames to .tar.gz (from SVGs) ###
    if not svg_files:
         print("\nNo SVG frames generated. Skipping archive.")
    else:
        print(f"\nArchiving SVG frames... saving to {tar_output_path}")
        try:
            with tarfile.open(tar_output_path, "w:gz") as tarf:
                for frame_path in tqdm(svg_files, desc="Archiving Frames (tar.gz)"):
                    tarf.add(frame_path, arcname=os.path.basename(frame_path))

            print(f"Frames archived successfully to {tar_output_path}")
        except Exception as e:
            print(f"Error: Failed to create .tar.gz file. {e}")

    ### 8. Optional Cleanup (Deletes both folders) ###
    if cleanup_frames:
        print(f"\nCleaning up frame directories...")
        try:
            if os.path.isdir(svg_frame_dir):
                shutil.rmtree(svg_frame_dir)
            if os.path.isdir(png_frame_dir):
                shutil.rmtree(png_frame_dir)
            print("Cleanup complete.")
        except Exception as e:
            print(f"Error: Failed to delete frame directories. {e}")

    print(f"--- Finished: {dgp_name}, {selector}, Seed: {seed} ---")


def main():
    ### 9. Argument Parsing ###
    parser = argparse.ArgumentParser(description="Generate Active Learning selection visualizations.")

    parser.add_argument('--dgp_name', type=str, required=True,
                        help='Name of the data generating process (e.g., "dgp_three_regime")')
    parser.add_argument('--selector', type=str, required=True,
                        help='Name of the selector method (e.g., "WiGS (SAC)")')
    parser.add_argument('--seed', type=int, required=True,
                        help='Simulation seed to visualize (e.g., 0)')
    parser.add_argument('--video_length', type=float, required=True,
                        help='Desired length of the output video in seconds (e.g., 30)')
    parser.add_argument('--output_dir', type=str, default="Results/visualizations",
                        help='Base directory relative to project root to save the outputs')
    parser.add_argument('--num_iterations', type=int, required=True,
                        help='Total number of iterations (frames) to generate.')
    parser.add_argument('--cleanup_frames', action='store_true',
                        help='If set, delete individual frame files after archiving/video creation.')

    args = parser.parse_args()

    ### Define Project Root Dynamically ###
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
    except NameError:
        project_root = os.getcwd()

    ### Make output_dir absolute ###
    if not os.path.isabs(args.output_dir):
         absolute_output_dir = os.path.join(project_root, args.output_dir)
    else:
         absolute_output_dir = args.output_dir

    create_visualization(args.dgp_name, args.selector, args.seed,
                         args.video_length, absolute_output_dir, args.cleanup_frames,
                         args.num_iterations)
if __name__ == "__main__":
    main()