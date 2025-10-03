### Import Packages ###
import os
import glob
import math
import argparse 
from PIL import Image
def create_image_grid(image_paths, grid_layout, output_path, bg_color='white', resize_factor=0.5):
    """
    Creates a single image by arranging multiple images in a grid, with resizing.
    """
    if not image_paths:
        print(f"  > Warning: No images found to create a grid for {output_path}.")
        return
    try:
        with Image.open(image_paths[0]) as img:
            new_width = int(img.width * resize_factor)
            new_height = int(img.height * resize_factor)
            img_size = (new_width, new_height)
    except (FileNotFoundError, IndexError):
        print(f"  > Error: Could not open the first image. Check if paths are correct.")
        return

    cols, rows = grid_layout
    grid_width = img_size[0] * cols
    grid_height = img_size[1] * rows
    grid_image = Image.new('RGB', (grid_width, grid_height), bg_color)

    for i, path in enumerate(image_paths):
        if i >= (cols * rows): break
        row = i // cols
        col = i % cols
        try:
            with Image.open(path) as img:
                img = img.resize(img_size, Image.Resampling.LANCZOS)
                x_offset = col * img_size[0]
                y_offset = row * img_size[1]
                grid_image.paste(img, (x_offset, y_offset))
        except FileNotFoundError:
            print(f"  > Warning: Could not find image {path}. Skipping.")

    grid_image.save(output_path)
    print(f"  > Successfully created image grid at: {output_path}")
    
def compile_all_plots_in_grid(image_glob_pattern, grid_layout, output_prefix, output_dir):
    """
    Finds all images matching a pattern and arranges them into one or more grids.
    """
    all_image_paths = sorted(glob.glob(image_glob_pattern))
    if not all_image_paths:
        print(f"No images found for pattern: {image_glob_pattern}")
        return

    print(f"\nProcessing {len(all_image_paths)} images for '{output_prefix}'...")

    cols, rows = grid_layout
    plots_per_grid = cols * rows
    num_grids = math.ceil(len(all_image_paths) / plots_per_grid)

    for i in range(num_grids):
        start_index = i * plots_per_grid
        end_index = start_index + plots_per_grid
        image_slice = all_image_paths[start_index:end_index]
        output_path = os.path.join(output_dir, f"{output_prefix}_{i+1}.png")
        create_image_grid(
            image_paths=image_slice,
            grid_layout=grid_layout,
            output_path=output_path
        )

### Main Execution ###
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile plot images into a grid.")
    parser.add_argument("--eval_type", type=str, required=True, help="Evaluation type (e.g., 'test_set', 'full_pool').")
    parser.add_argument("--metric", type=str, required=True, help="Metric name (e.g., 'RMSE', 'MAE').")
    parser.add_argument("--plot_type", type=str, required=True, help="Plot type (e.g., 'trace', 'trace_relative_iGS').")
    parser.add_argument("--columns", type=int, required=True, help="Number of columns in the grid.")
    parser.add_argument("--rows", type=int, required=True, help="Number of rows in the grid.")
    args = parser.parse_args()

    ## Define Paths ##
    try:
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
    except NameError:
        PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))

    IMAGE_DIR = os.path.join(PROJECT_ROOT, 'Results', 'images')    
    SPECIFIC_COMPILED_DIR = os.path.join(IMAGE_DIR, 'compiled', args.eval_type, args.plot_type)
    os.makedirs(SPECIFIC_COMPILED_DIR, exist_ok=True)
    grid_layout = (args.columns, args.rows)
    output_prefix = args.metric
    image_glob_pattern = os.path.join(IMAGE_DIR, args.eval_type, args.metric, args.plot_type, 'trace', '*_TracePlot.png')

    ## Call ##
    compile_all_plots_in_grid(
        image_glob_pattern=image_glob_pattern,
        grid_layout=grid_layout,
        output_prefix=output_prefix,
        output_dir=SPECIFIC_COMPILED_DIR 
    )