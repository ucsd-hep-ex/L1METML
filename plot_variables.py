import uproot
import matplotlib.pyplot as plt
import awkward as ak
import argparse
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

def find_root_files(input_path, pattern="*.root"):
    """Find ROOT files in input path with tilde expansion and glob support."""
    # Expand ~ to user home directory
    expanded_path = os.path.expanduser(input_path)
    path_obj = Path(expanded_path)

    # 1. If it's a direct file
    if path_obj.is_file():
        if path_obj.suffix == ".root":
            return [str(path_obj)]
        else:
            raise ValueError(f"Input file is not a ROOT file: {expanded_path}")

    # 2. If it's a directory
    elif path_obj.is_dir():
        root_files = list(path_obj.glob(pattern))
        if not root_files:
            raise ValueError(f"No ROOT files found in {expanded_path} with pattern {pattern}")
        return [str(f) for f in sorted(root_files)]

    # 3. If it's a glob pattern (e.g., path/to/*.root)
    else:
        import glob
        root_files = glob.glob(expanded_path)
        if not root_files:
            # Try applying the pattern if it was just a directory path that doesn't exist yet
            raise ValueError(f"Input path/pattern does not match any files: {expanded_path}")
        
        # Filter only .root files
        root_files = [f for f in root_files if f.endswith(".root")]
        if not root_files:
            raise ValueError(f"No ROOT files found matching pattern: {expanded_path}")
            
        return sorted(root_files)

def plot_distributions(input_path, output_dir, pattern="*.root", max_events=None):
    # Variable list from convertNanoToHDF5.py
    varList = [
        "nL1PuppiCands",
        "L1PuppiCands_pt",
        "L1PuppiCands_eta",
        "L1PuppiCands_phi",
        "L1PuppiCands_charge",
        "L1PuppiCands_pdgId",
        "L1PuppiCands_puppiWeight",
        "L1PuppiCands_dxyErr",
        "HGCal3DCl_firstHcal1layers",
        "HGCal3DCl_firstHcal3layers",
        "HGCal3DCl_firstHcal5layers",
        "HGCal3DCl_hoe",
        "HGCal3DCl_showerlength",
        "HGCal3DCl_coreshowerlength",
        "genMet_pt",
        "genMet_phi",
    ]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Find all matching ROOT files
    root_files = find_root_files(input_path, pattern)
    print(f"Found {len(root_files)} files. Concatenating data...")

    # Prepare file paths for uproot (appending ':Events' to each file)
    file_paths = [f"{f}:Events" for f in root_files]

    # Get available branches from the first file to avoid errors
    with uproot.open(root_files[0]) as f:
        tree = f["Events"]
        available_branches = [b for b in varList if b in tree.keys()]
    
    print(f"Reading branches: {available_branches}")
    
    # Concatenate data from all files
    # uproot.concatenate will read data from all files and return a single awkward array
    data = uproot.concatenate(file_paths, expressions=available_branches, entry_stop=max_events)
    print(f"Total events loaded: {len(data)}")

    for var in available_branches:
        print(f"Plotting {var}...")
        vals = data[var]
        
        # Handle Jagged Arrays (e.g., L1PuppiCands_pt) by flattening them
        if hasattr(vals, "layout") and (isinstance(vals.layout, ak.contents.ListOffsetArray) or 
                                      isinstance(vals.layout, ak.contents.ListArray)):
            vals_to_plot = ak.flatten(vals).to_numpy()
        elif isinstance(vals, ak.Array):
            try:
                # Attempt to flatten multi-dimensional or jagged arrays
                vals_to_plot = ak.flatten(vals, axis=None).to_numpy()
            except:
                vals_to_plot = vals.to_numpy()
        else:
            vals_to_plot = np.array(vals)

        # Remove NaNs or Inf if any
        vals_to_plot = vals_to_plot[np.isfinite(vals_to_plot)]

        plt.figure(figsize=(10, 7))
        
        # Histogram with 50 bins
        plt.hist(vals_to_plot, bins=50, histtype='step', color='blue', linewidth=1.5, density=True)
        
        plt.title(f"Combined Distribution of {var} ({len(root_files)} files)")
        plt.xlabel(var)
        plt.ylabel("Normalized Frequency")
        plt.yscale('log')
        plt.grid(True, which="both", linestyle='--', alpha=0.5)
        
        save_path = os.path.join(output_dir, f"combined_{var}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Successfully saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot combined distributions from multiple NanoAOD ROOT files")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input directory or file pattern")
    parser.add_argument("-p", "--pattern", type=str, default="*.root", help="File pattern if input is a directory")
    parser.add_argument("-o", "--output", type=str, default="combined_plots", help="Output directory for PNG files")
    parser.add_argument("-n", "--n-events", type=int, default=None, help="Total number of events to process across all files (default: all)")
    
    args = parser.parse_args()
    try:
        plot_distributions(args.input, args.output, args.pattern, args.n_events)
    except Exception as e:
        print(f"Error: {e}")
