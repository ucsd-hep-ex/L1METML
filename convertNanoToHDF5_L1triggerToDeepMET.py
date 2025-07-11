#!/usr/bin/env python

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import uproot
# import progressbar
from tqdm import tqdm

from utils import to_np_array

"""
widgets=[
    progressbar.SimpleProgress(), ' - ', progressbar.Timer(), ' - ', progressbar.Bar(), ' - ', progressbar.AbsoluteETA()
]
"""


def deltaR(eta1, phi1, eta2, phi2):
    """calculate deltaR"""
    dphi = phi1 - phi2
    while dphi > np.pi:
        dphi -= 2 * np.pi
    while dphi < -np.pi:
        dphi += 2 * np.pi
    deta = eta1 - eta2
    return np.hypot(deta, dphi)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert NanoAOD ROOT files to HDF5 format for L1 MET ML training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Input directory containing ROOT files or single ROOT file path",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output directory for HDF5 files",
    )

    parser.add_argument(
        "-N",
        "--maxevents",
        type=int,
        default=-1,
        help="Maximum number of events to process per file (-1 for all)",
    )

    parser.add_argument(
        "--data",
        action="store_true",
        help="Input is data (not MC). Excludes generator-level variables.",
    )

    parser.add_argument(
        "--pattern",
        type=str,
        default="*.root",
        help="File pattern to match ROOT files (default: *.root)",
    )

    return parser.parse_args()


def find_root_files(input_path, pattern="*.root"):
    """Find ROOT files in input path."""
    input_path = Path(input_path)

    if input_path.is_file():
        # Single file provided
        if input_path.suffix == ".root":
            return [input_path]
        else:
            raise ValueError(f"Input file is not a ROOT file: {input_path}")

    elif input_path.is_dir():
        # Directory provided - find all ROOT files
        root_files = list(input_path.glob(pattern))
        if not root_files:
            raise ValueError(
                f"No ROOT files found in {input_path} with pattern {pattern}"
            )
        return sorted(root_files)

    else:
        raise ValueError(f"Input path does not exist: {input_path}")


def convert_single_file(input_file, output_file, maxevents=-1, is_data=False):
    """Convert a single ROOT file to HDF5."""
    print(f"Processing: {input_file} -> {output_file}")

    # Variable lists
    varList = [
        "nL1PuppiCands",
        "L1PuppiCands_pt",
        "L1PuppiCands_eta",
        "L1PuppiCands_phi",
        "L1PuppiCands_charge",
        "L1PuppiCands_pdgId",
        "L1PuppiCands_puppiWeight",
        "L1PuppiCands_dxyErr",
        "HGCal3DCl_hoe",
        "HGCal3DCl_showerlength",
        "HGCal3DCl_coreshowerlength",
    ]

    varList_mc = [
        "genMet_pt",
        "genMet_phi",
    ]

    d_encoding = {
        "L1PuppiCands_charge": {-999.0: 0, -1.0: 1, 0.0: 2, 1.0: 3},
        "L1PuppiCands_pdgId": {
            -999.0: 0,
            -211.0: 1,
            -130.0: 2,
            -22.0: 3,
            -13.0: 4,
            -11.0: 5,
            11.0: 5,
            13.0: 4,
            22.0: 3,
            130.0: 2,
            211.0: 1,
        },
    }

    if not is_data:
        varList = varList + varList_mc

    # Open ROOT file and read data
    upfile = uproot.open(input_file)
    tree = upfile["Events"].arrays(varList, entry_stop=maxevents)

    # Setup arrays
    maxNPuppi = 128
    nFeatures = 12
    maxEntries = len(tree["nL1PuppiCands"])

    X = np.zeros(shape=(maxEntries, maxNPuppi, nFeatures), dtype=float, order="F")
    Y = np.zeros(shape=(maxEntries, 2), dtype=float, order="F")

    # Extract features
    pt = to_np_array(tree["L1PuppiCands_pt"], maxN=maxNPuppi)
    eta = to_np_array(tree["L1PuppiCands_eta"], maxN=maxNPuppi)
    phi = to_np_array(tree["L1PuppiCands_phi"], maxN=maxNPuppi)
    pdgid = to_np_array(tree["L1PuppiCands_pdgId"], maxN=maxNPuppi, pad=-999)
    charge = to_np_array(tree["L1PuppiCands_charge"], maxN=maxNPuppi, pad=-999)
    puppiw = to_np_array(tree["L1PuppiCands_puppiWeight"], maxN=maxNPuppi)
    dxyErr = to_np_array(tree["L1PuppiCands_dxyErr"], maxN=maxNPuppi, pad=-999)
    hoe = to_np_array(tree["HGCal3DCl_hoe"], maxN=maxNPuppi, pad=-999)
    showerlength = to_np_array(tree["HGCal3DCl_showerlength"], maxN=maxNPuppi, pad=-999)
    coreshowerlength = to_np_array(
        tree["HGCal3DCl_coreshowerlength"], maxN=maxNPuppi, pad=-999
    )

    # Fill feature array
    X[:, :, 0] = pt
    X[:, :, 1] = pt * np.cos(phi)
    X[:, :, 2] = pt * np.sin(phi)
    X[:, :, 3] = eta
    X[:, :, 4] = phi
    X[:, :, 5] = puppiw

    # Encoding
    X[:, :, 6] = np.vectorize(d_encoding["L1PuppiCands_pdgId"].__getitem__)(
        pdgid.astype(float)
    )
    X[:, :, 7] = np.vectorize(d_encoding["L1PuppiCands_charge"].__getitem__)(
        charge.astype(float)
    )

    # Fill additional features
    X[:, :, 8] = dxyErr
    X[:, :, 9] = hoe
    X[:, :, 10] = showerlength
    X[:, :, 11] = coreshowerlength

    # Truth info
    if not is_data:
        Y[:, 0] += tree["genMet_pt"].to_numpy() * np.cos(tree["genMet_phi"].to_numpy())
        Y[:, 1] += tree["genMet_pt"].to_numpy() * np.sin(tree["genMet_phi"].to_numpy())

    # Save file
    with h5py.File(output_file, "w") as h5f:
        h5f.create_dataset("X", data=X, compression="lzf")
        h5f.create_dataset("Y", data=Y, compression="lzf")

    print(f"Saved: {output_file} ({X.shape[0]} events)")


def main():
    # Parse arguments
    args = parse_arguments()

    # Find ROOT files
    try:
        root_files = find_root_files(args.input, args.pattern)
        print(f"Found {len(root_files)} ROOT files to process")
    except ValueError as e:
        sys.exit(f"Error: {e}")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each file
    for root_file in tqdm(root_files, desc="Converting files"):
        try:
            # Generate output filename
            output_file = output_dir / f"{root_file.stem}.h5"

            # Convert file
            convert_single_file(root_file, output_file, args.maxevents, args.data)

        except Exception as e:
            print(f"Error processing {root_file}: {e}")
            continue

    print("Conversion completed!")


if __name__ == "__main__":
    main()
