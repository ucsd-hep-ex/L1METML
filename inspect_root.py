import glob
import uproot

# Directory containing the ROOT files
directory = "/ceph/cms/store/user/dprimosc/l1deepmet_data/25May15_140X_v0"

# File pattern to match
pattern = f"{directory}/perfNano_TT_PU200.root"

# Find all files matching the pattern
file_list = glob.glob(pattern)
print(f"Found {len(file_list)} files matching the pattern: {pattern}")
total_events = 0

for filename in file_list:
    print(f"Processing file: {filename}")
    # Open the ROOT file
    with uproot.open(filename) as file:
        # List all keys (content) in the file
        print("Content of the ROOT file:")
        for key in file.keys():
            print(f" - {key}")
        # Iterate over all keys to find TTrees
        for key in file.keys():
            classname = file.classname_of(key)
            if classname == "TTree":
                tree = file[key]
                num_events = tree.num_entries
                print(f"Number of events in tree '{key}': {num_events}")
                total_events += num_events

                # Print branches of the tree
                print(f"Branches in tree '{key}':")
                for branch in tree.keys():
                    print(f"   - {branch}")
    print("")  

print(f"Total number of events: {total_events}")