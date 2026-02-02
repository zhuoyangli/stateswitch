import numpy as np
from nilearn import datasets

def verify_parcel_labels():
    print("Fetching Schaefer 400 (17 Networks) Atlas...")
    # Load the atlas
    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=17)
    
    # Convert byte labels to strings
    # The atlas['labels'] list usually includes "Background" as index 0.
    all_labels = [l.decode() if hasattr(l, 'decode') else str(l) for l in atlas['labels']]
    roi_labels = [l for l in all_labels if l != 'Background']
    
    # INDICES TO CHECK
    # Assuming these are 1-based indices (standard for atlas lookups)
    
    queries = {
        "left/dmPFC (161-166, 175-180)": [161, 162, 163, 164, 165, 166, 175, 176, 177, 178, 179, 180],
        "rmPFC (368-373)": [368, 369, 370, 371, 372, 373],
        "AG (Mixed)": [149, 150, 173, 174, 188, 359, 360, 385, 386],
        "Combined right parahippocampal cortex and temporal pole ROIs": [389, 319, 320, 322, 376]
    }
    keywords = {
        "Precuneus": ['pCun'],
        "Cingulate": ['Cing'],
        "Salience": ['Sal']
    }

    print("\n" + "="*60)
    print(f"{'INDEX':<8} | {'LABEL'}")
    print("="*60)

    for group_name, indices in queries.items():
        print(f"\n--- {group_name} ---")
        for idx in indices:
            # Handle potential 0 vs 1 indexing confusion
            # If the atlas has 'Background' at 0, then label 1 is index 1.
            # If your mate's "1" means the first ROI, that's index 1 here.
            
            try:
                # We access idx directly because nilearn usually puts 'Background' at 0
                # So ROI #1 is at all_labels[1]
                label = roi_labels[idx-1] 
                print(f"{idx:<8} | {label}")
            except IndexError:
                print(f"{idx:<8} | ** INDEX OUT OF BOUNDS **")
    
    for keyword, keys in keywords.items():
        print(f"\n--- Parcels containing '{keyword}' ---")
        for i, label in enumerate(roi_labels, start=1):
            if any(key in label for key in keys):
                print(f"{i:<8} | {label}")

if __name__ == "__main__":
    verify_parcel_labels()