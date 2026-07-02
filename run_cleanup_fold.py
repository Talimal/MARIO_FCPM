import os
import argparse
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor

def get_subdirectories(root_dir, depth=1):
    """
    Returns a list of all subdirectories up to a certain depth relative to root_dir.
    If depth is 1, returns immediate children.
    If depth is 2, returns immediate children AND their children.
    """
    subdirs = []
    root_depth = root_dir.rstrip(os.sep).count(os.sep)
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        current_depth = dirpath.rstrip(os.sep).count(os.sep) - root_depth
        if current_depth < depth:
            for dirname in dirnames:
                subdirs.append(os.path.join(dirpath, dirname))
        else:
            # Don't recurse deeper than needed for the list
            # But os.walk recurses anyway. We just don't add them.
            # Optimization: modify dirnames in-place to prevent recursion
            dirnames[:] = [] 
            
    return subdirs

def delete_path(path):
    """ Deletes a file or directory safely. """
    try:
        if os.path.isfile(path) or os.path.islink(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)
        return True
    except Exception as e:
        print(f"Error deleting {path}: {e}")
        return False

def cleanup_fold_parallel(fold_dir):
    """
    Deletes the fold directory using parallel execution for efficiency.
    Targeting depth 2 for parallelization as requested.
    """
    print(f"--- Starting Parallel Cleanup for Fold: {fold_dir} ---")
    
    if not os.path.exists(fold_dir):
        print(f"Warning: Fold directory does not exist: {fold_dir}")
        return

    # Basic safety check
    if "results" in os.path.basename(fold_dir):
        print(f"CRITICAL ERROR: Attempted to delete a 'results' directory: {fold_dir}. ABORTING.")
        sys.exit(1)
        
    start_time = time.time()
    
    # Strategy:
    # 1. List all items at Depth 2. These are likely the heavy 'mining' or 'feature_matrix' folders.
    # 2. Delete them in parallel.
    # 3. Then delete the root fold_dir (which should be mostly empty or contain light files).
    
    # Actually, simpler strategy:
    # Get immediate children of the fold (Depth 1).
    # Then for each child, get ITS children (Depth 2).
    # Submit all Depth 2 items to the pool.
    
    items_to_delete = []
    
    # Get breadth-first list of items at depth 0 (files in root), depth 1 (abs folders), depth 2 (mining folders)
    # We want to delete from the bottom up (max depth) to avoid conflicts?
    # No, shutil.rmtree doesn't care if we delete a parent, but we want to parallelize the heavy lifting.
    # If we delete 'fold/abs_1', it deletes 'fold/abs_1/mine_1' etc serially.
    # So we want to explicitly call delete on 'fold/abs_1/mine_1', 'fold/abs_1/mine_2' in parallel.
    
    try:
        # Get immediate subdirectories (Level 1 - e.g. Abstraction settings)
        level1_dirs = [os.path.join(fold_dir, d) for d in os.listdir(fold_dir) if os.path.isdir(os.path.join(fold_dir, d))]
        
        level2_items = []
        for d1 in level1_dirs:
            try:
                # Level 2 - Mining settings / Model runs
                items = [os.path.join(d1, d) for d in os.listdir(d1)]
                level2_items.extend(items)
            except OSError:
                pass
                
        print(f"Found {len(level1_dirs)} Level 1 directories.")
        print(f"Found {len(level2_items)} Level 2 items to delete in parallel.")
        
        # Use ThreadPoolExecutor
        # Max workers: default is likely fine, or 4-8. Deletion is I/O bound.
        with ThreadPoolExecutor(max_workers=8) as executor:
            # First, parallelize Level 2 deletion (the heavy stuff)
            futures = [executor.submit(delete_path, item) for item in level2_items]
            # We explicitly wait for these to finish?
            # Actually, `executor.__exit__` waits. But we want to confirm before moving to Level 1.
            for f in futures:
                f.result() # Wait
                
        print("Level 2 items deleted.")
        
        # Now delete the rest (Level 1 dirs, and files in root)
        # Just nuke the whole fold_dir now, it should be fast/light
        shutil.rmtree(fold_dir)
        print(f"Root fold directory deleted: {fold_dir}")
        
    except Exception as e:
        print(f"ERROR during parallel cleanup: {e}")
        # Fallback to standard delete if something weird happened
        if os.path.exists(fold_dir):
            print("Attempting standard fallback deletion...")
            shutil.rmtree(fold_dir)

    print(f"Cleanup finished in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cleanup experiment fold directory.")
    parser.add_argument("--fold_dir", required=True, help="Path to the fold directory to delete.")
    
    args = parser.parse_args()
    
    cleanup_fold_parallel(args.fold_dir)
    sys.exit(0)
