import os
import subprocess
import re
import argparse
import csv
from tqdm import tqdm

def run_speed_test(file_path, base_path):
    try:
        if base_path:
            result = subprocess.run(
                ["python", "speed.py", file_path, "--base-path", base_path],
                capture_output=True,
                text=True,
                check=True
            )
        else:
            result = subprocess.run(
                ["python", "speed.py", file_path],
                capture_output=True,
                text=True,
                check=True
            )
        output = result.stdout

        # Throughput
        th_match = re.search(r"Throughput \(token/s\):\s*([\d.]+)", output)
        throughput = float(th_match.group(1)) if th_match else None

        # MAT
        mat_match = re.search(r"MAT:\s*([\d.]+)", output)
        mat = float(mat_match.group(1)) if mat_match else None
        
        # Speedup
        speedup_match = re.search(r"Speedup:\s*([\d.]+)", output)
        speedup = float(speedup_match.group(1)) if speedup_match else None

        return mat, throughput, speedup
    except subprocess.CalledProcessError as e:
        print(f"Failed to execute with {file_path}: {e}")
    return None, None, None

def batch_test(folder_path, base_path):
    results = {}

    for fname in tqdm(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, fname)
        if os.path.isfile(file_path):
            mat, throughput, speedup = run_speed_test(file_path, base_path)
            if throughput is not None:
                results[fname] = (mat, throughput, speedup)

    # Sort in descending order of Throughput
    sorted_results = sorted(results.items(), key=lambda x: x[1][1], reverse=True)

    best_file, (best_mat, max_throughput, max_speedup) = (sorted_results[0] if sorted_results else (None, (None, None, None)))

    return sorted_results, best_file, best_mat, max_throughput

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch test speed.py outputs and extract MAT and max Throughput (Speedup if baseline provided)")
    parser.add_argument("folder", help="The folder containing files to be tested.")
    parser.add_argument(
        "--base-path",
        default=None,
        type=str,
        help="The file path of evaluated baseline.",
    )
    args = parser.parse_args()

    sorted_results, best_file, best_mat, max_throughput = batch_test(args.folder, args.base_path)

    print("Results (sorted by Throughput):")
    if args.base_path:
        for f, (mat, t, s) in sorted_results:
            print(f"MAT={mat}, Speedup={s}, Throughput={t}, File={f}")
    else:
        for f, (mat, t, _) in sorted_results:
            print(f"MAT={mat}, Throughput={t}, File={f}")

    if best_file:
        print("\nMax Throughput:")
        print(f"MAT={best_mat}, Throughput={max_throughput}, File={best_file}")
    else:
        print("No results")
