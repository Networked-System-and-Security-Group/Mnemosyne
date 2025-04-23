import os
import re
import subprocess
import argparse
from collections import defaultdict

def get_sm_cubin_files(input_dir, sm_version):
    sm_cubins = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".cubin") and sm_version in file:
                sm_cubins.append(os.path.join(root, file))
    return sm_cubins

def run_cuobjdump(cubin_file):
    try:
        result = subprocess.run(
            ["cuobjdump", "-elf", cubin_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"Error running cuobjdump on {cubin_file}: {result.stderr}")
        return result.stdout
    except FileNotFoundError:
        raise RuntimeError("cuobjdump not found. Please ensure it's installed and in your PATH.")

def parse_cuobjdump_output(cubin_file, cuobjdump_output):
    func_pattern = re.compile(r"\.nv\.info\.(.*?)\n")
    param_pattern = re.compile(
        r"EIATTR_KPARAM_INFO.*?Ordinal\s+:\s+0x([0-9a-fA-F]+).*?Size\s+:\s+0x([0-9a-fA-F]+)", re.S
    )
    
    func_map = defaultdict(lambda: {"cubin_file": os.path.abspath(cubin_file), "param_count": 0, "param_lengths": []})
    
    functions = re.split(r"(?=\.nv\.info\.)", cuobjdump_output)
    
    for func_block in functions:
        func_match = func_pattern.search(func_block)
        if func_match:
            func_name = func_match.group(1).strip()
            
            param_matches = param_pattern.findall(func_block)
            if param_matches:
                params = sorted(
                    [(int(ordinal, 16), int(size, 16)) for ordinal, size in param_matches],
                    key=lambda x: x[0]
                )
                
                param_lengths = [size for _, size in params]
                param_count = len(param_lengths)
                
                func_map[func_name]["param_count"] = param_count
                func_map[func_name]["param_lengths"] = param_lengths
    
    return func_map

def main(input_dir, sm_version):
    sm_cubins = get_sm_cubin_files(input_dir, sm_version)
    if not sm_cubins:
        print(f"No {sm_version} cubin files found.")
        return

    print(f"Found {len(sm_cubins)} {sm_version} cubin files.")

    all_func_info = defaultdict(dict)
    for cubin_file in sm_cubins:
        print(f"Processing {cubin_file}...")
        
        try:
            cuobjdump_output = run_cuobjdump(cubin_file)
            func_info = parse_cuobjdump_output(cubin_file, cuobjdump_output)
            
            for func_name, info in func_info.items():
                all_func_info[func_name] = info
        
        except RuntimeError as e:
            print(f"Error processing {cubin_file}: {e}")
    
    print("\nFunction Map:")
    for func_name, info in all_func_info.items():
        print(f"Function: {func_name}")
        print(f"  Cubin File: {info['cubin_file']}")
        print(f"  Parameter Count: {info['param_count']}")
        print(f"  Parameter Lengths: {info['param_lengths']}")
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process .cubin files for a specific SM version.")
    parser.add_argument("--input_dir", type=str, default=".", help="Directory containing .cubin files")
    parser.add_argument("--sm_version", type=str, default="sm_60", help="SM version to filter .cubin files (e.g., sm_60, sm_70)")
    
    args = parser.parse_args()
    
    main(args.input_dir, args.sm_version)