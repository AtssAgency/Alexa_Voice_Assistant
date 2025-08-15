#!/usr/bin/env python3
"""
Script to generate Python gRPC stubs from protobuf definitions.
Run this script whenever .proto files are modified.
"""

import subprocess
import sys
from pathlib import Path
import re

def fix_imports_in_file(file_path: Path):
    """Fix absolute imports to relative imports in generated files"""
    content = file_path.read_text()
    
    # Fix imports for pb2 files
    content = re.sub(r'^import (\w+_pb2) as', r'from . import \1 as', content, flags=re.MULTILINE)
    
    # Fix imports for pb2_grpc files  
    content = re.sub(r'^import (\w+_pb2) as', r'from . import \1 as', content, flags=re.MULTILINE)
    
    file_path.write_text(content)

def generate_protos():
    """Generate Python gRPC stubs from all .proto files."""
    proto_dir = Path("services/proto")
    output_dir = Path("services/proto")
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all .proto files
    proto_files = list(proto_dir.glob("*.proto"))
    
    if not proto_files:
        print("No .proto files found in services/proto/")
        return False
    
    print(f"Found {len(proto_files)} proto files:")
    for proto_file in proto_files:
        print(f"  - {proto_file}")
    
    # Generate Python stubs for each proto file
    for proto_file in proto_files:
        cmd = [
            sys.executable, "-m", "grpc_tools.protoc",
            f"--proto_path={proto_dir}",
            f"--python_out={output_dir}",
            f"--grpc_python_out={output_dir}",
            "--pyi_out={}".format(output_dir),
            str(proto_file)
        ]
        
        print(f"\nGenerating stubs for {proto_file.name}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error generating stubs for {proto_file.name}:")
            print(result.stderr)
            return False
        else:
            print(f"Successfully generated stubs for {proto_file.name}")
    
    # Fix imports in generated files
    print("\nFixing imports in generated files...")
    for pb2_file in output_dir.glob("*_pb2.py"):
        fix_imports_in_file(pb2_file)
        print(f"Fixed imports in {pb2_file.name}")
    
    for grpc_file in output_dir.glob("*_pb2_grpc.py"):
        fix_imports_in_file(grpc_file)
        print(f"Fixed imports in {grpc_file.name}")
    
    print("\nAll protobuf stubs generated successfully!")
    return True

if __name__ == "__main__":
    success = generate_protos()
    sys.exit(0 if success else 1)