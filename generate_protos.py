#!/usr/bin/env python3
"""
Script to generate Python gRPC stubs from protobuf definitions.
"""

import subprocess
import sys
from pathlib import Path

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
    
    print("\nAll protobuf stubs generated successfully!")
    return True

if __name__ == "__main__":
    success = generate_protos()
    sys.exit(0 if success else 1)