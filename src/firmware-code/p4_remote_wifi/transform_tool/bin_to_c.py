import os

def bin_to_c():
    espdl_path = "transform_tool/cnn_model.espdl"
    header_path = "main/model_data.h"
    source_path = "main/model_data.c"
    
    # Allow overriding via arguments
    import sys
    if len(sys.argv) > 1:
        espdl_path = sys.argv[1]
        
    if not os.path.exists(espdl_path):
        print(f"Error: {espdl_path} does not exist.")
        return
        
    with open(espdl_path, "rb") as f:
        data = f.read()
        
    length = len(data)
    print(f"Reading {espdl_path} ({length} bytes)...")
    
    # Generate model_data.h
    print(f"Generating {header_path}...")
    with open(header_path, "w") as f:
        f.write('''/* Automatically generated from quadrant_model.espdl. Do not edit. */
#ifndef MODEL_DATA_H
#define MODEL_DATA_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

extern const uint8_t model_espdl[];
extern const unsigned int model_espdl_len;

#ifdef __cplusplus
}
#endif

#endif /* MODEL_DATA_H */
''')

    # Generate model_data.c
    print(f"Generating {source_path}...")
    with open(source_path, "w") as f:
        f.write('''/* Automatically generated from quadrant_model.espdl. Do not edit. */
#include "model_data.h"

const uint8_t model_espdl[] __attribute__((aligned(16))) = {
''')
        # Write bytes in chunks of 12 per line
        bytes_str = []
        for i, b in enumerate(data):
            bytes_str.append(f"0x{b:02x}")
            if (i + 1) % 12 == 0:
                f.write("    " + ", ".join(bytes_str) + ",\n")
                bytes_str = []
        if bytes_str:
            f.write("    " + ", ".join(bytes_str) + "\n")
            
        f.write(f'''}};

const unsigned int model_espdl_len = {length};
''')
        
    print("Files generated successfully.")

if __name__ == '__main__':
    bin_to_c()
