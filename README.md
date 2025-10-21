# Neurocore FPT 2025

This repository contains:
* Two datasets of 5000 designs each, as described in the FPT 2025 paper.  Each design has:
    * A Tcl script that can be run in Vivado 2022.2 to generate a corresponding design.
    * A .dump file containing the full implementation details of the design.
* Three trained models for detecting and locating IP in FPGA designs.


## Unpacking the datasets

1. Unzip the GATv2 model (the other two models are small enough to be stored directly in the repository):
   ```bash
   make unzip_gat2_model
   ```

1. Unpack the datasets 
   ```bash
   make unpack_datasets
   ```
