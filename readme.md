# Neurocore FPT 2025 Artifacts

These are the artifacts used to compute the results in the Neurocore FPT
paper.

This repository contains:
* Two datasets of 5000 designs each (4000 training designs, 1000 test designs), as described in the FPT 2025 paper.  Each design has a Tcl script that can be run in Vivado 2022.2 to generate a corresponding design.
* The first 25 test designs (4000--4024) have a *.dump file included for quick evaluation. The .dump file is a full dump of the implementation details of the design.
* Three trained models for detecting and locating IP in FPGA designs.


## Overall Workflow to Validate Paper Results
The overall workflow is broken into three parts.  The first two steps can be skipped for the first 25 test designs, for which have provided pre-extracted dump files.
1. First the design is built using Vivado, which runs the provided *design.tcl* script, and performs synthesis and implementation to compile the design.  
1. The implemented design details are extracted from Vivado to produce a `.dump` file.
1. The resulting `.dump` file is analyzed using the pre-trained GNN models with the provided python script, which will report the accuracy of the model on the design.

**These steps can then be repeated for multiple designs, and the accuracy results can be compared to those reported in the paper.**

Given the long runtime to compile all 1000 test designs in Vivado, we recommend running a subset of the test designs, computing the average accuracy, and verifying similarity to the reported results.

## Unpacking the model and datasets

1. Unzip the GATv2 model (the other two models are small enough to be stored directly in the repository):
   ```bash
   make unzip_gat2_model
   ```

1. Unpack the datasets 
   ```bash
   make unpack_datasets
   ```

## Compiling a Design and Generating its Dump File

To compile a design in Vivado (**Make sure to use 2022.2**), run the following:
```bash
cd datasets/dataset1/design_0000
vivado -mode batch -source design.tcl
```

The easiest way to extract the required dump file from an implemented
checkpoint is using the following command:

```bash
vivado -mode batch -source src/dump.tcl -tclargs <path/to/checkpoint_without_extension>
```

This path cannot have the trailing .dcp extension, as the dump script
uses the same path to compute both the dcp name as well as the name
for the resulting dump file.  The dump file will be placed in the same
directory as the checkpoint file.


## Model Inference

We have developed the models and the inference script using python 3.12.
To install all necessary dependencies, we recommend setting up a virtual
environment with the command:

```bash
python -m venv .venv
```

Then the environment can be activated using:

```bash
. .venv/bin/activate
```

and then populated as follows:

```bash
pip install -r requirements.txt
```

The inference script should then be run:

```bash
python src/infer.py <model> <path_to.dump>
```

where model is one of the following:

* gat2.pt
* sage.pt
* gcn.pt
