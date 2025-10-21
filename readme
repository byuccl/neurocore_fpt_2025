# Neurocore FPT 2025 Artifacts

These are the artifacts used to compute the results in the Neurocore[1]
paper.

The overall workflow is broken into three parts.  First the designs
are generated using Vivado, then the designs are extracted from Vivado,
and finally the models are analyzed using the provided python script.
In the absence of Vivado, we have provided 25 already-extracted designs
from each dataset with which to evaluate the models.

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
cd datasets/dataset1_tcl_25dumps/design_0000
vivado -mode batch -source design.tcl
```

The easiest way to extract the required dump file from an implemented
checkpoint is using the following command:

```bash
vivado -mode batch -source src/dump.tcl -tclargs <path/to/checkpoint>
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

Where model is one of the following:

* gat2.pt
* sage.pt
* gcn.pt
