#!/bin/bash

# Checking accuracy of GCN, SAGE model on dataset 1, designs 4000 to 4024
# While the paper reports GATv2 results on dataset1, the model provided in 
# this repository is trained on dataset2 only. Hence, we only report GCN and SAGE.
echo "design,gcn_accuracy,sage_accuracy,gatv2_accuracy"
for i in {4000..4024}; do
    echo -n "dataset1/design_$i,"

    out=$(python src/infer.py models/gcn.pt "datasets/dataset1/design_$i/design_$i.dump" --device cpu)
    echo -n "$(echo "$out" | awk '{print $2}'),"

    out=$(python src/infer.py models/sage.pt "datasets/dataset1/design_$i/design_$i.dump" --device cpu)
    echo "$(echo "$out" | awk '{print $2}'),"
done

# Checking accuracy of GATv2 model on dataset 2, designs 4000 to 4024
for i in {4000..4024}; do
    echo -n "dataset2/design_$i,"
    out=$(python src/infer.py models/gat2_pairnorm.pt "datasets/dataset2/design_$i/design_$i.dump" --device cpu)
    echo ",,$(echo "$out" | awk '{print $2}'),"
done
