import torch
import gcn
import sage
import gat2
import netlist
import argparse
from pathlib import Path
import numpy as np
from bidict import bidict
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

def rehydrate_model(name, model_water, checkpoint):
    if name == "gat2":
        return model_water(checkpoint['classes'], checkpoint['width'], checkpoint['depth'], checkpoint['activation'], checkpoint['state'], checkpoint['pairnorm'], checkpoint['heads'])
    else:
        return model_water(checkpoint['classes'], checkpoint['width'], checkpoint['depth'], checkpoint['activation'], checkpoint['state'], checkpoint['pairnorm'])

def do(model, input, device):
    hydration = {"gat2": gat2.rehydrate, "sage": sage.rehydrate, "gcn": gcn.rehydrate}

    checkpoint = torch.load(model, map_location = torch.device(device), weights_only=False)
    model_name = checkpoint['model']
    model_water = hydration[model_name]
    gnn = rehydrate_model(model_name, model_water, checkpoint).to(device)
    gnn.eval()
    g = None
    with input.open('rb') as fin:
        g = netlist.load(fin, checkpoint['classes'])
    results = gnn(g.x, g.edge_index).argmax(dim=1)
    names = bidict(g.order)
    y = g.y.argmax(dim=1)
    print(f"bel\tip_inferred\tip_actual")
    for i, r in enumerate(results):
        l = None
        r = r.item()
        l = checkpoint['classes'][r].decode('utf-8')

        print(f"{names.inv[i].decode('utf-8')}\t{l}\t{g.ip[i].decode('utf-8')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('input')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', choices=['cpu', 'cuda'], help="Device to run on (default: auto-detect)")
    args = parser.parse_args()
    do(args.model, Path(args.input), args.device)

