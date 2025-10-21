from itertools import combinations
from gserdes import Deserializer
from torch_geometric.data import Data
import torch
from pathlib import Path
from collections import Counter
import numpy as np

# FNV-1a 32-bit hash
class Hash:
    def __init__(self, seed=None):
        self.hash = 2166136261
        if seed:
            self.augment(seed)

    def augment(self, data):
        for b in data:
            self.hash ^= b
            self.hash *= 16777619
            self.hash &= 0xFFFFFFFF

    def float(self):
        return float(self.hash)

def oneindex(l, v):
    try:
        return l.index(v) + 1
    except ValueError:
        return 0

def onehot(index, total):
    result = [0.0] * total
    result[index] = 1.0
    return result

classes = [
    b"accumulator",
    b"axi",
    b"clk_wiz",
    b"gpio",
    b"intc",
    b"microblaze",
    b"uartlite",
]

features = 11

def color(props):
    rest = 3
#    bel = Hash(props[1]).float()
    sequential = 0.0
    combinational = 0.0
    neither = 1.0
    ff = 0.0
    lut = 0.0
    carry = 0.0
    buf = 0.0
    zero = 0.0
    one = 0.0
    mux = 0.0
    other = 1.0
    if b"LUT" in props[1]:
        combinational = 1.0
        lut = 1.0
        other = 0.0
        neither = 0.0
    if b"CARRY" in props[1]:
        combinational = 1.0
        carry = 1.0
        other = 0.0
        neither = 0.0
    if b"MUX" in props[1]:
        combinational = 1.0
        mux = 1.0
        other = 0.0
        neither = 0.0
    if b"FF" == props[1][-2:]:
        sequential = 1.0
        ff = 1.0
        other = 0.0
        neither = 0.0
    if b"BUF" in props[1]:
        combinational = 1.0
        buf = 1.0
        other = 0.0
        neither = 0.0
    if b"VCC" in props[1]:
        combinational = 1.0
        one = 1.0
        other = 0.0
        neither = 0.0
    if b"GND" in props[1]:
        combinational = 1.0
        zero = 1.0
        other = 0.0
        neither = 0.0

#    eqn = 0.0
#    if b"LUT" in props[1]:
#        eqn = Hash(props[3]).float()
#        rest = 4

#    hue = Hash()
#    for p in props[rest:]:
#        hue.augment(p)
#    hue = hue.float()

    return [
        sequential,
        combinational,
        neither,
        lut,
        ff,
        carry,
	mux,
        buf,
        zero,
        one,
        other,
    ]

def tanh_estimator(x):
    mean = np.mean(x, axis=0)
    stddev = np.std(x, axis=0)
    stddev[stddev == 0.0] = 1.0
    return (0.5 * (np.tanh(0.01 * ((x - mean) / stddev)) + 1))

def load(fin, classes):
    order = {}
    nextnode = 0
    colors = []
    truth = []
    adjacency = []
    classnum = len(classes)+1

    p = Deserializer(fin)
    nodes = p.sexp()
    for n in nodes:
        if n[0] in order:
            continue
        colors.append(color(n))
        try:
            ip = n[2].split(b"_", 2)[2]
            if ip in classes:
                truth.append(ip)
            else:
                truth.append(b"fabric")
        except IndexError:
            truth.append(b"fabric")
        order[n[0]] = nextnode
        nextnode += 1
    y_compressed = [oneindex(classes, x) for x in truth]
    y = [onehot(x, classnum) for x in y_compressed]
    tally = Counter(y_compressed)
    for i in range(classnum):
        if not tally[i]:
            tally[i] = 1
    weight = [tally.total() / tally[x] for x in range(classnum)]
    edges = [[order[left], order[right]] for [left, right] in p.sexp() if left in order and right in order]
    edges = torch.tensor(edges, dtype=torch.long).t().contiguous()
    colors = tanh_estimator(colors)
    data = Data(
        x=torch.tensor(colors, dtype=torch.float32),
        edge_index=edges,
        y=torch.tensor(y),
    )
    data.weight = torch.tensor(weight, dtype=torch.float32)
    data.order = order
    data.ip = truth
    return data


class NetlistDataset:
    def __init__(self, graphs, classes):
        self.graphs = graphs
        self.classes = classes

    @classmethod
    def new(cls, rawdir, classes):
        graphs = [(p, p.with_suffix('.dat')) for p in Path(rawdir).iterdir() if p.is_dir() and not p.name.startswith('.')]
        graphs.sort()
        return cls(graphs, classes)

    def __getitem__(self, index):
        graphs = self.graphs[index]
        if isinstance(index, slice):
            return NetlistDataset(graphs, self.classes)
        if isinstance(index, int):
            r, p = graphs
            if p.exists():
                with p.open("rb") as f:
                    return torch.load(f, weights_only=False)
            return self.process(r / f"{r.name}.dump", p)
        raise IndexError(f"The index {index} is invalid.")

    def __iter__(self):
        self.cursor = 0
        return self

    def __next__(self):
        try:
            result = self[self.cursor]
            self.cursor += 1
            return result
        except IndexError:
            raise StopIteration

    def __len__(self):
        return len(self.graphs)

    def process(self, pin, pout):
        with pin.open("rb") as fin, pout.open("wb") as fout:
            data = load(fin, self.classes)
            torch.save(data, fout)
            return data

    def hashCount(self):
        result = [Counter() for _ in range(len(self.classes) + 1)]

        for i, c in enumerate(self[0].x):
            bin = (self[0].y[i] == 1.0).nonzero(as_tuple=True)[0].item()
            result[bin][(c[0].item(), c[1].item())] += 1
        return result

    def __accumulate_overlap_matrix(self, use_key_value):
        counters = self.hashCount()
        all_keys, overlap_matrix = self.__initialize_overlap_matrix(counters)

        for key in all_keys:
            counter_indices = [i for i, c in enumerate(counters) if key in c]
            for i, j in combinations(counter_indices, 2):
                overlap_matrix[i][j] += counters[i][key] if use_key_value else 1
                overlap_matrix[j][i] += counters[j][key] if use_key_value else 1

        for i, c1 in enumerate(counters):
            # This is an expensive calculation that need only be done once per iteration.
            # It determines the keys that are unique to a class, taking in the
            # key, by checking if the key is not in any other counter.
            is_unique = lambda key: all(
                key not in c2 for j, c2 in enumerate(counters) if j != i
            )

            if use_key_value:
                value = sum(c1[key] for key in c1 if is_unique(key))
            else:
                value = sum(1 for key in c1 if is_unique(key))

            overlap_matrix[i][i] = value

        return overlap_matrix

    def __initialize_overlap_matrix(self, counters):
        all_keys = set().union(*counters)
        n_counters = len(counters)
        overlap_matrix = [[0] * n_counters for _ in range(n_counters)]
        return all_keys, overlap_matrix

    def compute_overlap_matrix_bels(self):
        """Provides a matrix of the number of bels belonging to colors shared between classes.
        Entry [i,j] in the matrix is the number of bels
        predicted to be in class i with colors shared between class i and class j.
        Diagonals are the number of bels with colors unique to a class."""
        return self.__accumulate_overlap_matrix(use_key_value=True)

    def compute_overlap_matrix_colors(self):
        """Provides a matrix of the number of unique colors in each class.
        Entry [i,j] in the matrix is the number of colors shared between class i and class j.
        Diagonals are the number of colors unique to a class."""
        return self.__accumulate_overlap_matrix(use_key_value=False)
