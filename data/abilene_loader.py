"""
Abilene network traffic loader.
Parses XML demand matrices and aggregates per-node in/out flow.
Output shape: (T, N, 2) where N=11, F=2 (in_flow, out_flow) in Mbit/s.
"""
import os
import glob
import numpy as np
from xml.etree import ElementTree as ET


NODES = [
    'ATLAM5', 'ATLAng', 'CHINng', 'DNVRng', 'HSTNng',
    'IPLSng', 'KSCYng', 'LOSAng', 'NYCMng', 'SNVAng', 'WASHng'
]
NODE_INDEX = {n: i for i, n in enumerate(NODES)}


def parse_xml(path: str) -> np.ndarray:
    """Parse one XML file, return (11, 2) array [in_flow, out_flow]."""
    tree = ET.parse(path)
    root = tree.getroot()
    ns = {'s': 'http://sndlib.zib.de/network'}

    features = np.zeros((11, 2), dtype=np.float32)
    for demand in root.findall('.//s:demand', ns):
        src = demand.find('s:source', ns).text.strip()
        tgt = demand.find('s:target', ns).text.strip()
        val = float(demand.find('s:demandValue', ns).text.strip())
        if src in NODE_INDEX and tgt in NODE_INDEX:
            features[NODE_INDEX[src], 1] += val  # out_flow
            features[NODE_INDEX[tgt], 0] += val  # in_flow
    return features


def load_abilene(data_dir: str, max_files: int = None) -> np.ndarray:
    """
    Load all XML files sorted by timestamp.
    Returns (T, 11, 2) numpy array.
    """
    pattern = os.path.join(data_dir, 'demandMatrix-abilene-zhang-5min-*.xml')
    files = sorted(glob.glob(pattern))
    if max_files is not None:
        files = files[:max_files]
    print(f"Loading {len(files)} files...")
    snapshots = [parse_xml(f) for f in files]
    arr = np.stack(snapshots)  # (T, 11, 2)
    print(f"Loaded shape: {arr.shape}")
    return arr


if __name__ == '__main__':
    data_dir = os.path.join(os.path.dirname(__file__), 'abilene')
    arr = load_abilene(data_dir, max_files=3000)  # ~10 days
    out_path = os.path.join(os.path.dirname(__file__), 'abilene_data.npy')
    np.save(out_path, arr)
    print(f"Saved to {out_path}")
    print(f"in_flow mean per node: {arr[:,:,0].mean(axis=0).round(3)}")
    print(f"out_flow mean per node: {arr[:,:,1].mean(axis=0).round(3)}")
