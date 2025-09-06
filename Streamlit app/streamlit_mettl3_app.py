# streamlit_app.py
# DeepCGASPred – Streamlit UI (three routes)
#   A) Docked SDF(s) -> PLEC4092 + Avalon512 -> predict
#   B) MOL2 (undocked) -> smina -> SDF -> features -> predict
#   C) SMILES -> obabel (3D+min) -> smina -> SDF -> features -> predict
#
# Run:
#   pip install streamlit numpy pandas torch rdkit-pypi oddt requests
#   streamlit run streamlit_app.py
#
# External tools:
#   - smina (CLI) for docking
#   - obabel (Open Babel CLI) for SMILES -> MOL2 with 3D + minimization

from __future__ import annotations

import os
import sys
import io
import zipfile
import shutil
import tempfile
import subprocess
from typing import List, Optional, Tuple, Sequence

import requests
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import streamlit as st
from rdkit import Chem
from rdkit.Avalon import pyAvalonTools
from rdkit import DataStructs
from PIL import Image


# =============== General settings (avoid oversubscription) ===============
def set_low_threads(n: int = 1) -> None:
    for k in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "RDKIT_THREADCOUNT",
    ):
        os.environ[k] = str(n)

set_low_threads(1)


# =============== Model definition (same as CLI) ===============
drop_out_rating: float = 0.001

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, input_dim, n_heads, ouput_dim=None):
        super(MultiHeadAttention, self).__init__()
        self.d_k = self.d_v = input_dim // n_heads
        self.n_heads = n_heads
        self.ouput_dim = input_dim if ouput_dim is None else ouput_dim
        self.W_Q = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_K = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_V = torch.nn.Linear(input_dim, self.d_v * self.n_heads, bias=False)
        self.fc = torch.nn.Linear(self.n_heads * self.d_v, self.ouput_dim, bias=False)

    def forward(self, X):
        Q = self.W_Q(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        K = self.W_K(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        V = self.W_V(X).view(-1, self.n_heads, self.d_v).transpose(0, 1)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        attn = torch.nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).reshape(-1, self.n_heads * self.d_v)
        output = self.fc(context)
        return output

class EncoderLayer(torch.nn.Module):
    def __init__(self, input_dim, n_heads):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(input_dim, n_heads)
        self.AN1 = torch.nn.LayerNorm(input_dim)
        self.l1 = torch.nn.Linear(input_dim, input_dim)
        self.AN2 = torch.nn.LayerNorm(input_dim)

    def forward(self, X):
        output = self.attn(X)
        X = self.AN1(output + X)
        output = self.l1(X)
        X = self.AN2(output + X)
        return X

class feature_encoder(torch.nn.Module):
    def __init__(self, vector_size, n_heads, n_layers):
        super(feature_encoder, self).__init__()
        self.layers = torch.nn.ModuleList([EncoderLayer(vector_size, n_heads) for _ in range(n_layers)])
        self.AN = torch.nn.LayerNorm(vector_size)
        self.l1 = torch.nn.Linear(vector_size, vector_size // 2)
        self.bn1 = torch.nn.BatchNorm1d(vector_size // 2)
        self.l2 = torch.nn.Linear(vector_size // 2, vector_size // 4)
        self.l3 = torch.nn.Linear(vector_size // 4, vector_size // 2)
        self.bn3 = torch.nn.BatchNorm1d(vector_size // 2)
        self.l4 = torch.nn.Linear(vector_size // 2, vector_size)
        self.dr = torch.nn.Dropout(drop_out_rating)
        self.ac = gelu

    def forward(self, X):
        for layer in self.layers:
            X = layer(X)
        X1 = self.AN(X)
        X2 = self.dr(self.bn1(self.ac(self.l1(X1))))
        X3 = self.l2(X2)
        X4 = self.dr(self.bn3(self.ac(self.l3(self.ac(X3)))))
        X5 = self.l4(X4)
        return X1, X2, X3, X5

class feature_encoder2(torch.nn.Module):
    def __init__(self, vector_size):
        super(feature_encoder2, self).__init__()
        self.l1 = torch.nn.Linear(vector_size, vector_size // 2)
        self.bn1 = torch.nn.BatchNorm1d(vector_size // 2)
        self.l2 = torch.nn.Linear(vector_size // 2, vector_size // 4)
        self.bn2 = torch.nn.BatchNorm1d(vector_size // 4)
        self.dr = torch.nn.Dropout(drop_out_rating)
        self.ac = gelu

    def forward(self, X):
        X = self.dr(self.bn1(self.ac(self.l1(X))))
        X = self.dr(self.bn2(self.ac(self.l2(X))))
        return X

class Model(torch.nn.Module):
    def __init__(self, input_dim_A, input_dim_B, n_heads, n_layers, event_num):
        super(Model, self).__init__()
        self.input_dim_A = input_dim_A
        self.input_dim_B = input_dim_B
        self.drugEncoderA = feature_encoder(input_dim_A, n_heads, n_layers)
        self.drugEncoderB = feature_encoder(input_dim_B, n_heads, n_layers)

        self.feaEncoder1_3_input_dim = input_dim_A + input_dim_B // 4
        self.feaEncoder3_1_input_dim = input_dim_B + input_dim_A // 4
        self.feaEncoder2_input_dim = input_dim_A // 2 + input_dim_B // 2

        self.feaEncoder1 = feature_encoder2(self.feaEncoder1_3_input_dim)
        self.feaEncoder2 = feature_encoder2(self.feaEncoder2_input_dim)
        self.feaEncoder3 = feature_encoder2(self.feaEncoder3_1_input_dim)

        self.feaEncoder1_3_output_dim = self.feaEncoder1_3_input_dim // 4
        self.feaEncoder3_1_output_dim = self.feaEncoder3_1_input_dim // 4
        self.feaEncoder2_output_dim = self.feaEncoder2_input_dim // 4

        self.feaFui_input_dim = (
            self.feaEncoder1_3_output_dim +
            self.feaEncoder3_1_output_dim +
            self.feaEncoder2_output_dim +
            input_dim_A // 4 +
            input_dim_B // 4
        )
        self.feaFui = feature_encoder2(self.feaFui_input_dim)
        self.linear_input_dim = self.feaFui_input_dim // 4 + self.feaFui_input_dim

        self.l1 = torch.nn.Linear(self.linear_input_dim, self.linear_input_dim // 2)
        self.bn1 = torch.nn.BatchNorm1d(self.linear_input_dim // 2)
        self.l2 = torch.nn.Linear(self.linear_input_dim // 2, 1)

        self.ac = gelu
        self.dr = torch.nn.Dropout(drop_out_rating)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, XA, XB):
        XA1, XA2, XA3, XAC = self.drugEncoderA(XA)
        XB1, XB2, XB3, XBC = self.drugEncoderB(XB)

        X1 = torch.cat((XA1, XB3), 1)
        X2 = torch.cat((XA2, XB2), 1)
        X3 = torch.cat((XA3, XB1), 1)

        X1 = self.feaEncoder1(X1)
        X2 = self.feaEncoder2(X2)
        X3 = self.feaEncoder3(X3)

        XC = torch.cat((X1, X2, X3, XA3, XB3), 1)
        XC = self.feaFui(XC)

        X = torch.cat((XA3, XB3, X1, X2, X3, XC), 1)
        X = self.dr(self.bn1(self.ac(self.l1(X))))
        X = self.l2(X)
        X = self.sigmoid(X)
        return X, XC, torch.cat((XAC, XBC), 1), XAC, XBC


# =============== Safe checkpoint loader (bytes in, Model out) ===============
@st.cache_resource(show_spinner=False)
def load_checkpoint(model_bytes: bytes,
                    device: torch.device,
                    input_dim_A: int, input_dim_B: int,
                    n_heads: int = 1, n_layers: int = 1, event_num: int = 1) -> Model:
    model = Model(input_dim_A=input_dim_A, input_dim_B=input_dim_B,
                  n_heads=n_heads, n_layers=n_layers, event_num=event_num).to(device)
    # Try safe mode (PyTorch >= 2.4)
    try:
        bio = io.BytesIO(model_bytes)
        ckpt = torch.load(bio, map_location=device, weights_only=True)
    except TypeError:
        # Older versions fallback (may warn)
        bio = io.BytesIO(model_bytes)
        ckpt = torch.load(bio, map_location=device)

    if isinstance(ckpt, dict) and any(k in ckpt for k in ("model_state_dict", "state_dict")):
        state = ckpt.get("model_state_dict", ckpt.get("state_dict"))
    else:
        state = ckpt

    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", ""): v for k, v in state.items()}

    missing_unexpected = model.load_state_dict(state, strict=False)
    if missing_unexpected.missing_keys or missing_unexpected.unexpected_keys:
        st.warning(f"Non-strict state_dict mapping.\n"
                   f"Missing: {missing_unexpected.missing_keys}\n"
                   f"Unexpected: {missing_unexpected.unexpected_keys}")
    model.eval()
    return model


# =============== File helpers ===============
def write_upload_to_path(uploaded_file, dest_path: str) -> str:
    with open(dest_path, "wb") as w:
        w.write(uploaded_file.getbuffer())
    return dest_path

def unzip_if_needed(path: str, dest_dir: str, ext: str = ".sdf") -> List[str]:
    if not path.lower().endswith(".zip"):
        return [path] if path.lower().endswith(ext) else []
    os.makedirs(dest_dir, exist_ok=True)
    with zipfile.ZipFile(path, "r") as z:
        z.extractall(dest_dir)
    out = []
    for root, _, files in os.walk(dest_dir):
        for fn in files:
            if fn.lower().endswith(ext):
                out.append(os.path.join(root, fn))
    return sorted(out)


# =============== Docking box helpers ===============
def compute_box_from_reference(ref_path: str, padding: float = 6.0) -> Tuple[dict, dict]:
    ext = os.path.splitext(ref_path)[1].lower()
    m = None
    try:
        if ext == ".mol2":
            m = Chem.MolFromMol2File(ref_path, sanitize=True, removeHs=False)
        elif ext == ".sdf":
            suppl = Chem.SDMolSupplier(ref_path, removeHs=False, sanitize=True)
            m = suppl[0] if len(suppl) else None
        elif ext == ".pdb":
            m = Chem.MolFromPDBFile(ref_path, sanitize=True, removeHs=False)
    except Exception:
        m = None
    if m is None or m.GetNumConformers() == 0:
        raise ValueError("Failed to read reference ligand with RDKit for box.")
    conf = m.GetConformer()
    xs, ys, zs = [], [], []
    for i in range(m.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        xs.append(pos.x); ys.append(pos.y); zs.append(pos.z)
    minX, maxX = min(xs), max(xs)
    minY, maxY = min(ys), max(ys)
    minZ, maxZ = min(zs), max(zs)
    center = {
        "center_x": (maxX + minX) / 2.0,
        "center_y": (maxY + minY) / 2.0,
        "center_z": (maxZ + minZ) / 2.0,
    }
    size = {
        "size_x": (maxX - minX) + 2 * padding,
        "size_y": (maxY - minY) + 2 * padding,
        "size_z": (maxZ - minZ) + 2 * padding,
    }
    return center, size


# =============== SMILES -> MOL2 via Open Babel ===============
def smiles_to_mol2_files(smiles: Sequence[Tuple[str, str]], out_dir: str,
                         obabel_bin: str = "obabel", steps: int = 500) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, smi in smiles:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in (name or "mol"))
        out = os.path.join(out_dir, f"{safe}.mol2")
        # First try with shell quoting (lets us set title via :-'smiles')
        cmd = [
            obabel_bin, f"-:'{smi}'", "-O", out,
            "--gen3d", "--minimize", "--ff", "mmff94s", "--steps", str(steps)
        ]
        p = subprocess.run(" ".join(cmd), shell=True, capture_output=True, text=True)
        if p.returncode != 0 or not os.path.isfile(out):
            # Fallback without shell quoting
            cmd2 = [obabel_bin, "-:", smi, "-O", out, "--gen3d", "--minimize",
                    "--ff", "mmff94s", "--steps", str(steps)]
            p2 = subprocess.run(cmd2, capture_output=True, text=True)
            if p2.returncode != 0 or not os.path.isfile(out):
                continue
        paths.append(out)
    return paths


# =============== smina docking ===============
def run_smina_single(lig_path: str, receptor_pdb: str, center: dict, size: dict,
                     out_sdf: str, smina_bin: str = "smina", exhaustiveness: int = 8,
                     num_modes: int = 1, extra_args: Optional[List[str]] = None) -> bool:
    args = [
        smina_bin, "-r", receptor_pdb, "-l", lig_path, "-o", out_sdf,
        "--center_x", str(center["center_x"]),
        "--center_y", str(center["center_y"]),
        "--center_z", str(center["center_z"]),
        "--size_x", str(size["size_x"]),
        "--size_y", str(size["size_y"]),
        "--size_z", str(size["size_z"]),
        "--exhaustiveness", str(exhaustiveness),
        "--num_modes", str(num_modes),
    ]
    if extra_args:
        args.extend(map(str, extra_args))
    p = subprocess.run(args, capture_output=True, text=True)
    return p.returncode == 0 and os.path.isfile(out_sdf) and os.path.getsize(out_sdf) > 0

def dock_many(lig_paths: List[str], receptor_pdb: str, center: dict, size: dict,
              out_dir: str, smina_bin: str = "smina", exhaustiveness: int = 8,
              num_modes: int = 1) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    out_sdfs = []
    prog = st.progress(0.0)
    for i, lp in enumerate(lig_paths):
        base = os.path.splitext(os.path.basename(lp))[0]
        out = os.path.join(out_dir, f"{base}_docked.sdf")
        ok = run_smina_single(lp, receptor_pdb, center, size, out,
                              smina_bin=smina_bin, exhaustiveness=exhaustiveness, num_modes=num_modes)
        if ok:
            out_sdfs.append(out)
        prog.progress((i + 1) / max(1, len(lig_paths)))
    return out_sdfs


# =============== PLEC worker (subprocess with oddt) ===============
def plec_features_worker(file_list: List[str], receptor_pdb: str, plec_size: int = 4092,
                         max_per_file: int = 0
                         ) -> Tuple[np.ndarray, List[str], List[Optional[str]], List[str]]:
    code = r"""
import os, sys, numpy as np
import oddt
from oddt.fingerprints import PLEC

list_file = sys.argv[1]
receptor_pdb = sys.argv[2]
plec_size = int(sys.argv[3])
out_npy = sys.argv[4]
out_tsv = sys.argv[5]
max_per_file = int(sys.argv[6])

with open(list_file, 'r') as r:
    files = [line.strip() for line in r if line.strip()]

protein = next(oddt.toolkit.readfile('pdb', receptor_pdb))

vecs = []
rows = []  # name, smiles, source
for f in files:
    idx = 0
    for lig in oddt.toolkit.readfile('sdf', f):
        try:
            v = PLEC(lig, protein=protein, size=plec_size,
                     depth_protein=4, depth_ligand=2, distance_cutoff=4.5, sparse=False)
            vecs.append(np.asarray(v, dtype=np.float32))
            name = getattr(lig, 'title', '') or os.path.splitext(os.path.basename(f))[0]
            name = f"{name}#{idx}"
            smiles = getattr(lig, 'smiles', None) or ''
            rows.append((name, smiles, f, str(idx)))
            idx += 1
            if max_per_file > 0 and idx >= max_per_file:
                break
        except Exception:
            continue

arr = np.vstack(vecs) if len(vecs) else np.zeros((0, plec_size), dtype=np.float32)
np.save(out_npy, arr)
with open(out_tsv, 'w') as w:
    for name, smiles, src, idx in rows:
        w.write(f"{name}\t{smiles}\t{src}\t{idx}\n")
"""
    with tempfile.TemporaryDirectory() as td:
        list_path = os.path.join(td, "files.txt")
        npy_path = os.path.join(td, "plec.npy")
        tsv_path = os.path.join(td, "meta.tsv")
        with open(list_path, "w") as w:
            for f in file_list:
                w.write(f + "\n")
        env = os.environ.copy()
        cmd = [sys.executable, "-c", code, list_path, receptor_pdb, str(plec_size), npy_path, tsv_path, str(max_per_file)]
        p = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(f"[PLEC worker] failed:\nSTDERR:\n{p.stderr}\nSTDOUT:\n{p.stdout}")
        arr = np.load(npy_path)
        names, smiles, sources = [], [], []
        with open(tsv_path, "r") as r:
            for line in r:
                name, smi, src, idx = line.rstrip("\n").split("\t")
                names.append(name)
                smiles.append(smi if smi else None)
                sources.append(src)
    return arr, names, smiles, sources


# =============== Avalon from SMILES ===============
def avalon_from_smiles_list(smiles: List[Optional[str]], bits: int = 512) -> np.ndarray:
    X = np.zeros((len(smiles), bits), dtype=np.int32)
    for i, s in enumerate(smiles):
        if not s:
            continue
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        fp = pyAvalonTools.GetAvalonFP(m, nBits=bits)
        arr = np.zeros((bits,), dtype=np.int32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        X[i] = arr
    return X.astype(np.float32)


# =============== Predict helpers ===============
def predict_arrays(model: nn.Module, plec_arr: np.ndarray, avalon_arr: np.ndarray,
                   device: torch.device, batch_size: int = 128, threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    assert plec_arr.shape[0] == avalon_arr.shape[0], "Row mismatch between PLEC and Avalon arrays"
    model.to(device).eval()
    outs = []
    with torch.no_grad():
        for i in range(0, plec_arr.shape[0], batch_size):
            XA = torch.tensor(plec_arr[i:i+batch_size], dtype=torch.float32, device=device)
            XB = torch.tensor(avalon_arr[i:i+batch_size], dtype=torch.float32, device=device)
            p, *_ = model(XA, XB)
            outs.append(p.squeeze(-1).cpu().numpy())
    probs = np.concatenate(outs) if outs else np.empty((0,), dtype=np.float32)
    preds = (probs >= float(threshold)).astype(int)
    return probs, preds


# =============== UI (top) ===============
# --- Page config (set before any Streamlit output) ---
IMG_PATH = os.path.join(os.path.dirname(__file__), "figure-1.png")
try:
    _icon = Image.open(IMG_PATH)        # use your figure as page icon if present
except Exception:
    _icon = "🧠"                         # fallback emoji icon

st.set_page_config(page_title="AttentionScore", page_icon=_icon, layout="wide")

# --- Optional: a touch of CSS for a cleaner hero card ---
st.markdown("""
<style>
/* shrink default page padding a bit */
.block-container {padding-top: 1.2rem; padding-bottom: 1rem;}
/* nicer code font inside badges/lists */
code {font-size: 0.95rem;}
</style>
""", unsafe_allow_html=True)

# --- Hero section ---
col_l, col_r = st.columns([1, 1.5], vertical_alignment="center")

with col_l:
    if os.path.exists(IMG_PATH):
        st.image(IMG_PATH, use_container_width=True)
    else:
        st.markdown("**(Place `figure-1.png` next to this script to show the hero image)**")

with col_r:
    st.markdown("""
    <div style="padding:18px 22px;border-radius:16px;
                background:linear-gradient(135deg,#0f172a,#1e293b);
                color:#e2e8f0; border:1px solid #0b1220;">
      <h1 style="margin:0;font-size:2.4rem;line-height:1.1;">AttentionScore</h1>
      <p style="margin:10px 0 0;font-size:1.05rem;">
        An attention-driven scoring framework for structure-based virtual screening.
        It ingests docked complexes or raw ligands, assembles <b>PLEC-4092</b> and <b>Avalon-512</b> fingerprints,
        and predicts activity probabilities with a dual-stream attention network.
      </p>
      <ul style="margin:12px 0 0 18px;font-size:0.98rem;">
        <li>Three entry points: <b>Docked SDF</b>, <b>MOL2 → Dock</b>, <b>SMILES → MOL2 → Dock</b></li>
        <li>Backends: <code>ODDT</code> (PLEC), <code>RDKit</code> (Avalon), <code>smina</code>, <code>Open Babel</code></li>
        <li>Runs on CPU; uses GPU if available</li>
      </ul>
      <div style="margin-top:12px;font-size:1rem;opacity:0.9;">
        Developed by <b>Dr&nbsp;Muhammad&nbsp;Junaid</b>, Shenzhen University
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)  # spacer under the hero


with st.sidebar:
    st.header("Common")
    rec_file = st.file_uploader("Receptor PDB", type=["pdb"], accept_multiple_files=False)

    st.header("Checkpoint source")
    ckpt_mode = st.radio("Provide model checkpoint", ["Upload .pth", "Local path", "HTTP(S) URL"], index=0)
    ckpt_bytes: Optional[bytes] = None
    ckpt_info = ""

    if ckpt_mode == "Upload .pth":
        ckpt_file = st.file_uploader("Model checkpoint (.pth)", type=["pth"], accept_multiple_files=False)
        if ckpt_file is not None:
            ckpt_bytes = ckpt_file.getbuffer()
            ckpt_info = f"Uploaded: {ckpt_file.name}"

    elif ckpt_mode == "Local path":
        ckpt_path_text = st.text_input("Absolute path to .pth on this machine", value="")
        if ckpt_path_text:
            if os.path.isfile(ckpt_path_text):
                with open(ckpt_path_text, "rb") as r:
                    ckpt_bytes = r.read()
                ckpt_info = f"Loaded from path: {ckpt_path_text}"
            else:
                st.warning("Path not found.")

    elif ckpt_mode == "HTTP(S) URL":
        url = st.text_input("Direct URL to .pth (HTTPS)", value="")
        if url:
            try:
                with st.spinner("Downloading checkpoint…"):
                    r = requests.get(url, stream=True, timeout=120)
                    r.raise_for_status()
                    ckpt_bytes = r.content
                ckpt_info = f"Downloaded from: {url}"
            except Exception as e:
                st.error(f"Download failed: {e}")

    if ckpt_info:
        st.caption(ckpt_info)

    st.header("Model/Features")
    plec_size = st.number_input("PLEC size", min_value=512, max_value=8192, value=4092, step=256)
    avalon_bits = st.number_input("Avalon bits", min_value=128, max_value=4096, value=512, step=128)
    batch_size = st.slider("Batch size", 16, 1024, 128, step=16)
    threshold = st.slider("Classification threshold", 0.0, 1.0, 0.5, step=0.01)
    force_cpu = st.checkbox("Force CPU", value=True)

    st.header("Docking box")
    box_mode = st.radio("Define box by", ["Manual center/size", "From reference ligand"], index=1)
    padding = st.number_input("Padding (Å, for reference ligand box)", min_value=0.0, max_value=50.0, value=6.0, step=0.5)
    center = {"center_x": 0.0, "center_y": 0.0, "center_z": 0.0}
    size   = {"size_x": 20.0, "size_y": 20.0, "size_z": 20.0}
    ref_lig_file = None
    if box_mode == "Manual center/size":
        center["center_x"] = st.number_input("center_x", value=0.0, step=0.5, format="%.3f")
        center["center_y"] = st.number_input("center_y", value=0.0, step=0.5, format="%.3f")
        center["center_z"] = st.number_input("center_z", value=0.0, step=0.5, format="%.3f")
        size["size_x"] = st.number_input("size_x", value=20.0, step=0.5, format="%.3f")
        size["size_y"] = st.number_input("size_y", value=20.0, step=0.5, format="%.3f")
        size["size_z"] = st.number_input("size_z", value=20.0, step=0.5, format="%.3f")
    else:
        ref_lig_file = st.file_uploader("Reference ligand (SDF/MOL2/PDB)", type=["sdf", "mol2", "pdb"], accept_multiple_files=False)

    st.header("Docking (smina)")
    smina_bin = st.text_input("smina binary", value="smina")
    exhaustiveness = st.number_input("exhaustiveness", min_value=1, max_value=128, value=8, step=1)
    num_modes = st.number_input("num_modes", min_value=1, max_value=50, value=1, step=1)

    st.header("SMILES → MOL2 (Open Babel)")
    obabel_bin = st.text_input("obabel binary", value="obabel")
    obabel_steps = st.number_input("Open Babel minimize steps", min_value=0, max_value=5000, value=500, step=50)

tab_sdf, tab_mol2, tab_smiles = st.tabs(["Docked SDF", "MOL2 → Dock", "SMILES → Dock"])

with tab_sdf:
    sdf_files = st.file_uploader("Docked SDF file(s) or ZIP", type=["sdf", "zip"], accept_multiple_files=True)
    max_per_file = st.number_input("Max entries per SDF (0 = all)", min_value=0, max_value=10000, value=0, step=1)
    run_a = st.button("Run prediction (SDF route)", type="primary")

with tab_mol2:
    mol2_files = st.file_uploader("MOL2 file(s) or ZIP", type=["mol2", "zip"], accept_multiple_files=True)
    run_b = st.button("Dock + predict (MOL2 route)", type="primary")

with tab_smiles:
    st.write("Either upload a CSV OR paste SMILES (optionally name,smiles).")
    smiles_csv = st.file_uploader("CSV with columns [smiles] or [name,smiles]", type=["csv"], accept_multiple_files=False)
    smiles_text = st.text_area("SMILES (one per line, 'name,smiles' or just 'smiles')",
                               height=160,
                               placeholder="CCO\naspirin,CC(=O)OC1=CC=CC=C1C(=O)O")
    run_c = st.button("SMILES → MOL2 → Dock + predict", type="primary")

with st.expander("Notes", expanded=False):
    st.write(
        "- For **MOL2** and **SMILES** routes you must supply a docking box (manual center/size or from a reference ligand).\n"
        "- This app calls external binaries: **smina** for docking, **obabel** for SMILES→MOL2.\n"
        "- PLEC is computed via ODDT in a short subprocess for robustness.\n"
        "- Checkpoint can be uploaded, read from local path, or downloaded by URL (bypasses 200 MB upload cap)."
    )


# =============== Orchestration helpers ===============
def ensure_box(work_dir: str) -> Tuple[dict, dict]:
    if box_mode == "Manual center/size":
        return center, size
    if not ref_lig_file:
        raise RuntimeError("Please upload a reference ligand to compute the docking box.")
    ref_path = write_upload_to_path(ref_lig_file, os.path.join(work_dir, f"ref{os.path.splitext(ref_lig_file.name)[1].lower()}"))
    return compute_box_from_reference(ref_path, padding=float(padding))

def finish_and_download(df: pd.DataFrame, label: str = "predictions"):
    st.success(f"Done. {len(df)} rows.")
    st.dataframe(df, use_container_width=True)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv_bytes, file_name=f"{label}.csv", mime="text/csv")
    try:
        xlsx_buf = io.BytesIO()
        with pd.ExcelWriter(xlsx_buf, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name="predictions")
        st.download_button("Download Excel", data=xlsx_buf.getvalue(),
                           file_name=f"{label}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception as e:
        st.warning(f"Excel export not available: {e}")

def run_pipeline_from_sdfs(local_sdf_paths: List[str], rec_path: str, ckpt_bytes: bytes, device: torch.device,
                           label: str, max_per_file: int):
    st.info(f"Found {len(local_sdf_paths)} SDF file(s). Computing PLEC…")
    with st.spinner("Computing PLEC with ODDT…"):
        plec_arr, names, smiles, sources = plec_features_worker(
            local_sdf_paths, rec_path, plec_size=int(plec_size), max_per_file=int(max_per_file)
        )

    # Fill missing SMILES via RDKit if absent in SDF
    st.info("Completing SMILES (fallback via RDKit) if missing…")
    for i, (smi, src, name) in enumerate(zip(smiles, sources, names)):
        if smi:
            continue
        try:
            idx = int(name.rsplit("#", 1)[-1]) if "#" in name else 0
            suppl = Chem.SDMolSupplier(src, removeHs=False, sanitize=True)
            if 0 <= idx < len(suppl):
                m = suppl[idx]
                if m:
                    smiles[i] = Chem.MolToSmiles(m, isomericSmiles=True, canonical=True)
        except Exception:
            pass

    st.info("Computing Avalon fingerprints…")
    avalon_arr = avalon_from_smiles_list(smiles, bits=int(avalon_bits))

    # Load model & predict
    st.info("Loading model and predicting…")
    model = load_checkpoint(ckpt_bytes, device=device,
                            input_dim_A=int(plec_size), input_dim_B=int(avalon_bits),
                            n_heads=1, n_layers=1, event_num=1)

    probs, preds = predict_arrays(model, plec_arr, avalon_arr, device=device,
                                  batch_size=int(batch_size), threshold=float(threshold))

    df = pd.DataFrame({
        "molecule": names,
        "smiles": smiles,
        "source": sources,
        "probability": probs.round(4),
        "activity": preds.astype(int),
    })
    df["activity_label"] = df["activity"].map({1: "Active", 0: "Inactive"})
    finish_and_download(df, label=label)


# =============== Route A: Docked SDF(s) ===============
if run_a:
    if not rec_file:
        st.error("Please upload receptor PDB.")
    elif ckpt_bytes is None:
        st.error("Please provide a checkpoint (upload / local path / URL).")
    elif not sdf_files:
        st.error("Please provide at least one SDF (or ZIP).")
    else:
        device = torch.device("cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
        work_dir = tempfile.mkdtemp(prefix="deepcgaspred_")
        try:
            rec_path = write_upload_to_path(rec_file, os.path.join(work_dir, "receptor.pdb"))

            local_sdf_paths: List[str] = []
            tmp_unpack = os.path.join(work_dir, "unpacked_sdf")
            os.makedirs(tmp_unpack, exist_ok=True)
            for up in sdf_files:
                dst = write_upload_to_path(up, os.path.join(tmp_unpack, up.name))
                local_sdf_paths.extend(
                    unzip_if_needed(dst, os.path.join(tmp_unpack, "unzipped_" + os.path.splitext(up.name)[0]), ext=".sdf")
                )
            local_sdf_paths = sorted({p for p in local_sdf_paths if p.lower().endswith(".sdf")})
            if not local_sdf_paths:
                st.error("No .sdf files found after processing uploads.")
            else:
                run_pipeline_from_sdfs(local_sdf_paths, rec_path, ckpt_bytes, device,
                                       label="predictions_sdf", max_per_file=max_per_file)
        finally:
            try:
                shutil.rmtree(work_dir, ignore_errors=True)
            except Exception:
                pass


# =============== Route B: MOL2 -> Dock -> Predict ===============
if run_b:
    if not rec_file:
        st.error("Please upload receptor PDB.")
    elif ckpt_bytes is None:
        st.error("Please provide a checkpoint (upload / local path / URL).")
    elif not mol2_files:
        st.error("Please upload MOL2 file(s) or ZIP.")
    else:
        device = torch.device("cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
        work_dir = tempfile.mkdtemp(prefix="deepcgaspred_")
        try:
            rec_path = write_upload_to_path(rec_file, os.path.join(work_dir, "receptor.pdb"))

            # Box
            try:
                box_center, box_size = ensure_box(work_dir)
            except Exception as e:
                st.error(str(e))
                shutil.rmtree(work_dir, ignore_errors=True)
                st.stop()

            # Gather MOL2s (single or ZIP)
            local_mol2_paths: List[str] = []
            tmp_unpack = os.path.join(work_dir, "unpacked_mol2")
            os.makedirs(tmp_unpack, exist_ok=True)
            for up in mol2_files:
                dst = write_upload_to_path(up, os.path.join(tmp_unpack, up.name))
                if dst.lower().endswith(".mol2"):
                    local_mol2_paths.append(dst)
                else:
                    local_mol2_paths.extend(
                        unzip_if_needed(dst, os.path.join(tmp_unpack, "unzipped_" + os.path.splitext(up.name)[0]), ext=".mol2")
                    )
            local_mol2_paths = sorted({p for p in local_mol2_paths if p.lower().endswith(".mol2")})
            if not local_mol2_paths:
                st.error("No .mol2 files found after processing uploads.")
                st.stop()

            # Dock
            st.info(f"Docking {len(local_mol2_paths)} ligand(s) with smina…")
            out_dir = os.path.join(work_dir, "docked")
            sdfs = dock_many(local_mol2_paths, rec_path, box_center, box_size, out_dir,
                             smina_bin=smina_bin, exhaustiveness=int(exhaustiveness), num_modes=int(num_modes))
            if not sdfs:
                st.error("Docking produced no SDFs. Check smina/box settings.")
                st.stop()

            run_pipeline_from_sdfs(sdfs, rec_path, ckpt_bytes, device,
                                   label="predictions_mol2", max_per_file=0)
        finally:
            try:
                shutil.rmtree(work_dir, ignore_errors=True)
            except Exception:
                pass


# =============== Route C: SMILES -> MOL2 -> Dock -> Predict ===============
if run_c:
    if not rec_file:
        st.error("Please upload receptor PDB.")
    elif ckpt_bytes is None:
        st.error("Please provide a checkpoint (upload / local path / URL).")
    elif not (smiles_csv or (smiles_text and smiles_text.strip())):
        st.error("Please upload a CSV or paste SMILES.")
    else:
        device = torch.device("cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
        work_dir = tempfile.mkdtemp(prefix="deepcgaspred_")
        try:
            rec_path = write_upload_to_path(rec_file, os.path.join(work_dir, "receptor.pdb"))

            # Box
            try:
                box_center, box_size = ensure_box(work_dir)
            except Exception as e:
                st.error(str(e))
                shutil.rmtree(work_dir, ignore_errors=True)
                st.stop()

            # Parse SMILES -> (name, smi) pairs
            pairs: List[Tuple[str, str]] = []
            if smiles_csv is not None:
                df_csv = pd.read_csv(smiles_csv)
                if "smiles" in df_csv.columns:
                    if "name" in df_csv.columns:
                        pairs.extend([(str(n), str(s)) for n, s in zip(df_csv["name"], df_csv["smiles"])])
                    else:
                        pairs.extend([(f"mol_{i}", str(s)) for i, s in enumerate(df_csv["smiles"])])
                elif set(df_csv.columns[:2]) >= {"name", "smiles"}:
                    pairs.extend([(str(df_csv.iloc[i, 0]), str(df_csv.iloc[i, 1])) for i in range(len(df_csv))])
                else:
                    st.error("CSV must include 'smiles' column (and optional 'name').")
                    st.stop()

            if smiles_text and smiles_text.strip():
                for line in smiles_text.strip().splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    if "," in line:
                        nm, smi = line.split(",", 1)
                        pairs.append((nm.strip(), smi.strip()))
                    else:
                        pairs.append((f"mol_{len(pairs)}", line))

            if not pairs:
                st.error("No valid SMILES found.")
                st.stop()

            # SMILES -> MOL2
            st.info(f"Converting {len(pairs)} SMILES to MOL2 via Open Babel…")
            mol2_dir = os.path.join(work_dir, "mol2")
            mol2_paths = smiles_to_mol2_files(pairs, mol2_dir, obabel_bin=obabel_bin, steps=int(obabel_steps))
            if not mol2_paths:
                st.error("Open Babel conversion produced no MOL2 files. Check 'obabel' binary.")
                st.stop()

            # Dock
            st.info(f"Docking {len(mol2_paths)} ligand(s) with smina…")
            out_dir = os.path.join(work_dir, "docked")
            sdfs = dock_many(mol2_paths, rec_path, box_center, box_size, out_dir,
                             smina_bin=smina_bin, exhaustiveness=int(exhaustiveness), num_modes=int(num_modes))
            if not sdfs:
                st.error("Docking produced no SDFs. Check smina/box settings.")
                st.stop()

            run_pipeline_from_sdfs(sdfs, rec_path, ckpt_bytes, device,
                                   label="predictions_smiles", max_per_file=0)
        finally:
            try:
                shutil.rmtree(work_dir, ignore_errors=True)
            except Exception:
                pass

