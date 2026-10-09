#!/usr/bin/env bash
# submit_geneformer_gene_embeds.sh
# Run from the LOGIN NODE, in this checkout:
#   bash bash_scripts/submit_geneformer_gene_embeds.sh [--skip-existing] [--cols col1,col2]
#       [--files name1,name2] [--out DIR]
#
# Submits one SLURM job per (dataset, celltype_col, celltype) in DATA_DIR; --files
# limits it to those datasets (file names without .h5ad), e.g. for a test run:
#   bash bash_scripts/submit_geneformer_gene_embeds.sh --cols celltype_coarse \
#       --files cc_10x3p_descending_colon,cc_10x5p_ascending_colon --out gene_embeddings_test
# Output, in scGPT's format and layout (scgpt_embeds' submit_gene_embeds.sh), in this checkout:
#   <out>/<dataset>/<celltype_col>/<celltype>.npz   key layer_gene_embs,
#       (19, n_genes, 1152) float32: input embedding + 18 layers; n_cells: cells per gene
#   <out>/<dataset>/<celltype_col>/<celltype>_gene_names.txt   gene symbols
# Each gene is averaged over the cells of that celltype whose input contains it;
# genes expressed in fewer than min(10, n_cells) of them are left out. No cell embeddings.
#
# The jobs run this checkout's extract_embeddings.py and geneformer package inside
# geneformer_amd64.sif (the environment), so a code change needs only a pull. The
# checkout must be clean: its commit is written at the top of every job log. The
# weights (model.safetensors, config.json) and token dictionaries (geneformer/*.pkl)
# are gitignored and must be copied into it once.

set -uo pipefail

# ── Paths (host-side) ──────────────────────────────────────────────────────────
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"       # this checkout, bound to /mnt
HOST_DATA="/cluster/work/boeva/ktubis/data"                    # bound to /data
DATA_DIR="${HOST_DATA}/final_datasets"
OUT_BASE="gene_embeddings"
CELLTYPE_COLS=("celltype_coarse" "celltype_mid" "celltype_fine")
FILES=()

SKIP_EXISTING=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-existing) SKIP_EXISTING=true; shift ;;
        --cols|--files|--out)
            [[ $# -ge 2 ]] || { echo "error: $1 needs a value" >&2; exit 1; }
            case "$1" in
                --cols)  IFS=',' read -ra CELLTYPE_COLS <<< "$2" ;;
                --files) IFS=',' read -ra FILES <<< "$2" ;;
                --out)   OUT_BASE="$2" ;;
            esac
            shift 2 ;;
        *) echo "error: unrecognized arg '$1'" >&2
           echo "usage: $0 [--skip-existing] [--cols col1,col2] [--files name1,name2] [--out DIR]" >&2
           exit 1 ;;
    esac
done

# ── Code, weights, dictionaries ────────────────────────────────────────────────
CODE_COMMIT=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null) \
    || { echo "error: ${REPO} is not a git checkout" >&2; exit 1; }
if [[ -n "$(git -C "$REPO" status --porcelain -- extract_embeddings.py geneformer)" ]]; then
    echo "error: uncommitted changes in ${REPO}:" >&2
    git -C "$REPO" status --short -- extract_embeddings.py geneformer >&2
    exit 1
fi
MISSING=()
for f in model.safetensors config.json; do
    [[ -f "${REPO}/${f}" ]] || MISSING+=("${f}")
done
for f in token_dictionary gene_median_dictionary ensembl_mapping_dict gene_name_id_dict; do
    p="geneformer/${f}_gc104M.pkl"
    # a real pickle starts with byte 0x80; a git-lfs pointer is a short text file
    [[ -f "${REPO}/${p}" && "$(head -c1 "${REPO}/${p}" | od -An -tx1 | tr -d ' ')" == "80" ]] \
        || MISSING+=("${p}")
done
if [[ ${#MISSING[@]} -gt 0 ]]; then
    echo "error: missing in ${REPO} (gitignored, copy them in once, e.g. from a" >&2
    echo "Geneformer V2-316M clone of huggingface.co/ctheodoris/Geneformer):" >&2
    printf '  %s\n' "${MISSING[@]}" >&2
    exit 1
fi
echo "geneformer code: ${CODE_COMMIT} ($(git -C "$REPO" log -1 --format=%s))"

# ── Singularity ────────────────────────────────────────────────────────────────
SIF="/cluster/work/boeva/ktubis/singularity/geneformer_amd64.sif"  # container/build_amd64.sh
ENV_FILE="/cluster/work/boeva/ktubis/singularity_env"
# CPU-only: obs discovery on the login node (no GPU, no env-file, avoids torch/CUDA)
SIF_CPU="singularity exec \
    --bind ${REPO}:/mnt \
    --bind ${HOST_DATA}:/data \
    --pwd /mnt \
    ${SIF}"
# GPU: used inside each submitted job
SIF_RUN="singularity exec --nv \
    --bind ${REPO}:/mnt \
    --bind ${HOST_DATA}:/data \
    --env-file ${ENV_FILE} \
    --pwd /mnt \
    ${SIF}"

# ── SLURM settings ─────────────────────────────────────────────────────────────
PARTITION="gpu"
GRES="gpu:1"
MEM="32G"
TIME="02:00:00"
CPUS=4

mkdir -p "${REPO}/logs"

if [[ ${#FILES[@]} -gt 0 ]]; then
    DATA_PATHS=()
    for name in "${FILES[@]}"; do
        [[ -f "${DATA_DIR}/${name}.h5ad" ]] || { echo "error: no ${DATA_DIR}/${name}.h5ad" >&2; exit 1; }
        DATA_PATHS+=("${DATA_DIR}/${name}.h5ad")
    done
else
    DATA_PATHS=("$DATA_DIR"/*.h5ad)
fi

# ── Submit ─────────────────────────────────────────────────────────────────────
for DATA_PATH in "${DATA_PATHS[@]}"; do
    [[ -f "$DATA_PATH" ]] || continue
    DATASET=$(basename "$DATA_PATH" .h5ad)
    DATA_PATH_CONTAINER="/data/final_datasets/$(basename "$DATA_PATH")"

    for COL in "${CELLTYPE_COLS[@]}"; do
        # Discover celltypes via h5py — no torch/CUDA needed on the login node
        CELLTYPES=$(
            DATA_PATH_C="$DATA_PATH_CONTAINER" COL="$COL" \
            $SIF_CPU python3 - <<'PYEOF'
import h5py, os, sys

path = os.environ["DATA_PATH_C"]
col  = os.environ["COL"]

with h5py.File(path, "r") as f:
    obs = f.get("obs", {})
    if col not in obs:
        sys.exit(0)
    entry = obs[col]
    if isinstance(entry, h5py.Group) and "categories" in entry:
        raw = entry["categories"][:]
    else:
        raw = entry[:]
    vals = sorted({v.decode() if isinstance(v, bytes) else str(v) for v in raw})
    print("\n".join(vals))
PYEOF
        ) || true

        [[ -z "$CELLTYPES" ]] && continue
        echo "[$DATASET / $COL] submitting $(echo "$CELLTYPES" | wc -l | tr -d ' ') jobs"

        while IFS= read -r CELLTYPE; do
            [[ -z "$CELLTYPE" ]] && continue
            SAVE_NAME=$(echo "$CELLTYPE" | tr ' ' '_' | tr '/' '-')
            # Paths relative to /mnt (this checkout) inside the container
            OUTPUT_DIR_CONTAINER="${OUT_BASE}/${DATASET}/${COL}"
            SAVE_PATH_CONTAINER="${OUTPUT_DIR_CONTAINER}/${SAVE_NAME}"
            JOB_NAME="gf_${DATASET}_${COL}_${SAVE_NAME}"

            if $SKIP_EXISTING && [[ -f "${REPO}/${SAVE_PATH_CONTAINER}.npz" ]]; then
                echo "  skip (exists): ${DATASET}/${COL}/${SAVE_NAME}"
                continue
            fi

            sbatch \
                --partition="$PARTITION" \
                --gres="$GRES" \
                --mem="$MEM" \
                --time="$TIME" \
                --cpus-per-task="$CPUS" \
                --job-name="${JOB_NAME:0:64}" \
                --output="${REPO}/logs/${JOB_NAME}_%j.out" \
                --wrap="echo 'geneformer code: ${CODE_COMMIT}'; ${SIF_RUN} python /mnt/extract_embeddings.py \
                    --input '${DATA_PATH_CONTAINER}' \
                    --output_dir '${OUTPUT_DIR_CONTAINER}' \
                    --model_dir /mnt \
                    --cell_type '${CELLTYPE}' \
                    --cell_type_col '${COL}' \
                    --no_cell_embs \
                    --all_layers \
                    --save_path '${SAVE_PATH_CONTAINER}' \
                    --var_index var_names \
                    --nproc ${CPUS}"
        done <<< "$CELLTYPES"
    done
done
