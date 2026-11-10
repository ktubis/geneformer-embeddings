#!/usr/bin/env bash
# submit_offset_intervention.sh
# Run from the LOGIN NODE, in this checkout:
#   bash bash_scripts/submit_offset_intervention.sh --out DIR [--time HH:MM:SS] [--mem MEM]
#       [--exclude NODES] [--dry-run] -- <scripts/offset_intervention.py options>
#
# One GPU job: activation addition of the offset between instances (see the
# docstring of scripts/offset_intervention.py), with the pretrained model
# (MODEL_SUBDIR/) and Geneformer's V2 tokenizer, on final_datasets, as
# bash_scripts/submit_geneformer_gene_embeds.sh makes the gene embeddings. The
# counterpart of scgpt_embeds' bash_scripts/submit_offset_intervention.sh.
# Everything after -- goes to the script unchanged (quoted for the job). --out is
# relative to this checkout and must not exist yet. --exclude keeps the job off these
# nodes (default the RTX 5090 nodes gpu-biomed-03,07,08, which the containers' CUDA
# builds have no kernels for); --exclude '' allows every node.
#
# The checkout must be clean (its commit is written at the top of the job log), with
# the model and the token dictionaries copied in (gitignored), as for the gene
# embeddings. Batches are file groups (--batch NAME=PATTERN[,PATTERN...], pooled:
# site is not a factor); the healthy colon sites are the *colon files and rectum.
#
# Pilot (layers subsampled to measure run time and effect sizes first), two jobs:
#   colon, every site pooled per chemistry, the six celltype_coarse types:
#   bash bash_scripts/submit_offset_intervention.sh --out offset_runs/pilot_colon -- \
#       --batch "3p=cc_10x3p_*colon,cc_10x3p_rectum" \
#       --batch "5p=cc_10x5p_*colon,cc_10x5p_rectum" --batch-pairs 3p:5p \
#       --celltypes "T cell" "B cell" "Plasma cell" "Myeloid cell" "Epithelial cell" "Stromal cell" \
#       --celltype-pairs "T cell:B cell" "Epithelial cell:Stromal cell" \
#           "Plasma cell:Stromal cell" "T cell:Epithelial cell" \
#       --layers 0 4 9 13 17 18
#   pancreas, baron vs a UMI (muraro) and a full-length (segerstolpe) dataset:
#   bash bash_scripts/submit_offset_intervention.sh --out offset_runs/pilot_pancreas -- \
#       --batch-pairs baron:muraro baron:segerstolpe --celltype-col celltype_mid \
#       --celltypes alpha beta delta PP ductal acinar PSC \
#       --celltype-pairs alpha:beta beta:acinar --layers 0 4 9 13 17 18
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"       # this checkout, bound to /mnt
MODEL_SUBDIR="Geneformer-V2-316M"
HOST_DATA="/cluster/work/boeva/ktubis/data"                    # bound to /data
OUT=""; TIME="08:00:00"; MEM="128G"; EXCLUDE="gpu-biomed-03,gpu-biomed-07,gpu-biomed-08"
DRY=false; PASS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --out)     OUT="$2"; shift 2 ;;
        --time)    TIME="$2"; shift 2 ;;
        --mem)     MEM="$2"; shift 2 ;;
        --exclude) EXCLUDE="$2"; shift 2 ;;
        --dry-run) DRY=true; shift ;;
        --)        shift; PASS=("$@"); break ;;
        *) echo "error: unrecognized arg '$1' (script options go after --)" >&2; exit 1 ;;
    esac
done
[[ -n "$OUT" ]] || { echo "error: --out is required" >&2; exit 1; }
[[ ! -e "${REPO}/${OUT}" ]] || { echo "error: ${REPO}/${OUT} exists; give a new --out" >&2; exit 1; }
[[ ${#PASS[@]} -gt 0 ]] || { echo "error: no script options after --" >&2; exit 1; }

CODE_COMMIT=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null) \
    || { echo "error: ${REPO} is not a git checkout" >&2; exit 1; }
if [[ -n "$(git -C "$REPO" status --porcelain -- scripts extract_embeddings.py geneformer)" ]]; then
    echo "error: uncommitted changes in ${REPO}:" >&2
    git -C "$REPO" status --short -- scripts extract_embeddings.py geneformer >&2
    exit 1
fi
MISSING=()
p="${MODEL_SUBDIR}/model.safetensors"
[[ -f "${REPO}/${p}" && $(wc -c < "${REPO}/${p}") -gt 1000000 ]] || MISSING+=("${p}")
[[ -f "${REPO}/${MODEL_SUBDIR}/config.json" ]] || MISSING+=("${MODEL_SUBDIR}/config.json")
for f in token_dictionary gene_median_dictionary ensembl_mapping_dict gene_name_id_dict; do
    p="geneformer/${f}_gc104M.pkl"
    [[ -f "${REPO}/${p}" && "$(head -c1 "${REPO}/${p}" | od -An -tx1 | tr -d ' ')" == "80" ]] \
        || MISSING+=("${p}")
done
if [[ ${#MISSING[@]} -gt 0 ]]; then
    echo "error: missing in ${REPO}, or git-lfs pointers instead of the files (gitignored;" >&2
    echo "copy them in once from a clone of huggingface.co/ctheodoris/Geneformer):" >&2
    printf '  %s\n' "${MISSING[@]}" >&2
    exit 1
fi

SIF="/cluster/work/boeva/ktubis/singularity/geneformer_amd64.sif"
ENV_FILE="/cluster/work/boeva/ktubis/singularity_env"
SIF_RUN="singularity exec --nv \
    --bind ${REPO}:/mnt \
    --bind ${HOST_DATA}:/data \
    --env-file ${ENV_FILE} \
    --pwd /mnt \
    ${SIF}"

ARGS=$(printf '%q ' "${PASS[@]}")
NAME="gf_offset_$(basename "$OUT")"
CMD="echo 'geneformer code: ${CODE_COMMIT}'; ${SIF_RUN} python /mnt/scripts/offset_intervention.py \
    --data-dir /data/final_datasets --model-dir /mnt/${MODEL_SUBDIR} --nproc 4 \
    --out ${OUT} ${ARGS}"

mkdir -p "${REPO}/logs"
if $DRY; then
    echo "$CMD"
    exit 0
fi
sbatch --partition=gpu --gres=gpu:1 ${EXCLUDE:+--exclude="$EXCLUDE"} --mem="$MEM" --cpus-per-task=4 \
    --time="$TIME" --job-name="${NAME:0:64}" --output="${REPO}/logs/${NAME}_%j.out" \
    --wrap="$CMD"
