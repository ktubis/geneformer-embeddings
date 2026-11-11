#!/usr/bin/env bash
# submit_offset_intervention.sh
# Run from the LOGIN NODE, in this checkout:
#   bash bash_scripts/submit_offset_intervention.sh --out DIR [--tasks N] [--max-parallel K]
#       [--time-shifts HH:MM:SS] [--time-eval HH:MM:SS] [--mem MEM]
#       [--single [--time HH:MM:SS]] [--exclude NODES] [--dry-run]
#       -- <scripts/offset_intervention.py options>
#
# Three chained jobs (scripts/offset_intervention.py --stage), as in scgpt_embeds'
# submitter: shifts (one GPU job), eval (a GPU job array, one task per instance and
# --chunk-size cells, listed in tasks.tsv), merge (CPU). The array is submitted as
# 0..N-1 (--tasks, default 400) before tasks.tsv exists; tasks past the last exit at
# once; if tasks.tsv lists more than N, the shifts job's log says so (submit the rest by
# hand before the merge). --max-parallel K caps the tasks running at once (default 20).
# --single runs everything in one job (--stage all).
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
# Full runs (every cell, every layer, 5 random shifts):
#   colon, every site pooled per chemistry, the six celltype_coarse types:
#   bash bash_scripts/submit_offset_intervention.sh --out offset_runs/colon -- \
#       --batch "3p=cc_10x3p_*colon,cc_10x3p_rectum" \
#       --batch "5p=cc_10x5p_*colon,cc_10x5p_rectum" --batch-pairs 3p:5p \
#       --celltypes "T cell" "B cell" "Plasma cell" "Myeloid cell" "Epithelial cell" "Stromal cell" \
#       --celltype-pairs "T cell:B cell" "Epithelial cell:Stromal cell" \
#           "Plasma cell:Stromal cell" "T cell:Epithelial cell" \
#       --layers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 --n-random 5
#   pancreas, baron vs muraro (muraro calls ductal cells "duct"):
#   bash bash_scripts/submit_offset_intervention.sh --out offset_runs/pancreas -- \
#       --batch-pairs baron:muraro --celltype-col celltype_mid \
#       --celltypes alpha beta "ductal=ductal,duct" acinar \
#       --celltype-pairs alpha:beta beta:acinar \
#       --layers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 --n-random 5
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"       # this checkout, bound to /mnt
MODEL_SUBDIR="Geneformer-V2-316M"
HOST_DATA="/cluster/work/boeva/ktubis/data"                    # bound to /data
OUT=""; TIME="08:00:00"; TIME_SHIFTS="12:00:00"; TIME_EVAL="08:00:00"; MEM="128G"
TASKS=400; PAR=20; EXCLUDE="gpu-biomed-03,gpu-biomed-07,gpu-biomed-08"; DRY=false; SINGLE=false
PASS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --out)          OUT="$2"; shift 2 ;;
        --time)         TIME="$2"; shift 2 ;;
        --time-shifts)  TIME_SHIFTS="$2"; shift 2 ;;
        --time-eval)    TIME_EVAL="$2"; shift 2 ;;
        --mem)          MEM="$2"; shift 2 ;;
        --tasks)        TASKS="$2"; shift 2 ;;
        --max-parallel) PAR="$2"; shift 2 ;;
        --single)       SINGLE=true; shift ;;
        --exclude)      EXCLUDE="$2"; shift 2 ;;
        --dry-run)      DRY=true; shift ;;
        --)             shift; PASS=("$@"); break ;;
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
RUN="${SIF_RUN} python /mnt/scripts/offset_intervention.py \
    --data-dir /data/final_datasets --model-dir /mnt/${MODEL_SUBDIR} --nproc 4 \
    --out ${OUT} ${ARGS}"
GPU="--partition=gpu --gres=gpu:1 ${EXCLUDE:+--exclude=$EXCLUDE} --mem=$MEM --cpus-per-task=4"
LOG="${REPO}/logs/${NAME}"
HEAD="echo 'geneformer code: ${CODE_COMMIT}';"

mkdir -p "${REPO}/logs"
if $SINGLE; then
    $DRY && { echo "${RUN} --stage all"; exit 0; }
    sbatch $GPU --time="$TIME" --job-name="${NAME:0:64}" --output="${LOG}_%j.out" \
        --wrap="${HEAD} ${RUN} --stage all"
    exit $?
fi
if $DRY; then
    echo "shifts: ${RUN} --stage shifts"
    echo "eval:   ${RUN} --stage eval   (array task = \$SLURM_ARRAY_TASK_ID)"
    echo "merge:  ${SIF_RUN} python /mnt/scripts/offset_intervention.py --out ${OUT} --stage merge"
    exit 0
fi
J1=$(sbatch --parsable $GPU --time="$TIME_SHIFTS" --job-name="${NAME:0:56}_shifts" \
    --output="${LOG}_shifts_%j.out" \
    --wrap="${HEAD} ${RUN} --stage shifts; n=\$(( \$(wc -l < ${REPO}/${OUT}/tasks.tsv) - 1 )); \
            [ \$n -le ${TASKS} ] || echo \"WARNING: \$n tasks > --tasks ${TASKS}: submit the rest\"") \
    || { echo "sbatch (shifts) failed" >&2; exit 1; }
J2=$(sbatch --parsable $GPU --time="$TIME_EVAL" --array="0-$((TASKS - 1))%${PAR}" \
    --dependency=afterok:${J1} --job-name="${NAME:0:58}_eval" --output="${LOG}_eval_%A_%a.out" \
    --wrap="${HEAD} ${RUN} --stage eval") \
    || { echo "sbatch (eval) failed" >&2; exit 1; }
J3=$(sbatch --parsable --cpus-per-task=2 --mem=32G --time=02:00:00 --dependency=afterok:${J2} \
    --job-name="${NAME:0:57}_merge" --output="${LOG}_merge_%j.out" \
    --wrap="${SIF_RUN} python /mnt/scripts/offset_intervention.py --out ${OUT} --stage merge") \
    || { echo "sbatch (merge) failed" >&2; exit 1; }
echo "shifts ${J1}, eval ${J2} (tasks 0-$((TASKS - 1)), ${PAR} at a time), merge ${J3}"
