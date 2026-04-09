#!/bin/bash
# Check if training is complete, resubmit if not
# Usage: bash scripts/resubmit_if_needed.sh configs/s2_decay.yaml s2_decay
#        bash scripts/resubmit_if_needed.sh configs/s2_decay_optim.yaml s2_dopt

CONFIG=$1
JOBNAME=$2

OUTPUT_DIR=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['output_dir'])")

if [ -f "$OUTPUT_DIR/checkpoint_final.pt" ]; then
    echo "$JOBNAME: DONE (checkpoint_final.pt exists)"
else
    LATEST=$(ls -t ${OUTPUT_DIR}/checkpoint_*.pt 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then
        STEP=$(echo "$LATEST" | grep -o '[0-9]*' | tail -1)
        TOTAL=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(int(c['total_tokens']/c['seq_len']))")
        echo "$JOBNAME: step $STEP / $TOTAL — resubmitting"
    else
        echo "$JOBNAME: no checkpoints — resubmitting from scratch"
    fi
    cd /scratch/gpfs/HENDERSON/zs7353/ssm_ttt
    sbatch --export=CONFIG=$CONFIG --job-name=$JOBNAME scripts/run_phase1_all.sh
fi
