#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Edit the variables below, then:  bash run.sh
# ---------------------------------------------------------------------------

REAL_DIR="/data/real"
FAKE_DIRS="/data/gen_A /data/gen_B"   # space-separated, one per generator

ARCH="cnn"
ISIZE=128
NDIM=3
IN_CHANNELS=1

EPOCHS=20
BATCH_SIZE=4
PATIENCE=10
REAL_TRAIN_RATIO=0.6
FAKE_TRAIN_RATIO=0.6

# Convergence speed — increase GP_LAMBDA / decrease LR to slow down and
# keep GradNorm near 1.0 for longer.  Leave empty to use arch defaults.
#   GN keeps climbing → raise GP_LAMBDA (try 20, 30) or lower LR (try 1e-5)
GP_LAMBDA=""     # e.g. GP_LAMBDA="20"
LR=""            # e.g. LR="1e-5"

OUT_DIR="./runs/${ARCH}_isize${ISIZE}"
NUM_WORKERS=4
SEED=42

# Optional architecture overrides — leave empty to use auto-scaling
NDF=""           # e.g. NDF="32"
PATCH_SIZE=""
D_MODEL=""
N_HEADS=""
N_LAYERS=""

# ---------------------------------------------------------------------------
# Build optional flags only when a value is set
# ---------------------------------------------------------------------------

EXTRA=""
[ -n "$GP_LAMBDA" ]  && EXTRA="$EXTRA --gp_lambda $GP_LAMBDA"
[ -n "$LR" ]         && EXTRA="$EXTRA --lr $LR"
[ -n "$NDF" ]        && EXTRA="$EXTRA --ndf $NDF"
[ -n "$PATCH_SIZE" ] && EXTRA="$EXTRA --patch_size $PATCH_SIZE"
[ -n "$D_MODEL" ]    && EXTRA="$EXTRA --d_model $D_MODEL"
[ -n "$N_HEADS" ]    && EXTRA="$EXTRA --n_heads $N_HEADS"
[ -n "$N_LAYERS" ]   && EXTRA="$EXTRA --n_layers $N_LAYERS"

# ---------------------------------------------------------------------------

python train.py \
    --real_dir       "$REAL_DIR" \
    --fake_dirs      $FAKE_DIRS \
    --arch           "$ARCH" \
    --isize          "$ISIZE" \
    --ndim           "$NDIM" \
    --in_channels    "$IN_CHANNELS" \
    --epochs         "$EPOCHS" \
    --batch_size     "$BATCH_SIZE" \
    --patience       "$PATIENCE" \
    --real_train_ratio "$REAL_TRAIN_RATIO" \
    --fake_train_ratio "$FAKE_TRAIN_RATIO" \
    --out_dir        "$OUT_DIR" \
    --num_workers    "$NUM_WORKERS" \
    --seed           "$SEED" \
    $EXTRA
