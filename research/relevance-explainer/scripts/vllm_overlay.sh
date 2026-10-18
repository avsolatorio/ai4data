#!/usr/bin/env bash
# Build an environment that reuses the vllm environment's packages with another transformers release, for a model
# that the locked transformers cannot load (Mistral Small 4: vllm 0.30.0 imports PixtralRotaryEmbedding, which
# transformers 5.18.0 renamed; 5.16.1 still has it). The vllm environment itself is left as locked.
#   scripts/vllm_overlay.sh <dir> [transformers version, default 5.16.1]
# then set RELEXP_VLLM_OVERLAY=<dir>/bin/vllm in scripts/local.env (read for models with "overlay": true in synth/models.py).
set -euo pipefail
cd "$(dirname "$0")/.."
. scripts/env.sh
DIR=${1:?directory for the overlay environment}
TF=${2:-5.16.1}
BASE=$(dirname "$RELEXP_VLLM")
SITE=$("$BASE/python" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
"$BASE/python" -m venv --without-pip "$DIR"
OWN=$("$DIR/bin/python" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
echo "$SITE" > "$OWN/zz_vllm_env.pth"  # after the overlay's own packages, so its transformers is found first
"$BASE/pip" install --no-deps --target "$OWN" "transformers==$TF"
cat > "$DIR/bin/vllm" <<EOF
#!$DIR/bin/python
import sys
from vllm.entrypoints.cli.main import main
if __name__ == "__main__":  # vllm's workers are spawned and import this file again
    sys.exit(main())
EOF
chmod +x "$DIR/bin/vllm"
"$DIR/bin/python" -c "import transformers, vllm; print('transformers', transformers.__version__, 'vllm', vllm.__version__)"
