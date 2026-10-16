#!/usr/bin/env bash
# Download the third-party files the data pipeline and the live app read, and check them against the checksums of the
# copies used to build the released data:
#   data/geo/lid.176.ftz   fastText language identification (Joulin et al. 2016; CC BY-SA 3.0), used to check that a
#                          search or a tl;dr is in the requested language (synth/realistic.py language_ok)
# The gazetteer, lexicon and common-word lists are committed in src/relevance_explainer/resources.
#   scripts/fetch_resources.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/geo
fetch() {  # url, path, sha256
  if [ ! -f "$2" ]; then
    curl -fsSL --retry 3 "$1" -o "$2.part" && mv "$2.part" "$2"
  fi
  echo "$3  $2" | sha256sum -c -
}
fetch https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz data/geo/lid.176.ftz \
  8f3472cfe8738a7b6099e8e999c3cbfae0dcd15696aac7d7738a8039db603e83
