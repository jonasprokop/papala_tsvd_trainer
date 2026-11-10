#!/bin/bash
# Legacy alias: keep a single canonical pipeline implementation.
set -euo pipefail
exec "$(dirname "$0")/run_all.sh" "$@"
