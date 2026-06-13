#!/bin/bash
# Quick zip any directory
# Usage: bash zip_dir.sh <directory_path> [output_name]
# Examples:
#   bash zip_dir.sh results/optuna
#   bash zip_dir.sh results/optuna my_results

DIR="$1"
if [ -z "$DIR" ]; then
    echo "Usage: bash zip_dir.sh <directory_path> [output_name]"
    exit 1
fi

# Remove trailing slash
DIR="${DIR%/}"

# Output name: use argument or default to directory basename + timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTNAME="${2:-$(basename "$DIR")_$TIMESTAMP}"

tar -czf "${OUTNAME}.tar.gz" "$DIR" && echo "✅ Created ${OUTNAME}.tar.gz ($(du -h "${OUTNAME}.tar.gz" | cut -f1))"
