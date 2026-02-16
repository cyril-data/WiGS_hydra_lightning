#!/bin/bash
echo "Submitting all generated master job arrays..."
for f in RunSimulations/master_job_*.sbatch; do
    if [ -f "$f" ]; then
        echo "Submitting: $f"
        sbatch "$f"
    fi
done
echo "All jobs submitted."

