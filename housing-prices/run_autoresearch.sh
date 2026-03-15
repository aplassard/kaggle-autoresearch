#!/bin/bash

ITERATIONS=${1:-100}
WORKDIR="/Users/andrewplassard/git/kaggle-autoresearch/housing-prices"
PROMPT_FILE="$WORKDIR/AGENT_PROMPT.md"
MODEL="zai-coding-plan/glm-5"

echo "Starting autoresearch loop with $ITERATIONS iterations"
echo "Model: $MODEL"
echo "Prompt file: $PROMPT_FILE"
echo "Working directory: $WORKDIR"
echo "----------------------------------------"

for i in $(seq 1 $ITERATIONS); do
    git checkout main && git pull
    echo ""
    echo "========================================"
    echo "Iteration $i of $ITERATIONS"
    echo "Started at: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    echo "========================================"
    
    cd "$WORKDIR"
    
    PROMPT_CONTENT=$(cat "$PROMPT_FILE")
    
    opencode run \
        -m "$MODEL" \
        "Read @AGENT_PROMPT.md and complete the work fully independently" \
        2>&1
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ Iteration $i completed successfully"
    else
        echo "✗ Iteration $i failed with exit code $EXIT_CODE"
    fi
    
    echo "Finished at: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    
    if [ $i -lt $ITERATIONS ]; then
        echo "Waiting 2 seconds before next iteration..."
        sleep 2
    fi
done

echo ""
echo "========================================"
echo "Autoresearch loop completed"
echo "Total iterations: $ITERATIONS"
echo "Finished at: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "========================================"
