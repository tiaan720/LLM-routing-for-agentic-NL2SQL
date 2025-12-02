#!/bin/bash

if [ -d "hooks" ]; then
    cp -r hooks/* .git/hooks/ 2>/dev/null || echo "Could not copy git hooks"
    # Make the copied hooks executable
    chmod +x .git/hooks/* 2>/dev/null || echo "Could not make git hooks executable"
else
    echo "Hooks directory not found, skipping git hooks setup"
fi

if command -v gcloud >/dev/null 2>&1; then
    gcloud config set project research-su-llm-routing
else
    echo "gcloud not found, skipping project setup"
fi
