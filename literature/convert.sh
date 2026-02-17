#!/bin/bash
# Bulk convert PDFs to markdown using markitdown
# Generated: 2026-02-16

PDF_DIR="/Users/sam/Documents/Research/ds6050/image-cnn-transformers/literature/pdf"
MD_DIR="/Users/sam/Documents/Research/ds6050/image-cnn-transformers/literature/md"

mkdir -p "$MD_DIR"

SUCCESS=0
FAIL=0

for pdf in "$PDF_DIR"/*.pdf; do
    basename=$(basename "$pdf" .pdf)
    md_file="$MD_DIR/${basename}.md"

    if [ -f "$md_file" ]; then
        echo "SKIP: ${basename} (already converted)"
        continue
    fi

    echo -n "Converting ${basename}..."
    if markitdown "$pdf" > "$md_file" 2>/dev/null; then
        size=$(wc -c < "$md_file" | tr -d ' ')
        if [ "$size" -lt 1024 ]; then
            echo " WARNING: output is only ${size} bytes (possible failure)"
        else
            echo " OK (${size} bytes)"
        fi
        SUCCESS=$((SUCCESS + 1))
    else
        echo " FAILED"
        rm -f "$md_file"
        FAIL=$((FAIL + 1))
    fi
done

echo ""
echo "Done: ${SUCCESS} converted, ${FAIL} failed"
