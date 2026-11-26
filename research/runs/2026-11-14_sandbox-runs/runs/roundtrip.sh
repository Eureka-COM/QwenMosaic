#!/bin/bash
cd /home/claude/mosaic-qwen
rm -rf /home/claude/runs/roundtrip-0p8b /home/claude/runs/roundtrip-work
for step in prepare saved_fp32 saved_bf16; do
  python3 scripts/mq_roundtrip.py --model /home/claude/models/Qwen3.5-0.8B --delta /home/claude/runs/d1-s3-r10-delta --data /home/claude/data/cosmopedia-100k@4 --data /home/claude/data/ultrachat:messages@2 --data /home/claude/data/wikipedia-ja@2 --data /home/claude/data/codeparrot:content@1 --out /home/claude/runs/roundtrip-0p8b --work /home/claude/runs/roundtrip-work --step $step || echo "STEP $step FAILED $?"
done
echo FINISHED
