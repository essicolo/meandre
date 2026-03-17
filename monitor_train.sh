#!/bin/bash
# Monitor script for training with resource tracking

# Start monitoring in background
(
    echo "timestamp,cpu_percent,mem_used_gb,gpu_mem_used_mb,gpu_util"
    while true; do
        TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
        CPU=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1}')
        MEM=$(free -m | awk 'NR==2{printf "%.1f", $3/1024}')

        # GPU metrics (if available)
        if command -v nvidia-smi &> /dev/null; then
            GPU_INFO=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
            if [ $? -eq 0 ]; then
                GPU_MEM=$(echo $GPU_INFO | cut -d',' -f1 | tr -d ' ')
                GPU_UTIL=$(echo $GPU_INFO | cut -d',' -f2 | tr -d ' ')
            else
                GPU_MEM="N/A"
                GPU_UTIL="N/A"
            fi
        else
            GPU_MEM="N/A"
            GPU_UTIL="N/A"
        fi

        echo "$TIMESTAMP,$CPU,$MEM,$GPU_MEM,$GPU_UTIL"
        sleep 5
    done
) > training_monitor.csv &

MONITOR_PID=$!
echo "Started monitoring with PID: $MONITOR_PID"

# Run the training
echo "Starting training..."
uv run python scripts/train_slso.py

# Stop monitoring
echo "Training finished. Stopping monitor..."
kill $MONITOR_PID 2>/dev/null

# Show summary
echo -e "\n=== Resource Usage Summary ==="
if [ -f training_monitor.csv ]; then
    echo "Peak values from monitoring:"
    tail -n +2 training_monitor.csv | awk -F',' '{
        if ($2 > max_cpu) max_cpu = $2;
        if ($3 > max_mem) max_mem = $3;
        if ($4 != "N/A" && $4 > max_gpu_mem) max_gpu_mem = $4;
        if ($5 != "N/A" && $5 > max_gpu_util) max_gpu_util = $5;
    }
    END {
        print "  Max CPU: " max_cpu "%";
        print "  Max Memory: " max_mem " GB";
        if (max_gpu_mem > 0) print "  Max GPU Memory: " max_gpu_mem " MB";
        if (max_gpu_util > 0) print "  Max GPU Utilization: " max_gpu_util "%";
    }'
else
    echo "No monitoring data collected"
fi