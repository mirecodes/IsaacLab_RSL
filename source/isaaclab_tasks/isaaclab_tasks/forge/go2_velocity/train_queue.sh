#!/usr/bin/env bash
# Train the study policies one after another inside the container.
# Launch detached so it survives SSH disconnects:
#   docker exec -d -u 0 isaac-lab-kitless-v3 bash -lc 'nohup bash /workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/forge/go2_velocity/train_queue.sh > /workspace/isaaclab/logs/train_queue.log 2>&1'
# Each run logs to /workspace/isaaclab/logs/<run_name>.log; a failed run does not stop the next one.

cd /workspace/isaaclab || exit 1
export DISPLAY=
export WANDB_USERNAME=mirecodes

run() {
    local task=$1 run_name=$2
    echo "[$(date '+%F %T')] start ${run_name} (${task})"
    python scripts/reinforcement_learning/train.py --rl_library rsl_rl --task "${task}" --logger wandb \
        --run_name "${run_name}" --seed 0 --video > "/workspace/isaaclab/logs/${run_name}.log" 2>&1
    local status=$?
    echo "[$(date '+%F %T')] end ${run_name} (exit ${status})"
}

run IsaacContrib-Velocity-Go2Study-RobustTeacher robust_teacher_v2_seed0_rerun
run IsaacContrib-Velocity-Go2Study-EnergyTeacherRough energy_teacher_rough_seed0
