#!/bin/bash

# BERT4Rec粗排批量更新定时任务
# 每12小时全量重新计算所有用户的粗排结果
# 执行时间：00:00 和 12:00

set -euo pipefail  # 错误时立即退出

# ================== 配置参数 ==================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
DATA_DIR="${SCRIPT_DIR}/data"
BACKUP_DIR="${SCRIPT_DIR}/backup"

# Python环境
PYTHON_ENV="${SCRIPT_DIR}/venv/bin/python"
if [[ ! -f "$PYTHON_ENV" ]]; then
    PYTHON_ENV="python3"
fi

# Redis配置
REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"
REDIS_DB="${REDIS_DB:-0}"

# 批处理参数
BATCH_SIZE=1000
MAX_CONCURRENT_JOBS=4
RANKING_TOP_K=200

# 创建必要目录
mkdir -p "$LOG_DIR" "$DATA_DIR" "$BACKUP_DIR"

# 日志文件
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/batch_ranking_${TIMESTAMP}.log"
ERROR_LOG="${LOG_DIR}/error_${TIMESTAMP}.log"

# ================== 日志函数 ==================
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

error_log() {
    echo "[ERROR $(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$ERROR_LOG" >&2
}

# ================== 监控函数 ==================
send_alert() {
    local message="$1"
    local severity="${2:-INFO}"
    
    # 发送邮件告警（需要配置sendmail）
    if command -v mail >/dev/null 2>&1; then
        echo "$message" | mail -s "[RANKING-$severity] Batch Job Alert" ops@company.com
    fi
    
    # 发送到监控系统（示例：curl到监控API）
    if [[ "$severity" == "ERROR" ]]; then
        curl -X POST "http://monitor.internal/alert" \
             -H "Content-Type: application/json" \
             -d "{\"service\":\"batch_ranking\",\"level\":\"$severity\",\"message\":\"$message\"}" \
             2>/dev/null || true
    fi
}

# ================== 健康检查 ==================
health_check() {
    log "开始健康检查..."
    
    # 检查Redis连接
    if ! redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" ping >/dev/null 2>&1; then
        error_log "Redis连接失败: $REDIS_HOST:$REDIS_PORT"
        return 1
    fi
    
    # 检查磁盘空间（至少需要10GB）
    local available_space
    available_space=$(df "$DATA_DIR" | awk 'NR==2 {print $4}')
    if [[ $available_space -lt 10485760 ]]; then  # 10GB in KB
        error_log "磁盘空间不足: 可用空间 ${available_space}KB"
        return 1
    fi
    
    # 检查Python环境和依赖
    if ! "$PYTHON_ENV" -c "import numpy, torch" 2>/dev/null; then
        error_log "Python环境检查失败：缺少必要依赖"
        return 1
    fi
    
    log "健康检查通过"
    return 0
}

# ================== 数据备份 ==================
backup_previous_results() {
    log "备份上一轮结果..."
    
    local backup_file="${BACKUP_DIR}/ranking_backup_${TIMESTAMP}.rdb"
    
    # 备份Redis中的排序结果
    redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" --rdb "$backup_file" 2>/dev/null || {
        error_log "Redis备份失败"
        return 1
    }
    
    # 清理7天前的备份
    find "$BACKUP_DIR" -name "ranking_backup_*.rdb" -mtime +7 -delete
    
    log "备份完成: $backup_file"
}

# ================== 获取用户列表 ==================
get_active_users() {
    log "获取活跃用户列表..."
    
    # 从用户行为表获取最近7天有活动的用户
    local user_file="${DATA_DIR}/active_users_${TIMESTAMP}.txt"
    
    # 方法1: 从Redis获取（如果用户会话存储在Redis）
    redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" KEYS "session:*" | \
        sed 's/session://g' | head -100000 > "$user_file" 2>/dev/null || {
        
        # 方法2: 从数据库获取（示例SQL）
        mysql -h "${DB_HOST:-localhost}" -u "${DB_USER:-root}" -p"${DB_PASS:-}" \
              -D "${DB_NAME:-ecommerce}" \
              -e "SELECT DISTINCT user_id FROM user_behavior WHERE created_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)" \
              -N > "$user_file" 2>/dev/null || {
            
            # 方法3: 使用测试用户（开发环境）
            seq 10000 110000 > "$user_file"
        }
    }
    
    local user_count
    user_count=$(wc -l < "$user_file")
    log "获取到 $user_count 个活跃用户"
    
    echo "$user_file"
}

# ================== 批处理核心逻辑 ==================
process_user_batch() {
    local batch_file="$1"
    local batch_id="$2"
    local batch_log="${LOG_DIR}/batch_${batch_id}_${TIMESTAMP}.log"
    
    log "开始处理批次 $batch_id ($(wc -l < "$batch_file") 用户)"
    
    # 调用Python脚本处理这一批用户
    local python_script="${SCRIPT_DIR}/batch_ranking_processor.py"
    
    "$PYTHON_ENV" "$python_script" \
        --user-file "$batch_file" \
        --batch-id "$batch_id" \
        --redis-host "$REDIS_HOST" \
        --redis-port "$REDIS_PORT" \
        --top-k "$RANKING_TOP_K" \
        --log-file "$batch_log" 2>&1 || {
        
        error_log "批次 $batch_id 处理失败"
        return 1
    }
    
    log "批次 $batch_id 处理完成"
    return 0
}

# ================== 并行处理管理 ==================
run_parallel_batches() {
    local user_file="$1"
    local total_users
    total_users=$(wc -l < "$user_file")
    
    log "开始并行处理 $total_users 用户，批次大小: $BATCH_SIZE"
    
    # 拆分用户文件
    local batch_dir="${DATA_DIR}/batches_${TIMESTAMP}"
    mkdir -p "$batch_dir"
    
    split -l "$BATCH_SIZE" -d "$user_file" "${batch_dir}/batch_" --additional-suffix=".txt"
    
    local batch_files=("$batch_dir"/batch_*.txt)
    local total_batches=${#batch_files[@]}
    local failed_batches=0
    local jobs=0
    
    log "总共 $total_batches 个批次，最大并发: $MAX_CONCURRENT_JOBS"
    
    for batch_file in "${batch_files[@]}"; do
        local batch_id
        batch_id=$(basename "$batch_file" .txt)
        
        # 控制并发数
        while [[ $jobs -ge $MAX_CONCURRENT_JOBS ]]; do
            wait -n  # 等待任意一个后台任务完成
            jobs=$((jobs - 1))
        done
        
        # 后台执行批处理
        (
            process_user_batch "$batch_file" "$batch_id"
            exit $?
        ) &
        
        jobs=$((jobs + 1))
    done
    
    # 等待所有批次完成
    while [[ $jobs -gt 0 ]]; do
        wait -n
        local exit_code=$?
        if [[ $exit_code -ne 0 ]]; then
            failed_batches=$((failed_batches + 1))
        fi
        jobs=$((jobs - 1))
    done
    
    # 清理临时文件
    rm -rf "$batch_dir"
    
    if [[ $failed_batches -gt 0 ]]; then
        error_log "有 $failed_batches 个批次处理失败"
        return 1
    fi
    
    log "所有批次处理成功"
    return 0
}

# ================== 结果验证 ==================
validate_results() {
    log "验证处理结果..."
    
    # 检查Redis中的数据
    local total_keys
    total_keys=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" EVAL "
        local keys = redis.call('keys', 'rec:v2:user:*')
        return #keys
    " 0)
    
    log "Redis中有 $total_keys 个用户的推荐结果"
    
    # 抽样检查数据质量
    local sample_key="rec:v2:user:10001"
    local sample_count
    sample_count=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" ZCARD "$sample_key" 2>/dev/null || echo "0")
    
    if [[ $sample_count -lt 10 ]]; then
        error_log "数据质量检查失败: 样本用户推荐数量过少 ($sample_count)"
        return 1
    fi
    
    log "结果验证通过"
    return 0
}

# ================== 性能统计 ==================
generate_performance_report() {
    log "生成性能报告..."
    
    local report_file="${LOG_DIR}/performance_report_${TIMESTAMP}.json"
    local end_time=$(date +%s)
    local total_duration=$((end_time - START_TIME))
    
    # 统计处理的用户数
    local processed_users
    processed_users=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" EVAL "
        local keys = redis.call('keys', 'rec:v2:user:*')
        return #keys
    " 0)
    
    # 计算QPS
    local qps=0
    if [[ $total_duration -gt 0 ]]; then
        qps=$((processed_users / total_duration))
    fi
    
    # 生成JSON报告
    cat > "$report_file" << EOF
{
    "timestamp": "${TIMESTAMP}",
    "start_time": "${START_TIME}",
    "end_time": "${end_time}",
    "total_duration_seconds": ${total_duration},
    "processed_users": ${processed_users},
    "qps": ${qps},
    "batch_size": ${BATCH_SIZE},
    "max_concurrent_jobs": ${MAX_CONCURRENT_JOBS},
    "ranking_top_k": ${RANKING_TOP_K},
    "redis_host": "${REDIS_HOST}",
    "success": true
}
EOF
    
    log "性能报告: 处理 $processed_users 用户，耗时 $total_duration 秒，QPS: $qps"
    log "详细报告: $report_file"
}

# ================== 清理函数 ==================
cleanup() {
    log "清理临时文件..."
    
    # 清理临时数据文件
    find "$DATA_DIR" -name "*_${TIMESTAMP}*" -mtime +1 -delete
    
    # 清理旧日志（保留30天）
    find "$LOG_DIR" -name "*.log" -mtime +30 -delete
    
    log "清理完成"
}

# ================== 主流程 ==================
main() {
    local start_time=$(date +%s)
    START_TIME=$start_time
    
    log "=========================================="
    log "开始BERT4Rec批量排序更新任务"
    log "时间戳: $TIMESTAMP"
    log "=========================================="
    
    # 1. 健康检查
    if ! health_check; then
        send_alert "健康检查失败，任务终止" "ERROR"
        exit 1
    fi
    
    # 2. 备份上一轮结果
    if ! backup_previous_results; then
        send_alert "备份失败，但继续执行" "WARN"
    fi
    
    # 3. 获取活跃用户
    local user_file
    user_file=$(get_active_users)
    
    if [[ ! -f "$user_file" ]] || [[ ! -s "$user_file" ]]; then
        error_log "获取用户列表失败或为空"
        send_alert "获取用户列表失败" "ERROR"
        exit 1
    fi
    
    # 4. 并行批处理
    if ! run_parallel_batches "$user_file"; then
        error_log "批处理执行失败"
        send_alert "批处理执行失败" "ERROR"
        exit 1
    fi
    
    # 5. 验证结果
    if ! validate_results; then
        error_log "结果验证失败"
        send_alert "结果验证失败" "ERROR"
        exit 1
    fi
    
    # 6. 生成报告
    generate_performance_report
    
    # 7. 清理
    cleanup
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    log "=========================================="
    log "批量排序更新任务完成"
    log "总耗时: $duration 秒"
    log "=========================================="
    
    send_alert "批量排序更新任务完成，耗时 ${duration} 秒" "INFO"
}

# ================== 信号处理 ==================
trap 'error_log "任务被中断"; exit 1' INT TERM

# ================== 程序入口 ==================
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi