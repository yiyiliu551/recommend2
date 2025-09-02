#!/bin/bash

# 召回模型重训练定时任务
# 每天4点开始计算新模型（等特征计算完后）
# 依赖：特征工程pipeline已完成

set -euo pipefail

# ================== 配置参数 ==================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
MODEL_DIR="${SCRIPT_DIR}/models"
DATA_DIR="${SCRIPT_DIR}/data"
BACKUP_DIR="${SCRIPT_DIR}/model_backup"

# Python环境
PYTHON_ENV="${SCRIPT_DIR}/venv/bin/python"
if [[ ! -f "$PYTHON_ENV" ]]; then
    PYTHON_ENV="python3"
fi

# 模型配置
MODEL_VERSION="v$(date +%Y%m%d_%H)"
EMBEDDING_DIM=128
BATCH_SIZE=1024
EPOCHS=10
LEARNING_RATE=0.001

# 数据配置
TRAIN_DAYS=30        # 使用最近30天数据训练
MIN_USER_ACTIONS=10  # 用户最少行为次数
MIN_ITEM_ACTIONS=5   # 商品最少被行为次数

# Spark配置
SPARK_MASTER="${SPARK_MASTER:-local[*]}"
SPARK_DRIVER_MEMORY="${SPARK_DRIVER_MEMORY:-8g}"
SPARK_EXECUTOR_MEMORY="${SPARK_EXECUTOR_MEMORY:-4g}"

# 创建目录
mkdir -p "$LOG_DIR" "$MODEL_DIR" "$DATA_DIR" "$BACKUP_DIR"

# 日志文件
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/recall_retrain_${TIMESTAMP}.log"
ERROR_LOG="${LOG_DIR}/recall_error_${TIMESTAMP}.log"

# ================== 日志和监控函数 ==================
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

error_log() {
    echo "[ERROR $(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$ERROR_LOG" >&2
}

send_alert() {
    local message="$1"
    local severity="${2:-INFO}"
    
    # 发送告警到监控系统
    curl -X POST "http://monitor.internal/alert" \
         -H "Content-Type: application/json" \
         -d "{\"service\":\"recall_retrain\",\"level\":\"$severity\",\"message\":\"$message\",\"timestamp\":\"$(date -Iseconds)\"}" \
         2>/dev/null || true
}

# ================== 前置检查 ==================
check_prerequisites() {
    log "检查前置条件..."
    
    # 1. 检查特征工程是否完成
    local feature_status_file="/tmp/feature_pipeline_status"
    if [[ -f "$feature_status_file" ]]; then
        local last_feature_update
        last_feature_update=$(cat "$feature_status_file" 2>/dev/null || echo "0")
        local current_time=$(date +%s)
        local time_diff=$((current_time - last_feature_update))
        
        # 如果特征更新超过6小时，可能有问题
        if [[ $time_diff -gt 21600 ]]; then
            error_log "特征工程可能未正常完成，上次更新: $(date -d @"$last_feature_update" 2>/dev/null || echo "unknown")"
            return 1
        fi
    else
        error_log "特征工程状态文件不存在: $feature_status_file"
        return 1
    fi
    
    # 2. 检查Hadoop/Spark集群状态
    if ! hadoop fs -ls / >/dev/null 2>&1; then
        error_log "Hadoop集群连接失败"
        return 1
    fi
    
    # 3. 检查GPU资源（如果需要）
    if command -v nvidia-smi >/dev/null 2>&1; then
        local gpu_memory
        gpu_memory=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
        if [[ $gpu_memory -lt 4000 ]]; then  # 需要至少4GB显存
            error_log "GPU内存不足: ${gpu_memory}MB"
            return 1
        fi
        log "GPU检查通过: ${gpu_memory}MB 可用显存"
    fi
    
    # 4. 检查磁盘空间
    local available_gb
    available_gb=$(df "$MODEL_DIR" | awk 'NR==2 {printf "%.0f", $4/1024/1024}')
    if [[ $available_gb -lt 50 ]]; then
        error_log "磁盘空间不足: ${available_gb}GB"
        return 1
    fi
    
    log "前置检查通过"
    return 0
}

# ================== 数据准备 ==================
prepare_training_data() {
    log "开始数据准备..."
    
    local data_date
    data_date=$(date +%Y-%m-%d)
    local output_path="/tmp/recall_training_data_${TIMESTAMP}"
    
    # 使用Spark SQL提取训练数据
    local spark_sql="${DATA_DIR}/extract_training_data.sql"
    
    cat > "$spark_sql" << EOF
-- 提取最近${TRAIN_DAYS}天的用户行为数据
SELECT 
    user_id,
    item_id,
    action_type,
    timestamp,
    -- 用户特征
    user_age_group,
    user_gender,
    user_region,
    -- 商品特征
    item_category,
    item_brand,
    item_price_range,
    -- 上下文特征
    hour_of_day,
    day_of_week
FROM user_behavior_enriched 
WHERE 
    dt >= date_sub('${data_date}', ${TRAIN_DAYS})
    AND dt <= '${data_date}'
    AND action_type IN ('view', 'click', 'purchase', 'cart')
    -- 过滤低质量数据
    AND user_id IS NOT NULL 
    AND item_id IS NOT NULL
    AND user_id != 0 
    AND item_id != 0
EOF
    
    # 执行Spark作业
    log "执行Spark数据提取作业..."
    spark-sql \
        --master "$SPARK_MASTER" \
        --driver-memory "$SPARK_DRIVER_MEMORY" \
        --executor-memory "$SPARK_EXECUTOR_MEMORY" \
        --conf spark.sql.adaptive.enabled=true \
        --conf spark.sql.adaptive.coalescePartitions.enabled=true \
        -f "$spark_sql" \
        --hiveconf hive.exec.dynamic.partition=true \
        --hiveconf hive.exec.dynamic.partition.mode=nonstrict > "${LOG_DIR}/spark_extract_${TIMESTAMP}.log" 2>&1 || {
        
        error_log "Spark数据提取失败"
        return 1
    }
    
    # 数据质量检查
    local total_interactions
    total_interactions=$(hadoop fs -cat "/tmp/recall_training_data_${TIMESTAMP}/part-*" | wc -l 2>/dev/null || echo "0")
    
    if [[ $total_interactions -lt 1000000 ]]; then  # 至少100万条交互
        error_log "训练数据量不足: $total_interactions 条"
        return 1
    fi
    
    log "数据准备完成: $total_interactions 条交互数据"
    echo "$output_path"
}

# ================== 模型训练 ==================
train_recall_models() {
    local training_data_path="$1"
    log "开始训练召回模型..."
    
    local model_output_dir="${MODEL_DIR}/recall_${MODEL_VERSION}"
    mkdir -p "$model_output_dir"
    
    # 训练配置文件
    local config_file="${model_output_dir}/training_config.json"
    cat > "$config_file" << EOF
{
    "model_version": "${MODEL_VERSION}",
    "training_data_path": "${training_data_path}",
    "output_dir": "${model_output_dir}",
    "hyperparameters": {
        "embedding_dim": ${EMBEDDING_DIM},
        "batch_size": ${BATCH_SIZE},
        "epochs": ${EPOCHS},
        "learning_rate": ${LEARNING_RATE},
        "dropout_rate": 0.2,
        "l2_reg": 0.0001
    },
    "data_config": {
        "train_ratio": 0.8,
        "valid_ratio": 0.1,
        "test_ratio": 0.1,
        "min_user_actions": ${MIN_USER_ACTIONS},
        "min_item_actions": ${MIN_ITEM_ACTIONS}
    }
}
EOF
    
    # 1. 训练ItemCF模型
    log "训练ItemCF模型..."
    "$PYTHON_ENV" "${SCRIPT_DIR}/train_itemcf.py" \
        --config "$config_file" \
        --model-type "itemcf" \
        --output "${model_output_dir}/itemcf_model" 2>&1 | tee -a "${LOG_DIR}/itemcf_${TIMESTAMP}.log" || {
        error_log "ItemCF模型训练失败"
        return 1
    }
    
    # 2. 训练向量召回模型（Two-Tower）
    log "训练Two-Tower向量模型..."
    "$PYTHON_ENV" "${SCRIPT_DIR}/train_two_tower.py" \
        --config "$config_file" \
        --model-type "two_tower" \
        --output "${model_output_dir}/two_tower_model" \
        --use-gpu \
        2>&1 | tee -a "${LOG_DIR}/two_tower_${TIMESTAMP}.log" || {
        error_log "Two-Tower模型训练失败"
        return 1
    }
    
    # 3. 训练Graph Neural Network召回模型
    log "训练GNN召回模型..."
    "$PYTHON_ENV" "${SCRIPT_DIR}/train_gnn_recall.py" \
        --config "$config_file" \
        --model-type "gnn" \
        --output "${model_output_dir}/gnn_model" \
        --graph-type "user_item_bipartite" \
        2>&1 | tee -a "${LOG_DIR}/gnn_${TIMESTAMP}.log" || {
        error_log "GNN模型训练失败"
        return 1
    }
    
    log "所有召回模型训练完成"
    echo "$model_output_dir"
}

# ================== 模型评估 ==================
evaluate_models() {
    local model_dir="$1"
    log "开始模型评估..."
    
    local evaluation_report="${model_dir}/evaluation_report.json"
    
    # 运行离线评估
    "$PYTHON_ENV" "${SCRIPT_DIR}/evaluate_recall_models.py" \
        --model-dir "$model_dir" \
        --test-data "/tmp/recall_test_data_${TIMESTAMP}" \
        --metrics "recall@50,recall@100,ndcg@50,coverage,diversity" \
        --output "$evaluation_report" 2>&1 | tee -a "${LOG_DIR}/evaluation_${TIMESTAMP}.log" || {
        error_log "模型评估失败"
        return 1
    }
    
    # 检查评估结果
    local recall_50
    recall_50=$(python3 -c "
import json
with open('$evaluation_report') as f:
    data = json.load(f)
print(data.get('recall@50', 0))
" 2>/dev/null || echo "0")
    
    if (( $(echo "$recall_50 < 0.1" | bc -l) )); then
        error_log "模型效果不达标: Recall@50 = $recall_50"
        return 1
    fi
    
    log "模型评估完成: Recall@50 = $recall_50"
    return 0
}

# ================== 模型部署 ==================
deploy_models() {
    local model_dir="$1"
    log "开始模型部署..."
    
    # 1. 备份当前生产模型
    local current_model_link="${MODEL_DIR}/current"
    if [[ -L "$current_model_link" ]]; then
        local current_model_path
        current_model_path=$(readlink "$current_model_link")
        local backup_name="backup_$(date +%Y%m%d_%H%M%S)"
        ln -sfn "$current_model_path" "${BACKUP_DIR}/$backup_name"
        log "当前模型已备份: $backup_name"
    fi
    
    # 2. 更新模型软链接
    ln -sfn "$model_dir" "$current_model_link"
    
    # 3. 重建向量索引
    log "重建向量索引..."
    "$PYTHON_ENV" "${SCRIPT_DIR}/build_vector_index.py" \
        --model-dir "$current_model_link" \
        --index-type "faiss" \
        --output "${MODEL_DIR}/vector_index" \
        --rebuild 2>&1 | tee -a "${LOG_DIR}/build_index_${TIMESTAMP}.log" || {
        error_log "向量索引构建失败"
        return 1
    }
    
    # 4. 通知召回服务重新加载模型
    log "通知召回服务重新加载..."
    curl -X POST "http://recall-service:8080/reload" \
         -H "Content-Type: application/json" \
         -d "{\"model_version\":\"$MODEL_VERSION\",\"model_path\":\"$current_model_link\"}" \
         2>/dev/null || {
        error_log "通知召回服务失败"
        return 1
    }
    
    # 5. 健康检查
    sleep 30  # 等待服务加载完成
    local health_status
    health_status=$(curl -s "http://recall-service:8080/health" | jq -r '.status' 2>/dev/null || echo "unknown")
    
    if [[ "$health_status" != "ok" ]]; then
        error_log "召回服务健康检查失败: $health_status"
        return 1
    fi
    
    log "模型部署完成"
    return 0
}

# ================== 清理函数 ==================
cleanup_old_models() {
    log "清理旧模型..."
    
    # 保留最近7天的模型
    find "$MODEL_DIR" -maxdepth 1 -type d -name "recall_v*" -mtime +7 -exec rm -rf {} \;
    
    # 清理训练数据
    hadoop fs -rm -r -skipTrash "/tmp/recall_training_data_*" 2>/dev/null || true
    hadoop fs -rm -r -skipTrash "/tmp/recall_test_data_*" 2>/dev/null || true
    
    log "清理完成"
}

# ================== 性能报告 ==================
generate_training_report() {
    local model_dir="$1"
    local end_time=$(date +%s)
    local total_duration=$((end_time - START_TIME))
    
    local report_file="${LOG_DIR}/training_report_${TIMESTAMP}.json"
    
    cat > "$report_file" << EOF
{
    "timestamp": "${TIMESTAMP}",
    "model_version": "${MODEL_VERSION}",
    "training_duration_seconds": ${total_duration},
    "training_config": {
        "train_days": ${TRAIN_DAYS},
        "embedding_dim": ${EMBEDDING_DIM},
        "batch_size": ${BATCH_SIZE},
        "epochs": ${EPOCHS}
    },
    "model_path": "${model_dir}",
    "success": true,
    "deployment_time": "$(date -Iseconds)"
}
EOF
    
    log "训练报告生成: $report_file"
    log "模型版本: $MODEL_VERSION"
    log "总耗时: $((total_duration / 3600))小时$((total_duration % 3600 / 60))分钟"
}

# ================== 主流程 ==================
main() {
    local start_time=$(date +%s)
    START_TIME=$start_time
    
    log "=========================================="
    log "开始召回模型重训练任务"
    log "模型版本: $MODEL_VERSION"
    log "时间戳: $TIMESTAMP"
    log "=========================================="
    
    # 1. 前置检查
    if ! check_prerequisites; then
        send_alert "前置检查失败，任务终止" "ERROR"
        exit 1
    fi
    
    # 2. 数据准备
    local training_data_path
    if ! training_data_path=$(prepare_training_data); then
        send_alert "数据准备失败" "ERROR"
        exit 1
    fi
    
    # 3. 模型训练
    local model_dir
    if ! model_dir=$(train_recall_models "$training_data_path"); then
        send_alert "模型训练失败" "ERROR"
        exit 1
    fi
    
    # 4. 模型评估
    if ! evaluate_models "$model_dir"; then
        send_alert "模型评估失败" "ERROR"
        exit 1
    fi
    
    # 5. 模型部署
    if ! deploy_models "$model_dir"; then
        send_alert "模型部署失败" "ERROR"
        exit 1
    fi
    
    # 6. 清理
    cleanup_old_models
    
    # 7. 生成报告
    generate_training_report "$model_dir"
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    log "=========================================="
    log "召回模型重训练任务完成"
    log "新模型版本: $MODEL_VERSION"
    log "总耗时: $duration 秒"
    log "=========================================="
    
    send_alert "召回模型重训练完成，版本: $MODEL_VERSION，耗时: ${duration}秒" "INFO"
    
    # 记录完成状态
    echo "$end_time" > "/tmp/recall_retrain_status"
}

# ================== 信号处理 ==================
trap 'error_log "任务被中断"; exit 1' INT TERM

# ================== 程序入口 ==================
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi