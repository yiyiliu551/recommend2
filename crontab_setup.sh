#!/bin/bash

# Crontab定时任务设置脚本
# 配置所有推荐系统相关的定时任务

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"

# 创建日志目录
mkdir -p "$LOG_DIR"

# ================== Crontab任务配置 ==================

setup_crontab() {
    local cron_file="/tmp/recommendation_crontab"
    
    cat > "$cron_file" << 'EOF'
# 推荐系统定时任务配置
# 生成时间: $(date)

# ==================== 特征工程任务 ====================
# 每天凌晨1点开始离线特征计算（最先执行，为后续任务提供基础）
0 1 * * * cd /path/to/sony-interview-projects && bash feature_pipeline_cron.sh >> logs/feature_cron.log 2>&1

# 实时特征流处理（持续运行，每小时检查一次状态）
0 * * * * cd /path/to/sony-interview-projects && bash check_flink_status.sh >> logs/flink_check.log 2>&1

# ==================== 召回模型训练 ====================
# 每天凌晨4点开始召回模型重训练（等特征计算完成后）
0 4 * * * cd /path/to/sony-interview-projects && bash recall_model_retrain_cron.sh >> logs/recall_retrain.log 2>&1

# ==================== 排序任务 ====================
# 粗排更新：每12小时执行一次（00:00 和 12:00）
0 0,12 * * * cd /path/to/sony-interview-projects && bash batch_ranking_cron.sh >> logs/batch_ranking.log 2>&1

# 精排热门用户：每2小时更新一次
0 */2 * * * cd /path/to/sony-interview-projects && bash hot_users_ranking.sh >> logs/hot_users_ranking.log 2>&1

# ==================== 系统监控任务 ====================
# 每5分钟检查系统健康状态
*/5 * * * * cd /path/to/sony-interview-projects && bash system_health_check.sh >> logs/health_check.log 2>&1

# 每小时统计推荐系统性能指标
0 * * * * cd /path/to/sony-interview-projects && bash collect_metrics.sh >> logs/metrics.log 2>&1

# ==================== 数据清理任务 ====================
# 每天凌晨3点清理过期的Redis缓存
0 3 * * * redis-cli EVAL "for i, name in ipairs(redis.call('KEYS', 'rec:*')) do if redis.call('TTL', name) == -1 then redis.call('DEL', name) end end" 0

# 每周日凌晨2点清理旧日志文件（保留30天）
0 2 * * 0 find /path/to/sony-interview-projects/logs -name "*.log" -mtime +30 -delete

# 每月1号清理旧模型文件（保留3个月）
0 2 1 * * find /path/to/sony-interview-projects/models -name "*" -mtime +90 -delete

# ==================== 数据备份任务 ====================
# 每天凌晨5点备份重要配置和模型
0 5 * * * cd /path/to/sony-interview-projects && bash backup_critical_data.sh >> logs/backup.log 2>&1

EOF

    # 替换实际路径
    sed -i "s|/path/to/sony-interview-projects|$SCRIPT_DIR|g" "$cron_file"
    sed -i "s|# 生成时间: \$(date)|# 生成时间: $(date)|" "$cron_file"
    
    echo "Crontab配置文件已生成: $cron_file"
    echo ""
    echo "要安装这些定时任务，请运行："
    echo "  crontab $cron_file"
    echo ""
    echo "要查看当前定时任务，请运行："
    echo "  crontab -l"
    echo ""
    echo "要编辑定时任务，请运行："
    echo "  crontab -e"
}

# ================== 辅助脚本生成 ==================

create_helper_scripts() {
    echo "创建辅助监控脚本..."
    
    # 1. Flink状态检查脚本
    cat > "${SCRIPT_DIR}/check_flink_status.sh" << 'EOF'
#!/bin/bash
# 检查Flink实时特征处理任务状态

FLINK_JOB_NAME="RealTimeFeatureProcessor"
LOG_FILE="logs/flink_check_$(date +%Y%m%d).log"

# 检查Flink任务是否在运行
RUNNING_JOBS=$(flink list -r 2>/dev/null | grep "$FLINK_JOB_NAME" | wc -l)

if [[ $RUNNING_JOBS -eq 0 ]]; then
    echo "[$(date)] Flink任务未运行，尝试重启..." >> "$LOG_FILE"
    
    # 重启Flink任务
    python3 flink_realtime_features_new.py --restart >> "$LOG_FILE" 2>&1
    
    if [[ $? -eq 0 ]]; then
        echo "[$(date)] Flink任务重启成功" >> "$LOG_FILE"
    else
        echo "[$(date)] Flink任务重启失败" >> "$LOG_FILE"
        # 发送告警
        curl -X POST "http://monitor.internal/alert" \
             -d "service=flink&level=ERROR&message=Flink任务重启失败" 2>/dev/null || true
    fi
else
    echo "[$(date)] Flink任务正常运行，任务数: $RUNNING_JOBS" >> "$LOG_FILE"
fi
EOF

    # 2. 系统健康检查脚本
    cat > "${SCRIPT_DIR}/system_health_check.sh" << 'EOF'
#!/bin/bash
# 系统健康检查

LOG_FILE="logs/health_$(date +%Y%m%d).log"

check_redis() {
    if redis-cli ping >/dev/null 2>&1; then
        echo "✅ Redis: OK"
    else
        echo "❌ Redis: FAILED"
        return 1
    fi
}

check_disk_space() {
    local usage=$(df . | awk 'NR==2 {print $5}' | sed 's/%//')
    if [[ $usage -lt 85 ]]; then
        echo "✅ Disk Space: ${usage}% used"
    else
        echo "⚠️  Disk Space: ${usage}% used (HIGH)"
        return 1
    fi
}

check_memory() {
    local mem_usage=$(free | awk 'NR==2{printf "%.0f", $3*100/$2}')
    if [[ $mem_usage -lt 85 ]]; then
        echo "✅ Memory: ${mem_usage}% used"
    else
        echo "⚠️  Memory: ${mem_usage}% used (HIGH)"
        return 1
    fi
}

{
    echo "[$(date)] 系统健康检查开始"
    
    ISSUES=0
    check_redis || ISSUES=$((ISSUES + 1))
    check_disk_space || ISSUES=$((ISSUES + 1))
    check_memory || ISSUES=$((ISSUES + 1))
    
    if [[ $ISSUES -eq 0 ]]; then
        echo "[$(date)] 系统状态正常"
    else
        echo "[$(date)] 发现 $ISSUES 个问题"
        # 发送告警
        curl -X POST "http://monitor.internal/alert" \
             -d "service=system&level=WARN&message=发现${ISSUES}个系统问题" 2>/dev/null || true
    fi
    echo "----------------------------------------"
} >> "$LOG_FILE"
EOF

    # 3. 热门用户排序脚本
    cat > "${SCRIPT_DIR}/hot_users_ranking.sh" << 'EOF'
#!/bin/bash
# 热门用户精排更新（VIP用户、高活跃用户）

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="logs/hot_users_$(date +%Y%m%d_%H).log"

{
    echo "[$(date)] 开始热门用户排序更新"
    
    # 获取VIP用户列表
    VIP_USERS=$(redis-cli SMEMBERS "users:vip" 2>/dev/null || echo "")
    
    # 获取高活跃用户（最近1小时有行为）
    ACTIVE_USERS=$(redis-cli ZRANGEBYSCORE "users:activity" "$(date -d '1 hour ago' +%s)" "+inf" 2>/dev/null || echo "")
    
    # 合并用户列表
    ALL_HOT_USERS=$(echo -e "$VIP_USERS\n$ACTIVE_USERS" | sort -u | head -1000)
    
    if [[ -n "$ALL_HOT_USERS" ]]; then
        echo "$ALL_HOT_USERS" > "/tmp/hot_users_$(date +%H).txt"
        
        # 调用批量处理脚本
        python3 batch_ranking_processor.py \
            --user-file "/tmp/hot_users_$(date +%H).txt" \
            --batch-id "hot_users_$(date +%H)" \
            --redis-host localhost \
            --redis-port 6379 \
            --top-k 200
        
        echo "[$(date)] 热门用户排序完成，处理用户数: $(echo "$ALL_HOT_USERS" | wc -l)"
    else
        echo "[$(date)] 未找到热门用户"
    fi
} >> "$LOG_FILE" 2>&1
EOF

    # 4. 指标收集脚本
    cat > "${SCRIPT_DIR}/collect_metrics.sh" << 'EOF'
#!/bin/bash
# 收集推荐系统性能指标

METRICS_FILE="logs/metrics_$(date +%Y%m%d_%H).json"

{
    echo "{"
    echo "  \"timestamp\": \"$(date -Iseconds)\","
    echo "  \"redis_info\": {"
    
    # Redis指标
    REDIS_INFO=$(redis-cli info stats 2>/dev/null || echo "")
    if [[ -n "$REDIS_INFO" ]]; then
        TOTAL_COMMANDS=$(echo "$REDIS_INFO" | grep "total_commands_processed" | cut -d: -f2 | tr -d '\r')
        KEYSPACE_HITS=$(echo "$REDIS_INFO" | grep "keyspace_hits" | cut -d: -f2 | tr -d '\r')
        KEYSPACE_MISSES=$(echo "$REDIS_INFO" | grep "keyspace_misses" | cut -d: -f2 | tr -d '\r')
        
        echo "    \"total_commands\": ${TOTAL_COMMANDS:-0},"
        echo "    \"keyspace_hits\": ${KEYSPACE_HITS:-0},"
        echo "    \"keyspace_misses\": ${KEYSPACE_MISSES:-0},"
        
        if [[ ${KEYSPACE_HITS:-0} -gt 0 ]] && [[ ${KEYSPACE_MISSES:-0} -gt 0 ]]; then
            HIT_RATE=$(echo "scale=4; $KEYSPACE_HITS / ($KEYSPACE_HITS + $KEYSPACE_MISSES)" | bc)
            echo "    \"hit_rate\": $HIT_RATE"
        else
            echo "    \"hit_rate\": 0"
        fi
    else
        echo "    \"total_commands\": 0,"
        echo "    \"keyspace_hits\": 0,"
        echo "    \"keyspace_misses\": 0,"
        echo "    \"hit_rate\": 0"
    fi
    
    echo "  },"
    echo "  \"system_info\": {"
    
    # 系统指标
    echo "    \"cpu_usage\": $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//' || echo "0"),"
    echo "    \"memory_usage\": $(free | awk 'NR==2{printf "%.1f", $3*100/$2}' || echo "0"),"
    echo "    \"disk_usage\": $(df . | awk 'NR==2 {print $5}' | sed 's/%//' || echo "0")"
    
    echo "  }"
    echo "}"
} > "$METRICS_FILE"
EOF

    # 5. 备份脚本
    cat > "${SCRIPT_DIR}/backup_critical_data.sh" << 'EOF'
#!/bin/bash
# 备份关键数据

BACKUP_DIR="backup/$(date +%Y%m%d)"
mkdir -p "$BACKUP_DIR"

LOG_FILE="logs/backup_$(date +%Y%m%d).log"

{
    echo "[$(date)] 开始数据备份"
    
    # 备份配置文件
    tar -czf "${BACKUP_DIR}/configs.tar.gz" *.py *.sh *.json 2>/dev/null || true
    
    # 备份当前模型软链接指向的目录
    if [[ -L "models/current" ]]; then
        MODEL_PATH=$(readlink models/current)
        tar -czf "${BACKUP_DIR}/current_model.tar.gz" "$MODEL_PATH" 2>/dev/null || true
    fi
    
    # 备份Redis关键数据
    redis-cli --rdb "${BACKUP_DIR}/redis_backup.rdb" 2>/dev/null || true
    
    # 清理15天前的备份
    find backup -type d -mtime +15 -exec rm -rf {} \; 2>/dev/null || true
    
    echo "[$(date)] 数据备份完成"
} >> "$LOG_FILE"
EOF

    # 使脚本可执行
    chmod +x "${SCRIPT_DIR}"/check_flink_status.sh
    chmod +x "${SCRIPT_DIR}"/system_health_check.sh
    chmod +x "${SCRIPT_DIR}"/hot_users_ranking.sh
    chmod +x "${SCRIPT_DIR}"/collect_metrics.sh
    chmod +x "${SCRIPT_DIR}"/backup_critical_data.sh
    
    echo "辅助脚本创建完成"
}

# ================== 任务依赖关系图 ==================
show_task_dependencies() {
    echo ""
    echo "=========================================="
    echo "推荐系统定时任务依赖关系图"
    echo "=========================================="
    echo ""
    echo "时间轴 | 任务名称                 | 依赖关系"
    echo "-------|-------------------------|------------------"
    echo "01:00  | 特征工程pipeline         | 独立执行"
    echo "03:00  | Redis缓存清理           | 独立执行"  
    echo "04:00  | 召回模型重训练           | 依赖: 特征工程完成"
    echo "00:00  | 批量排序更新             | 依赖: 召回模型"
    echo "12:00  | 批量排序更新             | 依赖: 召回模型"
    echo "*/2hr  | 热门用户精排             | 依赖: 召回+排序"
    echo "*/5min | 系统健康检查             | 独立执行"
    echo "每小时  | Flink状态检查           | 独立执行"
    echo "每小时  | 性能指标收集             | 独立执行"
    echo ""
    echo "关键时间窗口："
    echo "  01:00-04:00: 特征工程 → 召回模型训练"
    echo "  04:00-06:00: 召回模型训练 → 部署"
    echo "  00:00/12:00: 批量排序更新（12小时一次）"
    echo ""
}

# ================== 主函数 ==================
main() {
    echo "推荐系统Crontab定时任务设置"
    echo "=============================="
    
    # 创建必要的脚本
    create_helper_scripts
    
    # 设置crontab
    setup_crontab
    
    # 显示任务依赖关系
    show_task_dependencies
    
    # 使主要脚本可执行
    chmod +x "${SCRIPT_DIR}/batch_ranking_cron.sh"
    chmod +x "${SCRIPT_DIR}/recall_model_retrain_cron.sh"
    
    echo "=========================================="
    echo "设置完成！"
    echo ""
    echo "下一步操作："
    echo "1. 检查并修改脚本中的路径配置"
    echo "2. 运行: crontab /tmp/recommendation_crontab"
    echo "3. 验证: crontab -l"
    echo "4. 监控: tail -f logs/*.log"
    echo ""
    echo "重要提醒："
    echo "- 确保Redis、Hadoop、Spark等服务正常运行"
    echo "- 配置监控告警系统的API端点"
    echo "- 定期检查磁盘空间和日志清理"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi