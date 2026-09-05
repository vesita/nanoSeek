#!/usr/bin/env bash
# 克隆开源数据仓库：代码算法库 + 中文文本语料库
# 全部 shallow clone (--depth 1) 节省时间与磁盘
set -u
BASE="/home/vesita/coding/my/nanoSeek/data/external"
mkdir -p "$BASE"
cd "$BASE"

clone_repo() {
    local url="$1"
    local name="$2"
    if [ -d "$name/.git" ]; then
        echo "✓ 已存在 $name，跳过"
        return
    fi
    echo ">>> 克隆 $name ..."
    if timeout 300 git clone --depth 1 "$url" "$name" 2>&1 | tail -3; then
        echo "✓ $name 克隆完成 ($(du -sh "$name" 2>/dev/null | cut -f1))"
    else
        echo "✗ $name 克隆失败"
    fi
}

# --- 代码算法库（补充代码语料）---
clone_repo "https://github.com/TheAlgorithms/Python.git" "TheAlgorithms-Python"
clone_repo "https://github.com/doocs/leetcode.git" "doocs-leetcode"

# --- 中文文本语料库（补充诗词/文化语料）---
clone_repo "https://github.com/chinese-poetry/chinese-poetry.git" "chinese-poetry"

echo "=== 全部克隆任务结束 ==="
ls -la "$BASE"
