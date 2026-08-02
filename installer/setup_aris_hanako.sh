#!/usr/bin/env bash
# LAAP Hanako 一键安装脚本 (Linux/macOS)
# 用法：chmod +x installer/setup_aris_hanako.sh && ./installer/setup_aris_hanako.sh
#
# 幂等：重复运行不会报错，已存在的环境变量/目录会跳过。

set -u

echo "============================================"
echo "  LAAP Hanako 一键安装"
echo "============================================"
echo ""

# ── 安装脚本所在目录的上一级即 LAAP 根目录 ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAAP_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
echo "[INFO] LAAP_ROOT = $LAAP_ROOT"

# ── 1. 检测 Python 3 ─────────────────────────────────────────
if command -v python3 >/dev/null 2>&1; then
    PY="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
    PY="$(command -v python)"
else
    echo "[FAIL] 未检测到 Python，请安装 Python 3.11+"
    echo "       下载地址：https://www.python.org/downloads/"
    exit 1
fi
echo "[OK]   Python: $($PY --version 2>&1)"

# ── 2. 检测 Node.js ──────────────────────────────────────────
if ! command -v node >/dev/null 2>&1; then
    echo "[FAIL] 未检测到 Node.js，请安装 Node.js 20+"
    echo "       下载地址：https://nodejs.org/"
    exit 1
fi
echo "[OK]   Node.js: $(node --version)"

# ── 3. 检测 npm ──────────────────────────────────────────────
if ! command -v npm >/dev/null 2>&1; then
    echo "[FAIL] 未检测到 npm，请随 Node.js 一起安装"
    exit 1
fi
echo "[OK]   npm: $(npm --version)"

# ── 4. 检测 Git（仅警告）─────────────────────────────────────
if command -v git >/dev/null 2>&1; then
    echo "[OK]   Git: $(git --version)"
else
    echo "[WARN] 未检测到 Git，部分功能可能受限"
fi

# ── 5. 运行环境检测 ──────────────────────────────────────────
echo ""
echo "[INFO] 运行环境依赖检测..."
"$PY" "$SCRIPT_DIR/check_env.py"
if [ $? -ne 0 ]; then
    echo ""
    echo "[FAIL] 环境检测未通过，请修复上述问题后重试"
    exit 1
fi

# ── 6. 设置环境变量（幂等，写入 shell rc 文件）──────────────
HANA_HOME_DEFAULT="$HOME/.hana"
LAAP_HOME_DEFAULT="$HOME/.laap"
LAAP_STATE_DEFAULT="$LAAP_HOME_DEFAULT/state"
LAAP_CACHE_DEFAULT="$LAAP_HOME_DEFAULT/cache"
LAAP_LOGS_DEFAULT="$LAAP_HOME_DEFAULT/logs"
ARIS_BRAIN_DEFAULT="$LAAP_HOME_DEFAULT/aris_brain"

# 确定 rc 文件
RC_FILE=""
if [ -n "${ZSH_VERSION:-}" ] || [ "$(basename "${SHELL:-}")" = "zsh" ]; then
    RC_FILE="$HOME/.zshrc"
elif [ -n "${BASH_VERSION:-}" ] || [ "$(basename "${SHELL:-}")" = "bash" ]; then
    RC_FILE="$HOME/.bashrc"
fi

ensure_env_var() {
    # 用法：ensure_env_var <VAR_NAME> <DEFAULT_VALUE>
    local var_name="$1"
    local default_val="$2"
    local current_val="${!var_name:-}"
    if [ -n "$current_val" ]; then
        echo "[OK]   $var_name 已设置：$current_val"
        return 0
    fi
    # 写入 rc 文件（幂等：先检查是否已存在该行）
    if [ -n "$RC_FILE" ]; then
        if ! grep -q "^export ${var_name}=" "$RC_FILE" 2>/dev/null; then
            echo "export ${var_name}=\"${default_val}\"" >> "$RC_FILE"
        fi
    fi
    export "$var_name"="$default_val"
    echo "[INFO] 设置 $var_name = $default_val"
}

ensure_env_var HANA_HOME "$HANA_HOME_DEFAULT"
ensure_env_var LAAP_HOME "$LAAP_HOME_DEFAULT"
ensure_env_var LAAP_ROOT "$LAAP_ROOT"
ensure_env_var LAAP_STATE_DIR "$LAAP_STATE_DEFAULT"
ensure_env_var LAAP_CACHE_DIR "$LAAP_CACHE_DEFAULT"
ensure_env_var LAAP_LOGS_DIR "$LAAP_LOGS_DEFAULT"
ensure_env_var ARIS_BRAIN_DIR "$ARIS_BRAIN_DEFAULT"

# ── 7. 创建数据目录（幂等）──────────────────────────────────
mkdir -p "$HANA_HOME"
mkdir -p "$LAAP_HOME"
mkdir -p "$LAAP_STATE_DIR" "$LAAP_CACHE_DIR" "$LAAP_LOGS_DIR" "$ARIS_BRAIN_DIR"
echo "[OK]   数据目录就绪"

# ── 8. 安装 Python 依赖 ─────────────────────────────────────
echo ""
echo "[INFO] 安装 LAAP Python 依赖..."
# requirements.txt 位于 LAAP 根目录（内部引用 pyproject.toml）
if [ -f "$LAAP_ROOT/requirements.txt" ]; then
    if command -v uv >/dev/null 2>&1; then
        uv pip install --python "$PY" -r "$LAAP_ROOT/requirements.txt"
    else
        "$PY" -m pip install -r "$LAAP_ROOT/requirements.txt"
    fi
    if [ $? -ne 0 ]; then
        echo "[FAIL] Python 依赖安装失败"
        exit 1
    fi
else
    echo "[WARN] 未找到 requirements.txt，跳过 Python 依赖安装"
fi

# ── 9. 安装 Hanako npm 依赖 ─────────────────────────────────
echo ""
echo "[INFO] 安装 Hanako npm 依赖..."
cd "$LAAP_ROOT/hanako" || { echo "[FAIL] hanako 目录不存在"; exit 1; }
npm install
if [ $? -ne 0 ]; then
    echo "[FAIL] npm 依赖安装失败"
    exit 1
fi

# ── 10. 构建桌面端 ───────────────────────────────────────────
echo ""
echo "[INFO] 构建 Hanako 桌面端..."
# package.json 提供 build:client（main + preload + renderer + splash + theme）
npm run build:client
if [ $? -ne 0 ]; then
    echo "[WARN] 桌面端构建失败，可稍后手动运行 npm run build:client"
fi

# ── 11. 初始化数据库 ────────────────────────────────────────
echo ""
echo "[INFO] 初始化 LAAP 记忆库（agent: aris）..."
cd "$LAAP_ROOT" || exit 1
# init_for_agent 是 VaultManager 实例方法，通过模块级单例 vault_manager 调用
"$PY" -c "from laap.memory_vault.vault_manager import vault_manager; p = vault_manager.init_for_agent('aris'); print('vault:', p)"
if [ $? -ne 0 ]; then
    echo "[WARN] 记忆库初始化失败，可稍后手动运行"
fi

# ── 完成 ────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "  安装完成！"
echo ""
echo "  启动方式："
echo "    python installer/launch.py"
echo ""
echo "  仅检测端口："
echo "    python installer/launch.py --check"
echo ""
if [ -n "$RC_FILE" ]; then
    echo "  注意：环境变量已写入 $RC_FILE"
    echo "  请执行 source $RC_FILE 或重开终端使其生效。"
fi
echo "============================================"
