"""LAAP Lifeform CLI — 命令行管理工具 (v2, 精美输出)

用法:
    python -m laap.lifeform.cli create my-lifeform.yaml
    python -m laap.lifeform.cli wake my-lifeform.yaml
    python -m laap.lifeform.cli status state.json
    python -m laap.lifeform.cli list
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from laap.config.paths import get_laap_root
from datetime import datetime


# ── ANSI 颜色 ────────────────────────────────────────────────

class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    PURPLE = "\033[38;5;99m"
    TEAL = "\033[38;5;37m"
    ORANGE = "\033[38;5;208m"
    GRAY = "\033[38;5;244m"
    DARK_GRAY = "\033[38;5;236m"

    @staticmethod
    def rgb(r, g, b): return f"\033[38;2;{r};{g};{b}m"

LAAP_PURPLE = C.rgb(124, 58, 237)  # #7C3AED — LAAP 品牌紫
LAAP_BLUE = C.rgb(59, 130, 246)    # #3B82F6
LAAP_GREEN = C.rgb(16, 185, 129)   # #10B981


# ── 边框工具 ─────────────────────────────────────────────────

def box(title: str, width: int = 56) -> tuple:
    """返回 (top, separator, bottom) 三行字符串"""
    title_safe = title[:width-6]
    padded = f" {title_safe} "
    tl, tr, bl, br = "╭", "╮", "╰", "╯"
    h = "─"
    top = f"{LAAP_PURPLE}{tl}{h}{C.RESET}{C.BOLD}{padded}{C.RESET}{LAAP_PURPLE}{h * (width - len(padded) - 3)}{tr}{C.RESET}"
    sep = f"{LAAP_PURPLE}│{C.RESET}"
    bot = f"{LAAP_PURPLE}{bl}{h * (width - 2)}{br}{C.RESET}"
    return top, sep, bot


def status_icon(ok: bool) -> str:
    return f"{C.GREEN}●{C.RESET}" if ok else f"{C.RED}○{C.RESET}"


def engine_badge(name: str, status: str) -> str:
    colors = {
        "ready": C.GREEN, "error": C.RED,
        "initializing": C.YELLOW, "sleeping": C.GRAY, "uninitialized": C.DARK_GRAY,
    }
    color = colors.get(status, C.GRAY)
    icons = {"ready": "●", "error": "✕", "initializing": "◐", "sleeping": "○", "uninitialized": "·"}
    icon = icons.get(status, "?")
    return f"{color}{icon} {name}{C.RESET}"


def progress_bar(value: float, max_val: float, width: int = 20) -> str:
    ratio = min(value / max_val, 1.0) if max_val > 0 else 0
    filled = int(ratio * width)
    empty = width - filled
    color = LAAP_GREEN if ratio > 0.6 else (C.YELLOW if ratio > 0.3 else C.RED)
    bar = f"{color}{'█' * filled}{C.DARK_GRAY}{'░' * empty}{C.RESET}"
    pct = f"{ratio * 100:.0f}%"
    return f"{bar} {C.GRAY}{pct}{C.RESET}"


def h1(text: str):
    """大型标题"""
    print(f"\n{C.BOLD}{LAAP_PURPLE}═══ {text} {C.RESET}")


def h2(text: str):
    print(f"  {C.BOLD}{LAAP_BLUE}▸ {text}{C.RESET}")


def kv(key: str, value: str, indent: int = 2):
    print(f"{' ' * indent}{C.GRAY}{key}:{C.RESET} {value}")


def json_colorize(data: dict, indent: int = 4) -> str:
    """语法高亮的 JSON 输出"""
    raw = json.dumps(data, indent=2, ensure_ascii=False)
    result = raw
    # 布尔值
    result = result.replace('"true"', f"{C.GREEN}true{C.RESET}")
    result = result.replace('"false"', f"{C.RED}false{C.RESET}")
    result = result.replace('"null"', f"{C.DARK_GRAY}null{C.RESET}")
    # 字符串值
    import re
    result = re.sub(r'": "([^"]+)"', f'": {C.YELLOW}"\\1"{C.RESET}', result)
    result = re.sub(r'": (\d+)', f'": {C.CYAN}\\1{C.RESET}', result)
    # 键
    result = re.sub(r'"([^"]+)":', f'{LAAP_PURPLE}"\\1"{C.RESET}:', result)
    return result


# ── 主入口 ───────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="laap-lifeform",
        description=f"{LAAP_PURPLE}LAAP Lifeform{C.RESET} — 可部署的数字生命体管理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
{C.BOLD}子命令:{C.RESET}
  {C.GREEN}create{C.RESET}  从 YAML 配置创建 Lifeform
  {C.GREEN}wake{C.RESET}    唤醒 Lifeform (初始化引擎)
  {C.GREEN}save{C.RESET}    持久化 Lifeform 状态到 JSON
  {C.GREEN}load{C.RESET}    从 JSON 恢复 Lifeform
  {C.GREEN}status{C.RESET}  查看 Lifeform 详细状态
  {C.GREEN}list{C.RESET}    列出目录下所有 Lifeform

{C.GRAY}示例:{C.RESET}
  python -m laap.lifeform.cli wake example.yaml
  python -m laap.lifeform.cli status state.json
  python -m laap.lifeform.cli list
""",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    create_p = sub.add_parser("create", help="从 YAML 配置创建 Lifeform")
    create_p.add_argument("config", help="YAML 配置路径")
    wp = sub.add_parser("wake", help="唤醒 Lifeform")
    wp.add_argument("config", help="YAML 配置路径")
    wp.add_argument("--save", "-s", help="唤醒后保存到文件")
    sp = sub.add_parser("save", help="持久化 Lifeform 状态")
    sp.add_argument("config", help="YAML 配置路径")
    sp.add_argument("output", help="输出 JSON 路径")
    lp = sub.add_parser("load", help="从 JSON 恢复 Lifeform")
    lp.add_argument("path", help="JSON 状态文件路径")
    stp = sub.add_parser("status", help="查看 Lifeform 详细状态")
    stp.add_argument("path", nargs="?", default=".", help="JSON 路径或目录")

    args = parser.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, str(get_laap_root()))

    from laap.lifeform.lifeform import Lifeform, LifeformConfig

    # ══════════════════════════════════════════════════════════════
    # create
    # ══════════════════════════════════════════════════════════════
    if args.command == "create":
        cfg = LifeformConfig.from_yaml(args.config)
        lf = Lifeform(cfg)
        out = f"{cfg.name.replace(' ', '_')}_lifeform.json"
        lf.save(out)
        size = os.path.getsize(out)
        print()
        t, s, b = box(f" ✦ Lifeform Created ✦ ")
        print(t)
        print(f"{s}  {C.BOLD}{LAAP_PURPLE}{lf.config.name}{C.RESET}")
        print(f"{s}  {C.GRAY}Role:{C.RESET} {lf.config.role}")
        print(f"{s}  {C.GRAY}Saved:{C.RESET} {C.YELLOW}{out}{C.RESET} ({size:,} bytes)")
        print(b)
        print()
        _print_status(lf)

    # ══════════════════════════════════════════════════════════════
    # wake
    # ══════════════════════════════════════════════════════════════
    elif args.command == "wake":
        cfg = LifeformConfig.from_yaml(args.config)
        lf = Lifeform(cfg)
        t0 = time.time()
        ok = lf.wake()
        elapsed = time.time() - t0

        print()
        t, s, b = box(f" ✦ Lifeform Wake: {lf.config.name} ✦ ", width=60)
        print(t)
        print(f"{s}  {status_icon(ok)} {C.BOLD}{lf.config.name}{C.RESET}  {C.GRAY}({lf.config.role}){C.RESET}")
        print(f"{s}  {C.GRAY}Wake time:{C.RESET} {elapsed:.2f}s")
        print(b)
        print()
        _print_status(lf)

        if args.save:
            lf.save(args.save)
            print(f"  {C.GREEN}✓{C.RESET} Saved to {C.YELLOW}{args.save}{C.RESET}\n")

    # ══════════════════════════════════════════════════════════════
    # save
    # ══════════════════════════════════════════════════════════════
    elif args.command == "save":
        cfg = LifeformConfig.from_yaml(args.config)
        lf = Lifeform(cfg)
        lf.wake()
        lf.save(args.output)
        size = os.path.getsize(args.output)
        print(f"\n  {C.GREEN}✦{C.RESET} {C.BOLD}{lf.config.name}{C.RESET} → {C.YELLOW}{args.output}{C.RESET} ({size:,} bytes)\n")

    # ══════════════════════════════════════════════════════════════
    # load
    # ══════════════════════════════════════════════════════════════
    elif args.command == "load":
        lf = Lifeform.load(args.path)
        print()
        t, s, b = box(f" ✦ Lifeform Restored ✦ ")
        print(t)
        print(f"{s}  {C.BOLD}{LAAP_PURPLE}{lf.config.name}{C.RESET}")
        print(f"{s}  {C.GRAY}Role:{C.RESET} {lf.config.role}")
        print(f"{s}  {C.GRAY}ID:{C.RESET}   {lf.sandbox_id}")
        print(f"{s}  {C.GRAY}From:{C.RESET} {C.YELLOW}{args.path}{C.RESET}")
        print(b)
        print()
        _print_status(lf)

    # ══════════════════════════════════════════════════════════════
    # status
    # ══════════════════════════════════════════════════════════════
    elif args.command == "status":
        path = args.path
        if os.path.isdir(path):
            # 列出目录下所有 Lifeform
            print()
            t, s, b = box(f" Lifeforms in {path} ", width=68)
            print(t)
            files = [f for f in os.listdir(path) if f.endswith(".json") and "lifeform" in f]
            if not files:
                print(f"{s}  {C.GRAY}(no lifeform files found){C.RESET}")
            else:
                for f in sorted(files):
                    fp = os.path.join(path, f)
                    try:
                        lf = Lifeform.load(fp)
                        st = lf.status()
                        size = os.path.getsize(fp)
                        engines_ok = st.get("engines_ready", "0/0")
                        print(f"{s}  {status_icon(True)} {C.BOLD}{st['name']:20}{C.RESET}  {C.GRAY}{st['role']:12}{C.RESET}  engines={engines_ok:>5}  {C.DARK_GRAY}{size:>6,}B{C.RESET}")
                    except Exception:
                        print(f"{s}  {C.RED}✕{C.RESET} {C.GRAY}{f}{C.RESET}  (corrupted)")
            print(b)
            print()
        else:
            lf = Lifeform.load(path)
            print()
            _print_status(lf)


def _print_status(lf):
    """打印 Lifeform 状态 (精美格式)"""
    s = lf.status()
    from laap.lifeform.lifeform import LifeformConfig

    # ── 头部面板 ──
    w = 60
    t, sep, b = box(f" {s['name']} ", w)
    print(t)
    print(f"{sep}  {C.GRAY}ID:{C.RESET}     {C.DIM}{s['id']}{C.RESET}")
    print(f"{sep}  {C.GRAY}Role:{C.RESET}   {C.BOLD}{s['role']}{C.RESET}")
    created = datetime.fromtimestamp(s['created_at']).strftime("%Y-%m-%d %H:%M")
    updated = datetime.fromtimestamp(s['updated_at']).strftime("%H:%M:%S")
    print(f"{sep}  {C.GRAY}Created:{C.RESET} {created}  {C.GRAY}Updated:{C.RESET} {updated}")
    print(b)
    print()

    # ── 引擎状态面板 ──
    t2, sep2, b2 = box(" Engine Status ", w)
    print(t2)
    engine_status = s.get("engine_status", {})
    for ename, estatus in sorted(engine_status.items()):
        badge = engine_badge(ename, estatus)
        print(f"{sep2}    {badge}")
    print(f"{sep2}  {C.DIM}Total: {len(engine_status)} engines, {s.get('engines_ready', '?')} ready{C.RESET}")
    print(b2)
    print()

    # ── 需求面板 ──
    needs = s.get("needs", {})
    if needs:
        t3, sep3, b3 = box(" PSI Needs ", w)
        print(t3)
        for nname, nvalue in sorted(needs.items()):
            bar = progress_bar(nvalue, 1.0)
            label = f"{nname.replace('_', ' ').title():20}"
            print(f"{sep3}  {C.GRAY}{label}{C.RESET} {bar}")
        print(b3)
        print()

    # ── 目标面板 ──
    goals_count = s.get("goals", 0)
    t4, sep4, b4 = box(" Goals ", w)
    print(t4)
    if goals_count > 0:
        print(f"{sep4}  {C.YELLOW}{goals_count} active goals{C.RESET}")
    else:
        print(f"{sep4}  {C.GRAY}No active goals{C.RESET}")
    print(b4)
    print()

    # ── 配置摘要 ──
    config = lf.config
    t5, sep5, b5 = box(" Config Summary ", w)
    print(t5)
    eng = config.engines
    enabled = []
    if eng.psi: enabled.append("PSI")
    if eng.qre: enabled.append("QRE")
    if eng.causal: enabled.append("Causal")
    if eng.world_model and eng.world_model != "none": enabled.append(f"World({eng.world_model})")
    if eng.conscious: enabled.append("Conscious")
    if eng.memory: enabled.append("Memory")
    print(f"{sep5}  {C.GRAY}Engines:{C.RESET} {', '.join(enabled) if enabled else C.DIM + 'none' + C.RESET}")
    print(f"{sep5}  {C.GRAY}Personality:{C.RESET} {C.DIM}O:{config.personality.get('openness', '?'):.1f} C:{config.personality.get('conscientiousness', '?'):.1f} E:{config.personality.get('extraversion', '?'):.1f} A:{config.personality.get('agreeableness', '?'):.1f} N:{config.personality.get('neuroticism', '?'):.1f}{C.RESET}")
    gov = config.governance
    print(f"{sep5}  {C.GRAY}Governance:{C.RESET} oversight={gov.human_oversight} audit={gov.audit_level}")
    lc = config.language_cortex
    if lc.provider:
        print(f"{sep5}  {C.GRAY}Language:{C.RESET} {lc.provider}/{lc.model}")
    print(b5)
    print()


if __name__ == "__main__":
    main()
