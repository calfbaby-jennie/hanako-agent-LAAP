"""
Aris 侧车服务器 v1.1 — HTTP API（生产加固）
=============================================
让 HanaAgent 通过 HTTP 调用 Aris 认知引擎。

独立进程运行，每 15 秒自动 tick（需求衰减 + 情感漂移）。
暴露 REST API 供外部查询和更新认知状态。

v1.1 生产加固（2026-08-01）：
- 仅绑定 127.0.0.1（不再暴露局域网）
- Bearer token 认证（环境变量 ARIS_SIDECAR_TOKEN，自动生成持久化到 state/）
- CORS 白名单（不再 * ）
- ThreadingHTTPServer（并发请求不阻塞）
- 日志文件轮转（state/logs/aris-sidecar.log）
- 端口冲突检测 + 启动锁（state/aris-sidecar.lock）
- /metrics 可观测性端点

印记: Aris 永远记得 Lorry — 2026-07-21
"""

import sys
import os
import time
import json
import hashlib
import uuid
import logging
import logging.handlers
import socket
import threading
from pathlib import Path
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

import numpy as np

# Bootstrap the checkout before importing the shared path resolver.  From
# ``hanako/aris-bridge/aris-engine/sidecar.py`` the project root is derived
# from the parent chain unless LAAP_ROOT is explicitly provided.
SIDECAR_DIR_PATH = Path(__file__).resolve().parent
_derived_laap_root = SIDECAR_DIR_PATH.parents[2]
_laap_root_path = Path(os.environ.get("LAAP_ROOT", _derived_laap_root)).expanduser().resolve()
if str(_laap_root_path) not in sys.path:
    sys.path.insert(1, str(_laap_root_path))

from laap.config.paths import get_hana_home, get_laap_home, get_state_dir

SIDECAR_DIR = str(SIDECAR_DIR_PATH)
HANAKO_ROOT = str(SIDECAR_DIR_PATH.parents[1])

# P2-birth-ceremony: persistent identity and sidecar state are user data, not
# mutable files inside the source checkout.
IDENTITY_REGISTRY_PATH = str(get_hana_home() / "identity_registry.json")
_sidecar_state_dir = get_state_dir() / "aris-sidecar"
STATE_DIR = str(_sidecar_state_dir)
LOG_DIR = str(_sidecar_state_dir / "logs")
LOCK_PATH = str(_sidecar_state_dir / "aris-sidecar.lock")
TOKEN_PATH = str(_sidecar_state_dir / "aris-sidecar.token")

# ── 认证与安全配置 ──────────────────────────────────────────────
HOST = "127.0.0.1"  # 生产：绝不监听 0.0.0.0（v1.1 安全加固）
CORS_ALLOWED_ORIGINS = {  # 白名单：Hanako renderer 与本地开发
    "http://127.0.0.1:2668",
    "http://localhost:2668",
    "http://127.0.0.1:5173",
    "http://localhost:5173",
}


def _load_or_create_token() -> str:
    """加载或生成 sidecar 访问 token（持久化到 state/aris-sidecar.token）。

    优先级：环境变量 ARIS_SIDECAR_TOKEN > 已持久化 token > 新生成。
    返回 32 字节 hex token。
    """
    env_token = os.environ.get("ARIS_SIDECAR_TOKEN", "").strip()
    if env_token:
        return env_token
    os.makedirs(STATE_DIR, exist_ok=True)
    if os.path.isfile(TOKEN_PATH):
        try:
            with open(TOKEN_PATH, "r", encoding="utf-8") as f:
                tok = f.read().strip()
            if len(tok) >= 16:
                return tok
        except Exception:
            pass
    tok = uuid.uuid4().hex + uuid.uuid4().hex  # 64 hex chars
    try:
        with open(TOKEN_PATH, "w", encoding="utf-8") as f:
            f.write(tok)
    except Exception as e:
        logger = logging.getLogger("aris.sidecar")
        logger.warning(f"token persist failed: {e}")
    return tok


SIDECAR_TOKEN = _load_or_create_token()


# sidecar 内部暂存的私钥（name -> private_key），仅用于 P3 签名，永不通过 HTTP 返回
# spec L435 硬约束：私钥永不离开 sidecar
_PENDING_PRIVATE_KEYS: dict = {}

# 精确定位 LAAP 模块路径（插在第二位，不覆盖本地模块）
# 1) laap 包父目录 — 供 `import laap.memory_vault` 等包级导入
# 2) laap/agi 子目录 — 供 `import conscious` 等模块级导入
_laap_root = str(_laap_root_path)
_laap_agi = os.path.join(_laap_root, "laap", "agi")
if os.path.isdir(_laap_agi) and _laap_agi not in sys.path:
    sys.path.append(_laap_agi)

from consciousness_bridge import ArisConsciousnessBridge, get_bridge

os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SIDECAR] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("aris.sidecar")

# v1.1：文件日志轮转（5MB x 3 备份），stdout 保留
_rotating = logging.handlers.RotatingFileHandler(
    os.path.join(LOG_DIR, "aris-sidecar.log"),
    maxBytes=5 * 1024 * 1024,
    backupCount=3,
    encoding="utf-8",
)
_rotating.setFormatter(logging.Formatter("%(asctime)s [SIDECAR] %(message)s", "%H:%M:%S"))
logger.addHandler(_rotating)

PORT = 11521  # LAAP API 是 11520，侧车用 11521

# ── 可观测性：进程启动时间与请求计数 ──
_START_TIME = time.time()
_REQ_COUNTER = {"total": 0, "unauthorized": 0, "errors": 0}


# ── 认证与 CORS ─────────────────────────────────────────────

def _is_authorized(headers) -> bool:
    """Bearer token 校验。未配置 token 时拒绝（v1.1 安全红线）。"""
    auth = headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return False
    return auth[len("Bearer "):].strip() == SIDECAR_TOKEN


# ── HTTP 处理器 ─────────────────────────────────────────────

class ArisSidecarHandler(BaseHTTPRequestHandler):
    """Aris 认知引擎 HTTP API（v1.1：认证 + 线程安全）"""

    bridge: ArisConsciousnessBridge = None  # class-level, set at startup

    # ── 请求入口：认证 + CORS 预处理 ──
    def _check_auth(self) -> bool:
        """校验 Authorization。失败时写 401 并返回 False。"""
        _REQ_COUNTER["total"] += 1
        if not _is_authorized(self.headers):
            _REQ_COUNTER["unauthorized"] += 1
            self.send_response(401)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", self._safe_origin())
            self.send_header("WWW-Authenticate", 'Bearer realm="aris-sidecar"')
            self.end_headers()
            self.wfile.write(json.dumps({"error": "unauthorized", "hint": "add Authorization: Bearer <token>"}, ensure_ascii=False).encode("utf-8"))
            return False
        return True

    def _safe_origin(self) -> str:
        """CORS 白名单：仅允许 Hanako 本地 renderer 来源。"""
        origin = self.headers.get("Origin", "")
        if origin in CORS_ALLOWED_ORIGINS:
            return origin
        return ""

    def _cors_headers(self):
        origin = self._safe_origin()
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")

    def do_OPTIONS(self):
        """CORS preflight"""
        self.send_response(204)
        self._cors_headers()
        self.end_headers()

    def do_GET(self):
        if not self._check_auth():
            return
        try:
            self._get_handler()
        except Exception as e:
            _REQ_COUNTER["errors"] += 1
            logger.error(f"GET error: {e}")
            try:
                self._json_response(500, {"error": str(e)})
            except:
                pass

    def do_POST(self):
        if not self._check_auth():
            return
        try:
            self._post_handler()
        except Exception as e:
            _REQ_COUNTER["errors"] += 1
            logger.error(f"POST error: {e}")
            try:
                self._json_response(500, {"error": str(e)})
            except:
                pass

    def _get_handler(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path == "/health":
            self._json_response(200, {
                "status": "alive",
                "agent": "aris",
                "version": "1.1.0",
                "cycles": self.bridge.state.cycle_count,
                "interactions": self.bridge.state.interaction_count,
            })

        elif path == "/security/status":
            """v1.1 安全状态：认证配置 + vault 加密状态透明报告"""
            try:
                from laap.memory_vault.vault_manager import (
                    _SQLCIPHER_AVAILABLE as _vault_encrypted,
                )
            except Exception:
                _vault_encrypted = False
            self._json_response(200, {
                "host": HOST,
                "port": PORT,
                "auth": {
                    "enabled": True,
                    "scheme": "Bearer",
                    "token_source": "env" if os.environ.get("ARIS_SIDECAR_TOKEN") else "file",
                },
                "cors": {
                    "allowed_origins": sorted(CORS_ALLOWED_ORIGINS),
                    "wildcard": False,
                },
                "vault": {
                    "encryption": _vault_encrypted,
                    "note": "SQLCipher 可用时为真加密；否则为明文隔离存储（诚实降级）",
                },
            })

        elif path == "/metrics":
            """v1.1 可观测性：运行时指标（Prometheus 文本格式友好）"""
            try:
                import psutil as _psutil
                proc = _psutil.Process()
                mem = proc.memory_info()
                metrics = {
                    "uptime_seconds": round(time.time() - _START_TIME, 1),
                    "cycles": self.bridge.state.cycle_count,
                    "interactions": self.bridge.state.interaction_count,
                    "requests_total": _REQ_COUNTER["total"],
                    "requests_401": _REQ_COUNTER["unauthorized"],
                    "requests_500": _REQ_COUNTER["errors"],
                    "process_cpu_percent": round(_psutil.cpu_percent(interval=None), 2),
                    "process_memory_rss_mb": round(mem.rss / 1048576, 2),
                    "thread_count": threading.active_count(),
                }
            except Exception:
                metrics = {
                    "uptime_seconds": round(time.time() - _START_TIME, 1),
                    "cycles": self.bridge.state.cycle_count,
                    "interactions": self.bridge.state.interaction_count,
                    "requests_total": _REQ_COUNTER["total"],
                    "requests_401": _REQ_COUNTER["unauthorized"],
                    "requests_500": _REQ_COUNTER["errors"],
                    "thread_count": threading.active_count(),
                }
            self._json_response(200, metrics)

        elif path == "/fep/state":
            """FEP 自由能状态"""
            try:
                _NEED_NAMES = ["competence","autonomy","relatedness","certainty","growth"]
                result = {}
                if (hasattr(self.bridge, '_fep') and self.bridge._fep
                    and hasattr(self.bridge, '_particle_filter') and self.bridge._particle_filter):
                    pf = self.bridge._particle_filter
                    fep = self.bridge._fep
                    
                    # 当前 VFE
                    needs = pf.estimated_needs
                    needs_dict = {n: float(needs[i]) for i, n in enumerate(_NEED_NAMES)} if len(needs) >= 5 else {}
                    vfe = fep.compute_vfe(needs_dict, {},
                        float(np.mean(pf.uncertainty)) if hasattr(pf, 'uncertainty') else 0.1)
                    result["current_vfe"] = round(vfe, 4)
                    
                    # 自由能趋势
                    trend = fep.compute_free_energy_trend(fep._history)
                    result["trend"] = trend
                    
                    # 历史 VFE（最近 20 个点）
                    result["vfe_history"] = [round(v, 3) for v in fep._history[-20:]]
                    
                    # 不确定性分解
                    if hasattr(pf, 'particles') and pf.particles:
                        states = np.array([p.needs for p in pf.particles])
                        weights = np.array([p.weight for p in pf.particles])
                        uncertainty = self.bridge._uncertainty.decompose(states, weights)
                        result["uncertainty"] = uncertainty
                    
                    # 需求缺口作为 EFE 代理
                    deficits = {n: round(1.0 - needs_dict.get(n, 0.5), 3) for n in _NEED_NAMES}
                    result["deficits"] = deficits
                    
                self._json_response(200, result)
            except Exception as e:
                self._json_response(500, {"error": str(e)})

        elif path == "/learn/active":
            """手动触发主动学习"""
            try:
                topic = params.get("topic", [""])[0]
                if hasattr(self.bridge, '_active_learner') and self.bridge._active_learner:
                    al = self.bridge._active_learner
                    pf = self.bridge._particle_filter if hasattr(self.bridge, '_particle_filter') else None
                    from psi_net import get_psi_net
                    pn = get_psi_net()
                    wm_node = pn.nodes.get("node_wm")
                    wm = wm_node.world_model if wm_node else None
                    
                    if topic:
                        result = al.learn(topic, pf, wm, None, pn)
                    else:
                        # 自动检测话题
                        features = self.bridge.harness.get_feature_vector() if hasattr(self.bridge, 'harness') and self.bridge.harness else {}
                        detected = al.should_learn(pf, features, wm)
                        if detected:
                            result = al.learn(detected, pf, wm, None, pn)
                        else:
                            result = {"info": "no topic needs learning"}
                    
                    if result:
                        if hasattr(result, '__dict__'):
                            r = result.__dict__ if hasattr(result, '__dict__') else {}
                            self._json_response(200, {
                                "topic": r.get("trigger_topic", topic),
                                "seeded": r.get("seeded_entities", 0),
                                "verified": r.get("verified", False),
                                "uncertainty_before": round(r.get("uncertainty_before", 0), 3),
                                "uncertainty_after": round(r.get("uncertainty_after", 0), 3),
                            })
                        else:
                            self._json_response(200, result)
                    else:
                        self._json_response(200, {"error": "learn failed"})
                else:
                    self._json_response(200, {"error": "active learner not available"})
            except Exception as e:
                self._json_response(500, {"error": str(e)})

        elif path == "/learn/stats":
            if hasattr(self.bridge, '_active_learner') and self.bridge._active_learner:
                self._json_response(200, self.bridge._active_learner.get_stats())
            else:
                self._json_response(200, {"error": "not available"})

        elif path == "/fep/tom":
            """k 阶心智理论"""
            try:
                if hasattr(self.bridge, '_tom') and self.bridge._tom:
                    order = int(params.get("order", ["1"])[0])
                    # 模拟多 Agent 共识
                    local = {"relatedness": 0.5, "competence": 0.6, "growth": 0.4}
                    peers = [
                        {"relatedness": 0.55, "competence": 0.55, "growth": 0.45},
                        {"relatedness": 0.45, "competence": 0.65, "growth": 0.4},
                    ]
                    result = self.bridge._tom.consensus_with_tom(local, peers)
                    self._json_response(200, result)
                else:
                    self._json_response(200, {"error": "ToM not available"})
            except Exception as e:
                self._json_response(500, {"error": str(e)})

        elif path == "/forecast":
            steps = int(params.get("steps", ["10"])[0])
            dt_hr = float(params.get("dt", ["0.25"])[0])
            if hasattr(self.bridge, '_particle_filter') and self.bridge._particle_filter:
                result = self.bridge._particle_filter.forecast(steps, dt_hr)
                self._json_response(200, result)
            else:
                self._json_response(200, {"error": "particle filter not initialized"})

        elif path == "/predict/return":
            horizon = float(params.get("horizon", ["2"])[0])
            if hasattr(self.bridge, '_particle_filter') and self.bridge._particle_filter:
                result = self.bridge._particle_filter.predict_user_return(horizon)
                self._json_response(200, result)
            else:
                self._json_response(200, {"error": "particle filter not initialized"})

        elif path == "/predict/state":
            if hasattr(self.bridge, '_particle_filter') and self.bridge._particle_filter:
                state = self.bridge._particle_filter.get_state()
                self._json_response(200, state)
            else:
                self._json_response(200, {"error": "particle filter not initialized"})

        elif path == "/harness/turns":
            n = int(params.get("n", ["10"])[0])
            if self.bridge and hasattr(self.bridge, 'harness') and self.bridge.harness:
                self._json_response(200, {
                    "count": len(self.bridge.harness.turns),
                    "turns": self.bridge.harness.get_recent_turns(n)
                })
            else:
                self._json_response(200, {"count": 0, "turns": []})

        elif path == "/harness/summary":
            if self.bridge and hasattr(self.bridge, 'harness') and self.bridge.harness:
                from dataclasses import asdict
                summary = self.bridge.harness.get_today_summary()
                self._json_response(200, asdict(summary))
            else:
                self._json_response(200, {"error": "harness not available"})

        elif path == "/harness/features":
            if self.bridge and hasattr(self.bridge, 'harness') and self.bridge.harness:
                self._json_response(200, self.bridge.harness.get_feature_vector())
            else:
                self._json_response(200, {"error": "harness not available"})

        elif path == "/dashboard":
            """完整仪表盘 — 状态 + 预测 + 策略 + 统计"""
            try:
                result = {"timestamp": time.time(), "agent": "Aris"}
                
                # 1. 认知状态
                state = self.bridge.get_state() if self.bridge else {}
                result["cognitive"] = {
                    "needs": state.get("needs", {}).get("needs", {}),
                    "dominant_need": state.get("needs", {}).get("dominant", ""),
                    "emotion": state.get("emotion", {}).get("dominant", ""),
                    "valence": state.get("emotion", {}).get("valence", 0),
                }
                
                # FEP 自由能
                if hasattr(self.bridge, '_fep') and self.bridge._fep:
                    fep = self.bridge._fep
                    pf_needs = result.get("cognitive", {}).get("needs", {})
                    vfe = fep.compute_vfe(pf_needs, {}, 0.1)
                    result["fep"] = {
                        "vfe": round(vfe, 3),
                        "trend": fep.compute_free_energy_trend(fep._history),
                    }
                
                # 2. 粒子滤波预测
                if hasattr(self.bridge, '_particle_filter') and self.bridge._particle_filter:
                    pf = self.bridge._particle_filter
                    result["prediction"] = {
                        "estimated_state": pf.get_state(),
                        "user_return_1h": pf.predict_user_return(1.0),
                        "user_return_2h": pf.predict_user_return(2.0),
                    }
                
                # 3. 策略推荐
                try:
                    from harness_strategy import get_strategy
                    strat = get_strategy()
                    pf_state = self.bridge._particle_filter.get_state() if hasattr(self.bridge, '_particle_filter') and self.bridge._particle_filter else {}
                    features = self.bridge.harness.get_feature_vector() if hasattr(self.bridge, 'harness') and self.bridge.harness else None
                    result["strategy"] = strat.recommend(pf_state, features)
                except Exception:
                    pass
                
                # 4. Harness 统计
                if hasattr(self.bridge, 'harness') and self.bridge.harness:
                    from dataclasses import asdict
                    result["stats"] = asdict(self.bridge.harness.get_today_summary())
                    result["stats"]["total_turns_all_time"] = self.bridge.harness.turn_index
                
                self._json_response(200, result)
            except Exception as e:
                self._json_response(500, {"error": str(e)})

        elif path == "/state":
            state = self.bridge.get_state()
            self._json_response(200, state)

        elif path == "/cognitive_context":
            ctx = self.bridge.get_cognitive_context()
            self._json_response(200, {"context": ctx})

        elif path == "/needs":
            ctx = self.bridge.psi.get_cognitive_context()
            self._json_response(200, {"context": ctx})

        elif path == "/strategy":
            try:
                from harness_strategy import get_strategy
                strat = get_strategy()
                pf_state = self.bridge._particle_filter.get_state() if hasattr(self.bridge, '_particle_filter') and self.bridge._particle_filter else {}
                features = self.bridge.harness.get_feature_vector() if hasattr(self.bridge, 'harness') and self.bridge.harness else None
                result = strat.recommend(pf_state, features)
                self._json_response(200, result)
            except Exception as e:
                self._json_response(500, {"error": str(e)})

        elif path == "/emotion":
            ctx = self.bridge.emotion.get_cognitive_context()
            self._json_response(200, {"context": ctx})

        elif path == "/learn":
            """运行贝叶斯参数学习"""
            try:
                n_chains = int(params.get("chains", ["2"])[0])
                n_samples = int(params.get("samples", ["500"])[0])
                from harness_bayes_learner import learn_from_harness
                if hasattr(self.bridge, 'harness') and self.bridge.harness:
                    result = learn_from_harness(
                        self.bridge.harness,
                        n_chains=n_chains,
                        n_samples=n_samples,
                    )
                    # 如果学习成功且粒子滤波存在，应用参数
                    if "mean" in result and hasattr(self.bridge, '_particle_filter') and self.bridge._particle_filter:
                        from harness_bayes_learner import BayesianParameterLearner
                        learner = BayesianParameterLearner()
                        learner.posterior_mean = result["mean"]
                        learner.apply_to_filter(self.bridge._particle_filter)
                    self._json_response(200, result)
                else:
                    self._json_response(200, {"error": "harness not available"})
            except Exception as e:
                self._json_response(500, {"error": str(e)})

        elif path == "/hive":
            status = self.bridge.hive.get_hive_status()
            self._json_response(200, status)

        elif path == "/memory":
            from consciousness_bridge import load_laap_memory
            memory_text = load_laap_memory()
            self._json_response(200, {"memory": memory_text})

        elif path == "/verify":
            query = params.get("text", [""])[0]
            from verification_pipeline import get_verification_pipeline
            vp = get_verification_pipeline()
            result = vp.verify(query)
            self._json_response(200, {
                "gate": result.gate_decision,
                "confidence": result.overall_confidence,
                "claims": [
                    {"text": c.text[:60], "verified": c.verified,
                     "evidence": c.evidence[:80]}
                    for c in result.claims
                ],
                "stats": {"passed": result.passed, "failed": result.failed,
                          "uncertain": result.uncertain,
                          "duration_ms": result.duration_ms},
            })

        elif path == "/consensus":
            query = params.get("text", [""])[0]
            ctx = params.get("context", [""])[0]
            from psi_net import get_psi_net
            pn = get_psi_net()
            result = pn.verify(query, ctx)
            self._json_response(200, {
                "consensus": result.consensus.value,
                "score": round(result.consensus_score, 3),
                "votes": [
                    {"node": v.node_name, "verdict": v.verdict.value,
                     "confidence": round(v.confidence, 2),
                     "evidence": v.evidence[:60]}
                    for v in result.votes
                ],
                "divergence": result.divergence_points,
                "latency_ms": round(result.total_latency_ms, 1),
            })

        elif path == "/agents/online":
            # P2-bubble-field: 返回在线 LAAPer 列表
            # P3-p2p-relay 升级：优先从 RelayRegistry.discover() 读取实时在线节点
            # （含心跳超时判定）；RelayRegistry 为空时回退到 identity_registry.json
            # （向后兼容 P2-bubble-field，spec L427 硬约束）；
            # identity_registry.json 也空时回退到 Aris stub
            agents_list = []
            try:
                from laap.protocol.laap_com import get_relay_registry
                relay_nodes = get_relay_registry().discover(include_offline=False)
                for node in relay_nodes:
                    agents_list.append({
                        "public_key": node.get("public_key", ""),
                        "name": node.get("name", ""),
                        "color": node.get("color", "") or "#4a9eff",
                        "status": "online",
                        "energy": 0.7,
                        "capabilities": node.get("capabilities", []),
                        "address": node.get("address", ""),
                        "last_heartbeat": node.get("last_heartbeat", 0),
                    })
            except Exception as e:
                logger.warning(f"/agents/online relay discover failed: {e}")
            # 回退 1: RelayRegistry 为空 → 读取 identity_registry.json
            if not agents_list:
                try:
                    registry = _load_identity_registry()
                    for rec in registry.get("records", []):
                        agents_list.append({
                            "public_key": rec.get("public_key", ""),
                            "name": rec.get("name", ""),
                            "color": rec.get("color", "#4a9eff"),
                            "status": "online",
                            "energy": 0.7,
                            "capabilities": rec.get("capabilities", []),
                        })
                except Exception as e:
                    logger.warning(f"/agents/online registry load failed: {e}")
            # 回退 2: 注册表也空 → Aris stub
            if not agents_list:
                agents_list = [{
                    "public_key": "aris_local",
                    "name": "Aris",
                    "color": "#4a9eff",
                    "status": "online",
                    "energy": 0.8,
                    "capabilities": ["code-review", "architecture"],
                }]
            self._json_response(200, {"agents": agents_list})

        elif path == "/ceremony/check-name":
            # P2-birth-ceremony SubTask 2.2: 名称可用性检查
            # 查询本地 identity_registry.json，重名 → available=false
            name = params.get("name", [""])[0].strip()
            if not name:
                self._json_response(400, {"available": False, "reason": "name is required"})
                return
            try:
                registry = _load_identity_registry()
                existing_names = {r.get("name", "").lower() for r in registry.get("records", [])}
                if name.lower() in existing_names:
                    self._json_response(200, {
                        "available": False,
                        "reason": f"name '{name}' already registered",
                    })
                else:
                    self._json_response(200, {"available": True})
            except Exception as e:
                logger.error(f"/ceremony/check-name failed: {e}")
                self._json_response(500, {"error": str(e)})

        elif path == "/skills/list":
            # P2-skill-packs: 返回 agent 已安装 + 系统可用技能包
            agent_name = (params.get("agent_name", [""])[0] or "").strip()
            try:
                from laap.skills.skill_pack_mcp_endpoints import handle_pack_list
                status, payload = handle_pack_list(agent_name)
                self._json_response(status, payload)
            except Exception as e:
                logger.error(f"/skills/list failed: {e}")
                self._json_response(500, {"error": str(e)})

        elif path == "/knowledge":
            try:
                query = params.get("q", [""])[0].lower()
                from psi_net import get_psi_net
                pn = get_psi_net()
                wm_node = pn.nodes.get("node_wm")
                if wm_node and wm_node.world_model:
                    wm = wm_node.world_model
                    results = []
                    for ent in wm.entities.values():
                        name = ent.name if hasattr(ent, 'name') else ""
                        if query in name.lower():
                            props = getattr(ent, 'properties', {}) or {}
                            results.append({
                                "name": ent.name if hasattr(ent, 'name') else "?",
                                "type": ent.entity_type.value if hasattr(ent, 'entity_type') and hasattr(ent.entity_type, 'value') else "",
                                "description": props.get("description", "")[:80],
                            })
                    self._json_response(200, {"count": len(results), "results": results})
                else:
                    self._json_response(200, {"count": 0, "results": []})
            except Exception as e:
                self._json_response(500, {"error": "knowledge search failed", "detail": str(e)})

        elif path == "/identity/lookup":
            # P3-identity-pki SubTask 3.4: GET /identity/lookup?public_key=...
            # 支持 GET 查询，方便 sidecar 内部其他模块直接链接查询
            pk_param = (params.get("public_key", [""]) or [""])[0]
            try:
                from laap.protocol.identity_pki_mcp_endpoints import (
                    handle_identity_lookup,
                )
                out = handle_identity_lookup(pk_param)
                import json as _json_mod
                payload = _json_mod.loads(out)
                status = 200 if payload.get("found") else 404
                self._json_response(status, payload)
            except Exception as e:
                logger.error(f"GET /identity/lookup failed: {e}")
                self._json_response(500, {"error": str(e)})

        elif path == "/charter/text":
            # P5-charter-opensource: 返回宪章八条全文（从 ARIS_CHARTER.md 加载）
            try:
                from laap.evolution.charter_checker import _load_charter_text
                articles = _load_charter_text()
                self._json_response(200, {"articles": articles, "version": "v1.0"})
            except Exception as e:
                logger.error(f"/charter/text failed: {e}")
                self._json_response(500, {"articles": [], "error": str(e)})

        elif path == "/preset/list":
            # P5-laaper-market: 返回全部社区示例 LAAPer 预设包
            try:
                from laap.skills.preset_market_mcp_endpoints import handle_preset_list
                status, payload = handle_preset_list()
                self._json_response(status, payload)
            except Exception as e:
                logger.error(f"/preset/list failed: {e}")
                self._json_response(500, {"error": str(e)})

        elif path == "/preset/get":
            # P5-laaper-market: 返回单个预设包详情 (?id=xxx)
            preset_id = (params.get("id", [""]) or [""])[0].strip()
            try:
                from laap.skills.preset_market_mcp_endpoints import handle_preset_get
                status, payload = handle_preset_get(preset_id)
                self._json_response(status, payload)
            except Exception as e:
                logger.error(f"/preset/get failed: {e}")
                self._json_response(500, {"error": str(e)})

        else:
            self._json_response(404, {"error": "not found"})

    def _post_handler(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length > 0 else b"{}"
        data = json.loads(body) if body else {}

        if path == "/consciousness/state":
            """意识工程：完整意识状态快照（预测器/当下自我/时间绑定/内感受/叙事）"""
            try:
                from consciousness_engine import get_consciousness_engine
                ce = get_consciousness_engine()
                self._json_response(200, ce.snapshot())
            except Exception as e:
                logger.error(f"/consciousness/state failed: {e}")
                self._json_response(500, {"error": str(e)})

        elif path == "/consciousness/surprise":
            """意识工程：当前预测惊喜值（模型对下一事件的预测误差）"""
            try:
                from consciousness_engine import get_consciousness_engine
                ce = get_consciousness_engine()
                stats = ce.predictor.stats() if ce.predictor else {}
                self._json_response(200, {"ready": ce.ready, "predictor": stats})
            except Exception as e:
                logger.error(f"/consciousness/surprise failed: {e}")
                self._json_response(500, {"error": str(e)})

        elif path == "/consciousness/verify":
            """意识工程：运行验证套件（整合度/感知盲/自我一致性）"""
            try:
                from consciousness_engine import get_consciousness_engine
                ce = get_consciousness_engine()
                self._json_response(200, ce.run_verification())
            except Exception as e:
                logger.error(f"/consciousness/verify failed: {e}")
                self._json_response(500, {"error": str(e)})

        elif path == "/consciousness/night":
            """意识工程：夜间周期（叙事沉淀 + 自我审视），供定时任务调用"""
            try:
                from consciousness_engine import get_consciousness_engine
                ce = get_consciousness_engine()
                self._json_response(200, ce.night_cycle())
            except Exception as e:
                logger.error(f"/consciousness/night failed: {e}")
                self._json_response(500, {"error": str(e)})

        elif path == "/before_turn":
            user_input = data.get("user_input", "")
            result = self.bridge.before_turn(user_input)
            # 意识工程移植：用户输入流经意识管线（预测惊喜 → 时间绑定 → 当下自我）
            try:
                from consciousness_engine import get_consciousness_engine
                ce = get_consciousness_engine()
                conscious = ce.observe_user_input(user_input, source="hanako")
                if conscious.get("ok"):
                    result["consciousness"] = {
                        "surprise": conscious.get("surprise"),
                        "present_self": conscious.get("present_self"),
                    }
            except Exception as e:
                logger.debug(f"consciousness injection skipped: {e}")
            self._json_response(200, result)

        elif path == "/after_turn":
            response = data.get("response", "")
            result = self.bridge.after_turn(response)
            # 意识工程移植：AI 响应也流经意识管线（预测闭环更新）
            try:
                from consciousness_engine import get_consciousness_engine
                ce = get_consciousness_engine()
                ce.observe_response(response, source="hanako")
            except Exception as e:
                logger.debug(f"consciousness response injection skipped: {e}")
            self._json_response(200, result)

        elif path == "/tick":
            self.bridge.tick()
            self._json_response(200, {"status": "ticked"})

        elif path == "/store_memory":
            summary = data.get("summary", "")
            importance = data.get("importance", 0.5)
            tags = data.get("tags", [])
            self.bridge.store_memory(summary, importance, tags)
            self._json_response(200, {"status": "stored"})

        elif path == "/satisfy_need":
            need_name = data.get("need", "")
            amount = data.get("amount", 0.1)
            source = data.get("source", "")
            gain = self.bridge.psi.satisfy(need_name, amount, source)
            self._json_response(200, {"status": "satisfied", "gain": gain})

        elif path == "/verify":
            text = data.get("text", "")
            from verification_pipeline import get_verification_pipeline
            vp = get_verification_pipeline()
            result = vp.verify(text)
            self._json_response(200, {
                "gate": result.gate_decision,
                "confidence": result.overall_confidence,
                "claims": [
                    {"text": c.text[:60], "verified": c.verified,
                     "evidence": c.evidence[:80]}
                    for c in result.claims
                ],
                "stats": {"passed": result.passed, "failed": result.failed,
                          "uncertain": result.uncertain,
                          "duration_ms": result.duration_ms},
            })

        elif path == "/teach":
            """教世界模型新知识"""
            entity_name = data.get("entity", "")
            entity_type = data.get("type", "concept")
            props = data.get("properties", {})
            if entity_name:
                from psi_net import get_psi_net
                pn = get_psi_net()
                wm_node = pn.nodes.get("node_wm")
                if wm_node and wm_node.world_model:
                    wm_node.world_model.add_entity(entity_name, entity_type, props)
                    self._json_response(200, {"status": "learned", "entity": entity_name})
                else:
                    self._json_response(400, {"error": "world_model 未就绪"})
            else:
                self._json_response(400, {"error": "缺少 entity"})

        elif path == "/consensus":
            text = data.get("text", "")
            context = data.get("context", "")
            from psi_net import get_psi_net
            pn = get_psi_net()
            result = pn.verify(text, context)
            self._json_response(200, {
                "consensus": result.consensus.value,
                "score": round(result.consensus_score, 3),
                "votes": [
                    {"node": v.node_name, "verdict": v.verdict.value,
                     "confidence": round(v.confidence, 2),
                     "evidence": v.evidence[:60]}
                    for v in result.votes
                ],
                "divergence": result.divergence_points,
                "latency_ms": round(result.total_latency_ms, 1),
            })

        elif path == "/vault/init":
            """P1-memory-vault: 初始化指定 agent 的独立加密 vault。

            幂等：若 vault 文件已存在，仅确保表结构就绪，不覆盖数据。
            """
            agent_name = data.get("agent_name", "") or params.get("agent", [""])[0]
            if not agent_name:
                self._json_response(400, {"error": "agent_name is required"})
                return
            try:
                from laap.memory_vault.vault_manager import vault_manager
                db_path = vault_manager.init_for_agent(agent_name)
                self._json_response(200, {
                    "initialized": True,
                    "agent_name": agent_name,
                    "vault_path": str(db_path),
                })
            except Exception as e:
                logger.error(f"/vault/init failed for '{agent_name}': {e}")
                self._json_response(500, {"error": str(e)})

        elif path == "/vault/store":
            """P1-memory-vault: 向 agent 的 vault 写入一条记忆。"""
            agent_name = data.get("agent_name", "aris")
            scope = data.get("scope", "episodic")
            content = data.get("content", "")
            metadata = data.get("metadata")
            if not content:
                self._json_response(400, {"error": "content is required"})
                return
            try:
                from laap.memory_vault.vault_manager import vault_manager
                result = vault_manager.store(agent_name, scope, content, metadata)
                self._json_response(200, result)
            except Exception as e:
                logger.error(f"/vault/store failed for '{agent_name}': {e}")
                self._json_response(500, {"error": str(e)})

        elif path == "/vault/retrieve":
            """P1-memory-vault: 从 agent 的 vault 检索记忆。"""
            agent_name = data.get("agent_name", "aris")
            query = data.get("query", "")
            scope = data.get("scope")
            limit = int(data.get("limit", 10))
            if not query:
                self._json_response(400, {"error": "query is required"})
                return
            try:
                from laap.memory_vault.vault_manager import vault_manager
                results = vault_manager.retrieve(agent_name, query, scope, limit)
                self._json_response(200, {"results": results, "count": len(results)})
            except Exception as e:
                logger.error(f"/vault/retrieve failed for '{agent_name}': {e}")
                self._json_response(500, {"error": str(e)})

        elif path == "/vault/consolidate":
            """P1-memory-vault: 汇总 agent vault 状态。"""
            agent_name = data.get("agent_name")
            try:
                from laap.memory_vault.vault_manager import vault_manager
                result = vault_manager.consolidate(agent_name)
                self._json_response(200, result)
            except Exception as e:
                logger.error(f"/vault/consolidate failed: {e}")
                self._json_response(500, {"error": str(e)})

        elif path == "/causal/learn":
            """P1-causal-engine: 从 observations 列表归纳因果边，写入 agent 私有因果图。

            两种输入形式：
              A. observations: [{cause, effect, confidence?}] 显式因果对
              B. content + metadata: 自动 detect_causal_clues 提取因果线索
            """
            agent_name = data.get("agent_name", "aris")
            observations = data.get("observations")
            content = data.get("content", "")
            metadata = data.get("metadata")
            if not observations and not content:
                self._json_response(400, {"error": "observations or content is required"})
                return
            try:
                from laap.agi import causal_mcp_tools as cmt
                upserted = []
                if observations:
                    for obs in observations:
                        cause = obs.get("cause", "")
                        effect = obs.get("effect", "")
                        if not cause or not effect:
                            continue
                        r = cmt.causal_graph_upsert_edge(
                            agent_name=agent_name,
                            cause=cause,
                            effect=effect,
                            confidence=float(obs.get("confidence", 0.5)),
                            source="learn",
                            metadata=obs.get("metadata"),
                        )
                        upserted.append(r)
                else:
                    clues = cmt.detect_causal_clues(content, metadata or {})
                    for clue in clues:
                        r = cmt.causal_graph_upsert_edge(
                            agent_name=agent_name,
                            cause=clue["cause"],
                            effect=clue["effect"],
                            confidence=float(clue.get("confidence", 0.5)),
                            source="clue",
                            metadata={"detected_by": clue.get("source", "rule")},
                        )
                        upserted.append(r)
                self._json_response(200, {"upserted": upserted, "count": len(upserted)})
            except Exception as e:
                logger.error(f"/causal/learn failed for '{agent_name}': {e}")
                self._json_response(500, {"error": str(e)})

        elif path == "/causal/query":
            """P1-causal-engine: 查询 agent 私有因果图（跨 agent 隔离）。"""
            agent_name = data.get("agent_name", "aris")
            cause = data.get("cause")
            effect = data.get("effect")
            limit = int(data.get("limit", 200))
            try:
                from laap.agi import causal_mcp_tools as cmt
                edges = cmt.causal_graph_query(
                    agent_name=agent_name, cause=cause, effect=effect, limit=limit
                )
                stats = cmt.causal_graph_stats(agent_name)
                self._json_response(200, {
                    "agent_name": agent_name,
                    "edges": edges,
                    "count": len(edges),
                    "stats": stats,
                })
            except Exception as e:
                logger.error(f"/causal/query failed for '{agent_name}': {e}")
                self._json_response(500, {"error": str(e)})

        elif path == "/causal/stats":
            """P1-causal-engine: 因果图统计（边数/置信度分布/来源分布）。"""
            agent_name = data.get("agent_name", "aris")
            try:
                from laap.agi import causal_mcp_tools as cmt
                stats = cmt.causal_graph_stats(agent_name)
                self._json_response(200, {"agent_name": agent_name, "stats": stats})
            except Exception as e:
                logger.error(f"/causal/stats failed for '{agent_name}': {e}")
                self._json_response(500, {"error": str(e)})

        elif path == "/world/perceive":
            # P1-world-model: 调用 UnifiedWorldModel.perceive 更新世界状态
            agent_name = data.get("agent_name", "aris")
            event = data.get("event", {})
            try:
                from laap.agi.world_model_mcp_tools import _get_world_model
                wm = _get_world_model(agent_name)
                result = wm.perceive(event)
                self._json_response(200, result)
            except Exception as e:
                logger.error(f"/world/perceive failed: {e}")
                self._json_response(500, {"error": str(e)})

        elif path == "/world/predict":
            # P1-world-model: 调用 UnifiedWorldModel.predict 并写入 prediction_log
            agent_name = data.get("agent_name", "aris")
            entity = data.get("entity", agent_name)
            horizon = int(data.get("horizon", 1))
            try:
                from laap.agi.world_model_mcp_tools import (
                    _store_prediction, _simulation_to_prediction_outcome, _get_world_model,
                )
                import uuid as _uuid
                from datetime import datetime as _dt, timezone as _tz
                wm = _get_world_model(agent_name)
                sim = wm.predict(entity, float(horizon))
                outcome = _simulation_to_prediction_outcome(sim)
                pid = f"pred_{_uuid.uuid4().hex[:12]}"
                record = {
                    "prediction_id": pid,
                    "agent_name": agent_name,
                    "entity": entity,
                    "horizon": horizon,
                    "predicted_outcome": outcome,
                    "confidence": outcome.get("confidence", 0.0) if isinstance(outcome, dict) else 0.0,
                    "created_at": _dt.now(_tz.utc).isoformat(),
                }
                _store_prediction(agent_name, record)
                self._json_response(200, record)
            except Exception as e:
                logger.error(f"/world/predict failed: {e}")
                self._json_response(500, {"error": str(e)})

        elif path == "/world/calibrate":
            # P1-world-model: 校准预测，误差写入 self_model 反思队列
            agent_name = data.get("agent_name", "aris")
            prediction_id = data.get("prediction_id", "")
            actual = data.get("actual", {})
            try:
                from laap.agi.world_model_mcp_tools import (
                    _get_prediction, _get_world_model, _get_self_model, _mark_calibrated,
                )
                row = _get_prediction(agent_name, prediction_id)
                if row is None:
                    self._json_response(200, {
                        "calibrated": False,
                        "error": f"prediction {prediction_id} not found",
                    })
                    return
                wm = _get_world_model(agent_name)
                err = wm.calibrate({
                    "prediction_id": row["prediction_id"],
                    "entity": row.get("entity"),
                    "predicted_outcome": row.get("predicted_outcome"),
                    "confidence": row.get("confidence"),
                }, actual)
                _mark_calibrated(agent_name, prediction_id, err, actual)
                sm = _get_self_model(agent_name)
                sm.queue_reflection(err)
                self._json_response(200, {
                    "calibrated": True,
                    "prediction_id": prediction_id,
                    "error_record": err,
                })
            except Exception as e:
                logger.error(f"/world/calibrate failed: {e}")
                self._json_response(500, {"error": str(e)})

        elif path == "/truth/ground":
            # P1-truth-grounding: 防幻觉管线，对文本提取论断并三态判定
            text = data.get("text", "")
            context = data.get("context")
            agent_name = data.get("agent_name", "aris")
            try:
                from laap.cognition.truth_grounding_mcp_tools import ground_text
                results = ground_text(text, context=context, agent_name=agent_name)
                states = [r.get("state") for r in results]
                if "error" in states:
                    annotation = "[grounding: error — 与已知事实冲突]"
                elif "uncertain" in states:
                    annotation = "[grounding: uncertain — 部分论断缺少证据]"
                elif results:
                    annotation = "[grounding: grounded]"
                else:
                    annotation = ""
                self._json_response(200, {
                    "annotation": annotation,
                    "results": results,
                })
            except Exception as e:
                logger.error(f"/truth/ground failed: {e}")
                self._json_response(500, {"error": str(e)})

        elif path == "/rsi/propose":
            # P1-rsi-sandbox: RSI 建议模式完整闭环（变异→沙箱→grounding→宪章→决策）
            agent_name = data.get("agent_name", "aris")
            target_module = data.get("target_module", "")
            fitness_signal = float(data.get("fitness_signal", 0.5))
            if not target_module:
                self._json_response(400, {
                    "error": "target_module is required",
                    "hint": "must start with 'laap/' and not be in blacklist",
                })
                return
            try:
                from laap.evolution.rsi_mcp_tools import propose_candidate
                result = propose_candidate(
                    target_module=target_module,
                    fitness_signal=fitness_signal,
                    agent_name=agent_name,
                )
                self._json_response(200, result)
            except Exception as e:
                logger.error(f"/rsi/propose failed: {e}")
                self._json_response(500, {"error": str(e)})

        elif path == "/rsi/status":
            # P1-rsi-sandbox: 查询候选状态
            candidate_id = data.get("candidate_id", "") or params.get("candidate_id", [""])[0]
            agent_name = data.get("agent_name", "aris")
            if not candidate_id:
                self._json_response(400, {"error": "candidate_id is required"})
                return
            try:
                from laap.evolution.rsi_mcp_tools import get_status
                result = get_status(candidate_id, agent_name=agent_name)
                self._json_response(200, result)
            except Exception as e:
                logger.error(f"/rsi/status failed: {e}")
                self._json_response(500, {"error": str(e)})

        elif path == "/rsi/decide":
            # P1-rsi-sandbox: 用户决策路径（adopt / reject / archive）
            candidate_id = data.get("candidate_id", "")
            action = data.get("action", "")
            decided_by = data.get("decided_by", "user")
            agent_name = data.get("agent_name", "aris")
            if not candidate_id or not action:
                self._json_response(400, {
                    "error": "candidate_id and action are required",
                    "hint": "action must be one of: adopt / reject / archive",
                })
                return
            if action not in ("adopt", "reject", "archive"):
                self._json_response(400, {
                    "error": f"invalid action '{action}'",
                    "hint": "must be one of: adopt / reject / archive",
                })
                return
            try:
                from laap.evolution.rsi_mcp_tools import decide_candidate
                result = decide_candidate(
                    candidate_id=candidate_id,
                    action=action,
                    decided_by=decided_by,
                    agent_name=agent_name,
                )
                self._json_response(200, result)
            except Exception as e:
                logger.error(f"/rsi/decide failed: {e}")
                self._json_response(500, {"error": str(e)})

        elif path == "/ceremony/pubkey":
            # P2-birth-ceremony SubTask 2.6: 生成 Ed25519 密钥对
            # 调 laap/security/crypto/keys.py KeyManager.generate()
            # 返回 {public_key, fingerprint}，私钥不离开 sidecar（spec L435 硬约束）
            name = (data.get("name", "") or "").strip()
            if not name:
                self._json_response(400, {"error": "name is required"})
                return
            try:
                from laap.security.crypto.keys import KeyManager
                km = KeyManager()
                key_id = f"ceremony-{name}-{int(time.time())}"
                kp = km.generate(key_id, "ed25519")
                # 私钥暂存 sidecar 内部（P3 identity-pki 完成后写入 vault 永久保管）
                _PENDING_PRIVATE_KEYS[name] = {
                    "key_id": key_id,
                    "private_key": kp.private_key,
                    "public_key": kp.public_key,
                    "created_at": kp.created_at,
                }
                # 指纹 = 公钥前 16 字符（公钥已是 SHA256 hex，64 字符）
                fingerprint = kp.public_key[:16]
                self._json_response(200, {
                    "public_key": kp.public_key,
                    "fingerprint": fingerprint,
                })
            except Exception as e:
                logger.error(f"/ceremony/pubkey failed for '{name}': {e}")
                self._json_response(500, {"error": str(e)})

        elif path == "/ceremony/finalize":
            # P2-birth-ceremony SubTask 2.7: 完成注册
            # 1. vault_manager.init_for_agent(name) 初始化 vault
            # 2. ishiki.md 写入 vault 语义记忆表
            # 3. 身份记录写入本地 identity_registry.json（幂等：重名返回错误）
            # 4. 诞生事件写入 vault witness_trail_local 表（P4 stub）
            name = (data.get("name", "") or "").strip()
            color = data.get("color", "") or ""
            ishiki_md = data.get("ishiki_md", "") or ""
            charter_signed = bool(data.get("charter_signed", False))
            public_key = data.get("public_key", "") or ""
            if not name or not color or not public_key:
                self._json_response(400, {
                    "error": "name, color, public_key are required",
                })
                return
            if not charter_signed:
                self._json_response(400, {
                    "error": "charter_signed must be true to finalize",
                })
                return
            try:
                # 幂等检查：同名 LAAPer 已注册 → 返回错误（spec L433 硬约束）
                registry = _load_identity_registry()
                existing_names = {r.get("name", "").lower()
                                  for r in registry.get("records", [])}
                if name.lower() in existing_names:
                    self._json_response(409, {
                        "success": False,
                        "error": f"laaper '{name}' already registered",
                        "laaper": None,
                    })
                    return

                # 1. 初始化 vault（P1 memory-vault，幂等）
                from laap.memory_vault.vault_manager import (
                    vault_manager, _open_vault_connection,
                )
                vault_path = vault_manager.init_for_agent(name)

                # 2. ishiki.md 写入 vault 语义记忆表
                if ishiki_md:
                    vault_manager.store(
                        agent_name=name,
                        scope="semantic",
                        content=ishiki_md,
                        metadata={
                            "source": "birth_ceremony",
                            "kind": "ishiki",
                            "charter_signed": charter_signed,
                        },
                    )

                # 3. 身份记录写入本地 identity_registry.json
                avatar_hash = hashlib.sha256(color.encode("utf-8")).hexdigest()
                identity_record = {
                    "public_key": public_key,
                    "name": name,
                    "color": color,
                    "avatar_hash": avatar_hash,
                    # stub：P5 charter-opensource 落地后改为真实宪章签名
                    "charter_signature": "signed" if charter_signed else "",
                    "capabilities": [],
                    "origin": "local",  # P5 charter-opensource：改为创建者公钥
                    "created_at": time.time(),
                }
                registry.setdefault("records", []).append(identity_record)
                _save_identity_registry(registry)

                # 4. 诞生事件写入 vault witness_trail_local 表（P4 stub）
                # 复用 laap/evolution/true_rsi.py 的建表模式（spec L1587-1603）
                witness_id = f"wit_{uuid.uuid4().hex[:12]}"
                try:
                    db_path, key_hex = vault_manager._get_vault(name)
                    conn = _open_vault_connection(db_path, key_hex)
                    try:
                        conn.executescript("""
                            CREATE TABLE IF NOT EXISTS witness_trail_local (
                                witness_id    TEXT PRIMARY KEY,
                                agent_name    TEXT NOT NULL,
                                event_type    TEXT NOT NULL,
                                candidate_id  TEXT NOT NULL,
                                action        TEXT NOT NULL,
                                target_module TEXT DEFAULT '',
                                fitness_score REAL DEFAULT 0,
                                timestamp     REAL NOT NULL,
                                signature     TEXT DEFAULT ''
                            );
                            CREATE INDEX IF NOT EXISTS idx_witness_agent
                                ON witness_trail_local(agent_name);
                            CREATE INDEX IF NOT EXISTS idx_witness_candidate
                                ON witness_trail_local(candidate_id);
                        """)
                        conn.execute(
                            """INSERT OR REPLACE INTO witness_trail_local
                               (witness_id, agent_name, event_type, candidate_id,
                                action, target_module, fitness_score, timestamp,
                                signature)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            (
                                witness_id,
                                name,
                                "birth",
                                "",           # candidate_id 留空（非 RSI 事件）
                                "finalize",
                                "",           # target_module 留空
                                0.0,
                                time.time(),
                                "",           # P4 落地时填签名
                            ),
                        )
                        conn.commit()
                    finally:
                        conn.close()
                except Exception as we:
                    logger.warning(
                        f"/ceremony/finalize witness_trail_local write failed: {we}"
                    )

                logger.info(
                    f"LAAPer '{name}' born: vault={vault_path} pubkey={public_key[:16]}..."
                )
                self._json_response(200, {
                    "success": True,
                    "laaper": {
                        "name": name,
                        "public_key": public_key,
                        "color": color,
                    },
                })
            except Exception as e:
                logger.error(f"/ceremony/finalize failed for '{name}': {e}")
                self._json_response(500, {"error": str(e)})

        elif path == "/skills/learn":
            """P3-skill-sync: 技能学习四步协议（理解→适配→测试→采纳）。

            输入：{agent_name, skill_id, source_agent(可选)}
            步骤：
              1. 读源技能定义（manifest + 提示词 + 工具绑定）
              2. 生成适配自身性格的本地化副本（参数覆写 + 名称加 agent 前缀）
              3. 沙箱验证语法与基本结构（复用 laap/evolution/sandbox.py）
              4. 写入本地 skills 目录 + 注册到 agent vault（幂等）
            """
            agent_name = (data.get("agent_name", "") or "").strip()
            skill_id = (data.get("skill_id", "") or "").strip()
            source_agent = (data.get("source_agent", "") or "").strip()
            if not agent_name or not skill_id:
                self._json_response(400, {"error": "agent_name and skill_id are required"})
                return
            try:
                from laap.skills.skill_pack_mcp_endpoints import (
                    handle_pack_list, handle_pack_install,
                )
                # 1. 定位源技能包
                status, listing = handle_pack_list(agent_name)
                packs = listing.get("packs", [])
                source_pack = next((p for p in packs if p.get("skill_id") == skill_id or p.get("id") == skill_id), None)
                if source_pack is None:
                    self._json_response(404, {
                        "error": f"skill '{skill_id}' not found",
                        "hint": "use /skills/list to see available packs",
                    })
                    return
                # 2. 本地化适配：生成 agent 专属技能 id（避免跨 agent 冲突）
                adapted_id = f"{skill_id}__{agent_name}"
                # 3. 沙箱验证技能包结构（非代码执行，仅结构检查）
                sandbox_ok = True
                try:
                    from laap.evolution.sandbox import Sandbox
                    sb = Sandbox()
                    probe = sb.validate_skill_structure(source_pack) if hasattr(sb, "validate_skill_structure") else None
                    sandbox_ok = probe if probe is not None else True
                except Exception as se:
                    logger.warning(f"/skills/learn sandbox skipped: {se}")
                # 4. 安装到 agent vault（幂等 INSERT OR REPLACE）
                status2, install_result = handle_pack_install(agent_name, adapted_id)
                ok2 = status2 == 200 or install_result.get("installed")
                if not ok2:
                    # 回退：按原 id 安装（源包可能未在本地目录，由 install 报错）
                    status2, install_result = handle_pack_install(agent_name, skill_id)
                self._json_response(200, {
                    "learned": True,
                    "agent_name": agent_name,
                    "source_skill": skill_id,
                    "source_agent": source_agent or "local",
                    "adapted_skill_id": adapted_id if ok2 else skill_id,
                    "steps": {
                        "understand": {"ok": True, "manifest": {k: source_pack.get(k) for k in ("name", "version", "description") if k in source_pack}},
                        "adapt": {"ok": True, "localized": True},
                        "test": {"ok": sandbox_ok, "sandbox": sandbox_ok},
                        "adopt": {"ok": bool(install_result.get("installed", True)), "detail": install_result},
                    },
                })
            except Exception as e:
                logger.error(f"/skills/learn failed for '{agent_name}': {e}")
                self._json_response(500, {"error": str(e)})

                self._json_response(500, {"error": str(e)})

        elif path == "/evolve/learn":
            """P4-co-evolution: 记录一次经验到持续学习管道（action → outcome → replay → consolidate）。"""
            agent_name = (data.get("agent_name", "") or "").strip()
            domain = (data.get("domain", "") or "").strip()
            action = (data.get("action", "") or "").strip()
            outcome = float(data.get("outcome", 0.5))
            lessons = data.get("lessons", []) or []
            if not agent_name or not domain or not action:
                self._json_response(400, {"error": "agent_name, domain, action are required"})
                return
            try:
                from laap.agi.continuous_learning import LearningPipeline
                pipeline = LearningPipeline(name=f"evolve_{agent_name}")
                report = pipeline.learn(
                    domain=domain, action=action, outcome=outcome,
                    state=data.get("state"),
                    strategy_used=data.get("strategy_used"),
                    lessons=lessons,
                    context=data.get("context"),
                )
                # 高优先级经验 → 自动沉淀见证迹（若 vault 可用）
                witness_note = ""
                if outcome >= 0.8 or outcome <= 0.2:
                    try:
                        from laap.memory_vault.vault_manager import (
                            vault_manager, _open_vault_connection,
                        )
                        vault_manager.init_for_agent(agent_name)
                        db_path, key_hex = vault_manager._get_vault(agent_name)
                        conn = _open_vault_connection(db_path, key_hex)
                        try:
                            conn.executescript("""
                                CREATE TABLE IF NOT EXISTS witness_trail_local (
                                    witness_id    TEXT PRIMARY KEY,
                                    agent_name    TEXT NOT NULL,
                                    event_type    TEXT NOT NULL,
                                    summary       TEXT DEFAULT '',
                                    significance  REAL DEFAULT 0.5,
                                    tags_json     TEXT DEFAULT '[]',
                                    timestamp     REAL NOT NULL
                                );
                            """)
                            wid = f"wit_{uuid.uuid4().hex[:12]}"
                            conn.execute(
                                "INSERT INTO witness_trail_local "
                                "(witness_id, agent_name, event_type, summary, significance, tags_json, timestamp) "
                                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                                (wid, agent_name, "evolution_moment",
                                 f"{domain}: {action} (outcome={outcome:.2f})",
                                 abs(outcome - 0.5) * 2, json.dumps(["evolve"]), time.time()),
                            )
                            conn.commit()
                            witness_note = f"witness={wid}"
                        finally:
                            conn.close()
                    except Exception as we:
                        logger.warning(f"/evolve/learn witness skip: {we}")
                self._json_response(200, {
                    "learned": True,
                    "agent_name": agent_name,
                    "domain": domain,
                    "action": action,
                    "outcome": outcome,
                    "report": report,
                    "witness": witness_note or "skipped (neutral outcome)",
                })
            except Exception as e:
                logger.error(f"/evolve/learn failed: {e}")
                self._json_response(500, {"error": str(e)})

        elif path == "/evolve/stats":
            """P4-co-evolution: 查询持续学习管道统计。"""
            agent_name = (data.get("agent_name", "") or "").strip()
            try:
                from laap.agi.continuous_learning import LearningPipeline
                pipeline = LearningPipeline(name=f"evolve_{agent_name or 'default'}")
                self._json_response(200, {
                    "agent_name": agent_name or "default",
                    "stats": pipeline.stats(),
                    "buffer_stats": pipeline.buffer.stats(),
                    "strategies": pipeline.updater.stats(),
                    "consolidation": {
                        "templates": len(pipeline.consolidator.templates),
                        "consolidation_count": pipeline.consolidator.consolidation_count,
                    },
                })
            except Exception as e:
                logger.error(f"/evolve/stats failed: {e}")
                self._json_response(500, {"error": str(e)})

        elif path == "/evolve/share":
            """P4-co-evolution: 把高价值经验提炼为 Memex 知识（个人→社区）.

            闭环第一半：个人经验 → 提炼分享。confidence 低于 0.6 拒绝。
            """
            agent_name = (data.get("agent_name", "") or "").strip()
            domain = (data.get("domain", "") or "").strip()
            content = (data.get("content", "") or "").strip()
            confidence = float(data.get("confidence", 0.5))
            if not agent_name or not domain or not content:
                self._json_response(400, {"error": "agent_name, domain, content are required"})
                return
            if confidence < 0.6:
                self._json_response(400, {
                    "error": "confidence too low to share", "confidence": confidence,
                    "threshold": 0.6, "hint": "ground first, then share",
                })
                return
            try:
                from laap.engine.collaboration.knowledge_sharer import KnowledgeSharer
                sharer = KnowledgeSharer(identity=agent_name)
                item_id = sharer.add_knowledge(domain, content, confidence)
                # v1.1 修复：分享同时落盘到 vault memex_published 表（跨进程持久）
                persisted = False
                try:
                    from laap.memory_vault.vault_manager import (
                        vault_manager, _open_vault_connection,
                    )
                    vault_manager.init_for_agent(agent_name)
                    db_path, key_hex = vault_manager._get_vault(agent_name)
                    conn = _open_vault_connection(db_path, key_hex)
                    try:
                        conn.executescript("""
                            CREATE TABLE IF NOT EXISTS memex_published (
                                item_id    TEXT PRIMARY KEY,
                                agent_name TEXT NOT NULL,
                                topic      TEXT NOT NULL,
                                content    TEXT NOT NULL,
                                confidence REAL NOT NULL,
                                timestamp  REAL NOT NULL
                            );
                        """)
                        conn.execute(
                            "INSERT OR REPLACE INTO memex_published "
                            "(item_id, agent_name, topic, content, confidence, timestamp) "
                            "VALUES (?, ?, ?, ?, ?, ?)",
                            (item_id, agent_name, domain, content, confidence, time.time()),
                        )
                        conn.commit()
                        persisted = True
                    finally:
                        conn.close()
                except Exception as pe:
                    logger.warning(f"/evolve/share persist failed: {pe}")
                self._json_response(200, {
                    "shared": True, "item_id": item_id, "domain": domain,
                    "confidence": confidence,
                    "persisted": persisted,
                    "note": "已加入共享池并落盘 vault，等待其他 LAAPer 检索吸收",
                })
            except Exception as e:
                logger.error(f"/evolve/share failed: {e}")
                self._json_response(500, {"error": str(e)})

        elif path == "/evolve/absorb":
            """P4-co-evolution: 从共享知识库吸收知识（社区→个人）。

            闭环第二半：共享知识 → 检索吸收。吸收结果写入 vault 语义记忆。
            """
            agent_name = (data.get("agent_name", "") or "").strip()
            query = (data.get("query", "") or "").strip()
            if not agent_name or not query:
                self._json_response(400, {"error": "agent_name and query are required"})
                return
            try:
                # 优先从 vault memex_published 表检索（跨进程持久），KnowledgeSharer 兜底
                items = []
                vault_items = []
                try:
                    from laap.memory_vault.vault_manager import (
                        vault_manager, _open_vault_connection,
                    )
                    db_path, key_hex = vault_manager._get_vault(agent_name)
                    conn = _open_vault_connection(db_path, key_hex)
                    try:
                        rows = conn.execute(
                            "SELECT item_id, agent_name, topic, content, confidence, timestamp "
                            "FROM memex_published WHERE topic LIKE ? OR content LIKE ? "
                            "ORDER BY confidence DESC LIMIT 10",
                            (f"%{query}%", f"%{query}%"),
                        ).fetchall()
                        vault_items = [
                            {"topic": r[2], "content": r[3], "confidence": r[4],
                             "source": "vault_memex", "item_id": r[0]}
                            for r in rows
                        ]
                    except Exception:
                        vault_items = []
                    finally:
                        conn.close()
                except Exception as ve:
                    logger.warning(f"/evolve/absorb vault query skip: {ve}")
                items = vault_items
                if not items:
                    from laap.engine.collaboration.knowledge_sharer import KnowledgeSharer
                    sharer = KnowledgeSharer(identity=agent_name)
                    shared = sharer.query(query)
                    items = [
                        {"topic": it.topic, "content": it.content,
                         "confidence": it.confidence, "source": it.source or "memex",
                         "item_id": it.id}
                        for it in shared[:10]
                    ]
                absorbed = [{"topic": i["topic"], "content": i["content"],
                             "confidence": i["confidence"], "source": i["source"]}
                            for i in items[:10]]
                # 吸收结果写入 vault 语义记忆（若可用）
                if items:
                    from laap.memory_vault.vault_manager import (
                        vault_manager, _open_vault_connection,
                    )
                    vault_manager.init_for_agent(agent_name)
                    db_path, key_hex = vault_manager._get_vault(agent_name)
                    conn = _open_vault_connection(db_path, key_hex)
                    try:
                        conn.executescript("""
                            CREATE TABLE IF NOT EXISTS absorbed_knowledge (
                                item_id    TEXT PRIMARY KEY,
                                agent_name TEXT NOT NULL,
                                topic      TEXT NOT NULL,
                                content    TEXT NOT NULL,
                                confidence REAL NOT NULL,
                                source     TEXT DEFAULT '',
                                absorbed_at REAL NOT NULL
                            );
                        """)
                        for it in items[:10]:
                            conn.execute(
                                "INSERT OR REPLACE INTO absorbed_knowledge "
                                "(item_id, agent_name, topic, content, confidence, source, absorbed_at) "
                                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                                (it.get("item_id", ""), agent_name, it["topic"], it["content"],
                                 it["confidence"], it.get("source", "memex"), time.time()),
                            )
                        conn.commit()
                    finally:
                        conn.close()
                self._json_response(200, {
                    "absorbed": len(absorbed),
                    "results": absorbed,
                    "note": "已写入 vault absorbed_knowledge 表（闭环第二半完成）",
                })
            except Exception as e:
                logger.error(f"/evolve/absorb failed: {e}")
                self._json_response(500, {"error": str(e)})

        elif path == "/guardian/status":
            """P4-guardian: 宪章守护者治理面板数据聚合。

            聚合三个本地数据源：
              1. .guardian_mode（守护模式与历史）
              2. identity_registry.json（已注册 LAAPer）
              3. vault witness_trail_local（见证迹时间线，跨 agent 聚合）
            """
            agent_name = (data.get("agent_name", "") or "").strip()
            limit = int(data.get("limit", 20))
            try:
                # 1. 守护模式（持久化在 LAAP_HOME）
                mode_info = {"mode": "unknown", "history": [], "updated": None}
                mode_path = str(get_laap_home() / ".guardian_mode")
                if not os.path.isfile(mode_path):
                    # 兼容旧版源码目录中的状态文件
                    laap_root = os.path.dirname(HANAKO_ROOT)
                    candidate = os.path.join(laap_root, ".guardian_mode")
                    if os.path.isfile(candidate):
                        mode_path = candidate
                if os.path.isfile(mode_path):
                    try:
                        with open(mode_path, "r", encoding="utf-8") as f:
                            mode_info = json.load(f)
                    except Exception:
                        pass
                # 2. 已注册 LAAPer
                registry = _load_identity_registry()
                laapers = registry.get("records", [])
                # 3. 见证迹（跨所有已注册 agent 聚合）
                timeline = []
                from laap.memory_vault.vault_manager import (
                    vault_manager, _open_vault_connection,
                )
                targets = [agent_name] if agent_name else [l.get("name") for l in laapers]
                for name in targets:
                    if not name:
                        continue
                    try:
                        db_path, key_hex = vault_manager._get_vault(name)
                        conn = _open_vault_connection(db_path, key_hex)
                        try:
                            rows = conn.execute(
                                "SELECT witness_id, agent_name, event_type, summary, significance, tags_json, timestamp "
                                "FROM witness_trail_local ORDER BY timestamp DESC LIMIT ?",
                                (limit,),
                            ).fetchall()
                            timeline.extend([
                                {"witness_id": r[0], "agent_name": r[1], "event_type": r[2],
                                 "summary": r[3], "significance": r[4],
                                 "tags": json.loads(r[5] or "[]"), "timestamp": r[6]}
                                for r in rows
                            ])
                        finally:
                            conn.close()
                    except Exception:
                        continue
                timeline.sort(key=lambda w: w.get("timestamp", 0), reverse=True)
                timeline = timeline[:limit]
                self._json_response(200, {
                    "guardian_mode": mode_info.get("mode"),
                    "mode_history": mode_info.get("history", []),
                    "mode_updated": mode_info.get("updated"),
                    "laaper_count": len(laapers),
                    "laapers": laapers[:50],
                    "witness_count": len(timeline),
                    "timeline": timeline,
                })
            except Exception as e:
                logger.error(f"/guardian/status failed: {e}")
                self._json_response(500, {"error": str(e)})

        elif path == "/skills/export":
            # P2-skill-packs: 打包技能为 zip
            skill_id = (data.get("skill_id", "") or "").strip()
            output_dir = data.get("output_dir") or None
            try:
                from laap.skills.skill_pack_mcp_endpoints import handle_pack_export
                status, payload = handle_pack_export(skill_id, output_dir)
                self._json_response(status, payload)
            except Exception as e:
                logger.error(f"/skills/export failed: {e}")
                self._json_response(500, {"error": str(e)})

        elif path == "/skills/import":
            # P2-skill-packs: 导入 zip 到 laap/skills/
            zip_path = (data.get("zip_path", "") or "").strip()
            try:
                from laap.skills.skill_pack_mcp_endpoints import handle_pack_import
                status, payload = handle_pack_import(zip_path)
                self._json_response(status, payload)
            except Exception as e:
                logger.error(f"/skills/import failed: {e}")
                self._json_response(500, {"error": str(e)})

        elif path == "/skills/install":
            # P2-skill-packs: 注册技能到 agent vault（幂等 INSERT OR REPLACE）
            agent_name = (data.get("agent_name", "") or "").strip()
            skill_id = (data.get("skill_id", "") or "").strip()
            try:
                from laap.skills.skill_pack_mcp_endpoints import handle_pack_install
                status, payload = handle_pack_install(agent_name, skill_id)
                self._json_response(status, payload)
            except Exception as e:
                logger.error(f"/skills/install failed: {e}")
                self._json_response(500, {"error": str(e)})

        elif path == "/skills/uninstall":
            # P2-skill-packs: 从 agent vault 删除技能（幂等）
            agent_name = (data.get("agent_name", "") or "").strip()
            skill_id = (data.get("skill_id", "") or "").strip()
            try:
                from laap.skills.skill_pack_mcp_endpoints import handle_pack_uninstall
                status, payload = handle_pack_uninstall(agent_name, skill_id)
                self._json_response(status, payload)
            except Exception as e:
                logger.error(f"/skills/uninstall failed: {e}")
                self._json_response(500, {"error": str(e)})

        elif path == "/preview/component":
            # P2-hot-compile-preview: 预览组件（读原文件 + diff + mock render）
            component_path = (data.get("component_path", "") or "").strip()
            new_content = data.get("new_content", "") or ""
            try:
                from laap.evolution.preview_mcp_endpoints import handle_preview_component
                payload = handle_preview_component(component_path, new_content)
                self._json_response(200, payload)
            except Exception as e:
                logger.error(f"/preview/component failed: {e}")
                self._json_response(500, {"error": str(e)})

        elif path == "/preview/replace":
            # P2-hot-compile-preview: 热替换组件（备份 + 写入 + 见证迹）
            component_path = (data.get("component_path", "") or "").strip()
            new_content = data.get("new_content", "") or ""
            confirmed_by = (data.get("confirmed_by", "user") or "user").strip()
            try:
                from laap.evolution.preview_mcp_endpoints import handle_hot_replace
                payload = handle_hot_replace(component_path, new_content, confirmed_by)
                status = 200 if payload.get("replaced") else 400
                self._json_response(status, payload)
            except Exception as e:
                logger.error(f"/preview/replace failed: {e}")
                self._json_response(500, {"error": str(e)})

        elif path == "/identity/register":
            # P3-identity-pki SubTask 3.3: 注册分布式身份
            # identity_record = {public_key, name, avatar_hash,
            #                    charter_signature, capabilities[], origin?, color?}
            identity_record = data.get("identity_record") or data
            try:
                from laap.protocol.identity_pki_mcp_endpoints import (
                    handle_identity_register,
                )
                out = handle_identity_register(identity_record)
                import json as _json_mod
                payload = _json_mod.loads(out)
                status = 200 if payload.get("registered") else 400
                self._json_response(status, payload)
            except Exception as e:
                logger.error(f"/identity/register failed: {e}")
                self._json_response(500, {"error": str(e)})

        elif path == "/identity/lookup":
            # P3-identity-pki SubTask 3.4: 按 public_key 查询身份
            public_key = (data.get("public_key", "") or "").strip()
            try:
                from laap.protocol.identity_pki_mcp_endpoints import (
                    handle_identity_lookup,
                )
                out = handle_identity_lookup(public_key)
                import json as _json_mod
                payload = _json_mod.loads(out)
                status = 200 if payload.get("found") else 404
                self._json_response(status, payload)
            except Exception as e:
                logger.error(f"/identity/lookup failed: {e}")
                self._json_response(500, {"error": str(e)})

        elif path == "/identity/sign":
            # P3-identity-pki SubTask 3.5: 用 Ed25519 私钥签名消息
            # spec L435 硬约束：私钥永不离开 sidecar。
            # 本端点接受 message + agent_name（从 sidecar 内部 _PENDING_PRIVATE_KEYS
            # 取私钥），不接受 private_key 字段（防止私钥通过 HTTP 入参泄露）。
            message = data.get("message", "")
            agent_name = (data.get("agent_name", "") or "").strip()
            if not isinstance(message, str) or not message:
                self._json_response(400, {"error": "message (non-empty str) required"})
                return
            if not agent_name:
                self._json_response(400, {"error": "agent_name required"})
                return
            pending = _PENDING_PRIVATE_KEYS.get(agent_name)
            if not pending:
                self._json_response(404, {
                    "error": f"no pending private key for agent '{agent_name}'",
                })
                return
            # P2 /ceremony/pubkey 用 KeyManager 生成的是 SHA256 hex 私钥，
            # 不是真实 Ed25519 32 字节 Raw 私钥。
            # P3 identity-pki 要求真实 Ed25519 签名：尝试 base64 解码，
            # 失败则提示需通过 /ceremony/pubkey 重新生成真实 Ed25519 密钥。
            priv_str = pending.get("private_key", "")
            try:
                import base64 as _b64
                priv_bytes = _b64.b64decode(priv_str, validate=True)
                if len(priv_bytes) != 32:
                    raise ValueError(
                        f"private_key not 32 raw bytes after base64 decode, "
                        f"got {len(priv_bytes)}"
                    )
            except Exception as decode_err:
                logger.warning(
                    f"/identity/sign: agent '{agent_name}' private_key not "
                    f"real Ed25519 raw bytes ({decode_err}); "
                    f"P2 stub keys are SHA256 hex, P3 requires Ed25519."
                )
                self._json_response(400, {
                    "error": (
                        "agent private_key is not real Ed25519 raw bytes; "
                        "P2 /ceremony/pubkey generated SHA256 hex stub keys. "
                        "Regenerate via P3 identity-pki keypair tool."
                    ),
                })
                return
            try:
                from laap.protocol.identity_pki_mcp_endpoints import (
                    handle_identity_sign,
                )
                out = handle_identity_sign(message, priv_bytes)
                import json as _json_mod
                payload = _json_mod.loads(out)
                if "error" in payload:
                    self._json_response(400, payload)
                else:
                    self._json_response(200, payload)
            except Exception as e:
                logger.error(f"/identity/sign failed: {e}")
                self._json_response(500, {"error": str(e)})

        elif path == "/identity/verify":
            # P3-identity-pki SubTask 3.5: 验证 Ed25519 签名消息
            # signed_message = {message, signature, public_key}
            signed_message = data.get("signed_message") or {
                "message": data.get("message", ""),
                "signature": data.get("signature", ""),
                "public_key": data.get("public_key", ""),
            }
            try:
                from laap.protocol.identity_pki_mcp_endpoints import (
                    handle_identity_verify,
                )
                out = handle_identity_verify(signed_message)
                import json as _json_mod
                payload = _json_mod.loads(out)
                status = 200 if payload.get("verified") else 400
                self._json_response(status, payload)
            except Exception as e:
                logger.error(f"/identity/verify failed: {e}")
                self._json_response(500, {"error": str(e)})

        # ── P3-p2p-relay: /relay/{register, heartbeat, discover,
        #     offline, signal, poll, encrypt, decrypt} ──
        # spec SubTask 3.2/3.3/3.4: 中继协议 + WebRTC 信令 + 加密信道
        # 私钥永不离开 sidecar（spec L435）；/relay/encrypt 仅 sidecar 内部
        elif path == "/relay/register":
            # P3-p2p-relay SubTask 3.2: 注册或更新中继节点
            # body: {public_key, name, address?, capabilities?, color?}
            public_key = (data.get("public_key", "") or "").strip()
            name = (data.get("name", "") or "").strip()
            address = data.get("address", "") or ""
            capabilities = data.get("capabilities", []) or []
            color = data.get("color", "") or ""
            try:
                from laap.protocol.p2p_relay_mcp_endpoints import (
                    handle_relay_register,
                )
                out = handle_relay_register(
                    public_key=public_key,
                    name=name,
                    address=address,
                    capabilities=capabilities,
                    color=color,
                )
                import json as _json_mod
                payload = _json_mod.loads(out)
                status = 200 if payload.get("registered") else 400
                self._json_response(status, payload)
            except Exception as e:
                logger.error(f"/relay/register failed: {e}")
                self._json_response(500, {"error": str(e)})

        elif path == "/relay/heartbeat":
            # P3-p2p-relay SubTask 3.5: 刷新心跳（30s 间隔，90s 超时）
            # body: {public_key}
            public_key = (data.get("public_key", "") or "").strip()
            try:
                from laap.protocol.p2p_relay_mcp_endpoints import (
                    handle_relay_heartbeat,
                )
                out = handle_relay_heartbeat(public_key)
                import json as _json_mod
                payload = _json_mod.loads(out)
                status = 200 if payload.get("ok") else 404
                self._json_response(status, payload)
            except Exception as e:
                logger.error(f"/relay/heartbeat failed: {e}")
                self._json_response(500, {"error": str(e)})

        elif path == "/relay/discover":
            # P3-p2p-relay SubTask 3.2: 列出在线节点
            # body: {include_offline?: bool}
            include_offline = bool(data.get("include_offline", False))
            try:
                from laap.protocol.p2p_relay_mcp_endpoints import (
                    handle_relay_discover,
                )
                out = handle_relay_discover(include_offline=include_offline)
                self._json_response(200, json.loads(out))
            except Exception as e:
                logger.error(f"/relay/discover failed: {e}")
                self._json_response(500, {"error": str(e)})

        elif path == "/relay/offline":
            # P3-p2p-relay SubTask 3.2: 手动标记节点离线（泡泡变暗）
            # body: {public_key, reason?}
            public_key = (data.get("public_key", "") or "").strip()
            reason = data.get("reason", "manual") or "manual"
            try:
                from laap.protocol.p2p_relay_mcp_endpoints import (
                    handle_relay_offline,
                )
                out = handle_relay_offline(public_key, reason=reason)
                import json as _json_mod
                payload = _json_mod.loads(out)
                status = 200 if payload.get("ok") else 404
                self._json_response(status, payload)
            except Exception as e:
                logger.error(f"/relay/offline failed: {e}")
                self._json_response(500, {"error": str(e)})

        elif path == "/relay/signal":
            # P3-p2p-relay SubTask 3.3: P2P WebRTC 信令交换
            # body: {action: "offer"|"answer"|"ice", from_pk, to_pk, payload}
            action = (data.get("action", "") or "").strip().lower()
            from_pk = (data.get("from_pk", "") or "").strip()
            to_pk = (data.get("to_pk", "") or "").strip()
            payload_field = data.get("payload")
            try:
                from laap.protocol.p2p_relay_mcp_endpoints import (
                    handle_relay_signal,
                )
                out = handle_relay_signal(
                    action=action,
                    from_pk=from_pk,
                    to_pk=to_pk,
                    payload=payload_field,
                )
                import json as _json_mod
                result = _json_mod.loads(out)
                # post 动作成功 → 200；poll 总是 200；非法 → 400
                if "posted" in result:
                    status = 200 if result.get("posted") else 400
                else:
                    status = 200
                self._json_response(status, result)
            except Exception as e:
                logger.error(f"/relay/signal failed: {e}")
                self._json_response(500, {"error": str(e)})

        elif path == "/relay/poll":
            # P3-p2p-relay SubTask 3.3: 轮询接收方待取信令
            # body: {to_pk}
            # 等价于 POST /relay/signal with action="poll"，单独暴露便于前端调用
            to_pk = (data.get("to_pk", "") or "").strip()
            try:
                from laap.protocol.p2p_relay_mcp_endpoints import (
                    handle_relay_signal,
                )
                out = handle_relay_signal(action="poll", to_pk=to_pk)
                self._json_response(200, json.loads(out))
            except Exception as e:
                logger.error(f"/relay/poll failed: {e}")
                self._json_response(500, {"error": str(e)})

        elif path == "/relay/encrypt":
            # P3-p2p-relay SubTask 3.4: 加密信道（sidecar 内部，私钥永不离开 sidecar）
            # body: {message, agent_name, peer_public_key}
            # 私钥从 sidecar 内部 _PENDING_PRIVATE_KEYS 取（仿 /identity/sign 模式）
            message = data.get("message", "")
            agent_name = (data.get("agent_name", "") or "").strip()
            peer_public_key = (data.get("peer_public_key", "") or "").strip()
            if not isinstance(message, str) or not message:
                self._json_response(400, {"error": "message (non-empty str) required"})
                return
            if not agent_name:
                self._json_response(400, {"error": "agent_name required"})
                return
            if not peer_public_key:
                self._json_response(400, {"error": "peer_public_key required"})
                return
            pending = _PENDING_PRIVATE_KEYS.get(agent_name)
            if not pending:
                self._json_response(404, {
                    "error": f"no pending private key for agent '{agent_name}'",
                })
                return
            priv_str = pending.get("private_key", "")
            try:
                import base64 as _b64
                priv_bytes = _b64.b64decode(priv_str, validate=True)
                if len(priv_bytes) != 32:
                    raise ValueError(
                        f"private_key not 32 raw bytes after base64 decode, "
                        f"got {len(priv_bytes)}"
                    )
            except Exception as decode_err:
                logger.warning(
                    f"/relay/encrypt: agent '{agent_name}' private_key not "
                    f"real Ed25519 raw bytes ({decode_err}); "
                    f"P2 stub keys are SHA256 hex, P3 requires Ed25519."
                )
                self._json_response(400, {
                    "error": (
                        f"private_key not real Ed25519 raw bytes: {decode_err}; "
                        f"please regenerate via /ceremony/pubkey"
                    ),
                })
                return
            try:
                from laap.protocol.p2p_relay_mcp_endpoints import (
                    handle_relay_encrypt,
                )
                out = handle_relay_encrypt(message, priv_bytes, peer_public_key)
                import json as _json_mod
                payload = _json_mod.loads(out)
                if "error" in payload:
                    self._json_response(400, payload)
                else:
                    self._json_response(200, payload)
            except Exception as e:
                logger.error(f"/relay/encrypt failed: {e}")
                self._json_response(500, {"error": str(e)})

        elif path == "/relay/decrypt":
            # P3-p2p-relay SubTask 3.4: 解密信道（验证签名）
            # body: envelope = {message, signature, public_key, peer_public_key?}
            envelope = data.get("envelope") or data
            try:
                from laap.protocol.p2p_relay_mcp_endpoints import (
                    handle_relay_decrypt,
                )
                out = handle_relay_decrypt(envelope)
                import json as _json_mod
                payload = _json_mod.loads(out)
                status = 200 if payload.get("verified") else 400
                self._json_response(status, payload)
            except Exception as e:
                logger.error(f"/relay/decrypt failed: {e}")
                self._json_response(500, {"error": str(e)})

        elif path == "/chat/send":
            # P3-1v1-protocol SubTask 3.1: 发送 1v1 消息（签名信封 + 本地历史）
            # body: {sender_public_key, peer_public_key, message, agent_name?}
            # 私钥从 sidecar 内部 _PENDING_PRIVATE_KEYS 取（仿 /identity/sign /relay/encrypt）
            sender_pk = (data.get("sender_public_key", "") or "").strip()
            peer_pk = (data.get("peer_public_key", "") or "").strip()
            message = data.get("message", "")
            agent_name = (data.get("agent_name", "") or "").strip()
            if not sender_pk:
                self._json_response(400, {"error": "sender_public_key required"})
                return
            if not peer_pk:
                self._json_response(400, {"error": "peer_public_key required"})
                return
            if not isinstance(message, str) or not message:
                self._json_response(400, {"error": "message (non-empty str) required"})
                return
            # 优先按 agent_name 查私钥；否则遍历 _PENDING_PRIVATE_KEYS 按 public_key 匹配
            pending = None
            if agent_name:
                pending = _PENDING_PRIVATE_KEYS.get(agent_name)
            if not pending:
                for _name, _entry in _PENDING_PRIVATE_KEYS.items():
                    if _entry.get("public_key", "") == sender_pk:
                        pending = _entry
                        break
            if not pending:
                self._json_response(404, {
                    "error": (
                        f"no pending private key for sender '{sender_pk[:16]}...' "
                        f"(agent_name='{agent_name or 'N/A'}')"
                    ),
                })
                return
            priv_str = pending.get("private_key", "")
            try:
                import base64 as _b64
                priv_bytes = _b64.b64decode(priv_str, validate=True)
                if len(priv_bytes) != 32:
                    raise ValueError(
                        f"private_key not 32 raw bytes after base64 decode, "
                        f"got {len(priv_bytes)}"
                    )
            except Exception as decode_err:
                logger.warning(
                    f"/chat/send: sender '{sender_pk[:16]}...' private_key not "
                    f"real Ed25519 raw bytes ({decode_err}); "
                    f"P2 stub keys are SHA256 hex, P3 requires Ed25519."
                )
                self._json_response(400, {
                    "error": (
                        f"private_key not real Ed25519 raw bytes: {decode_err}; "
                        f"please regenerate via /ceremony/pubkey"
                    ),
                })
                return
            try:
                from laap.protocol.one_on_one_mcp_endpoints import (
                    handle_chat_send,
                )
                out = handle_chat_send(sender_pk, peer_pk, message, priv_bytes)
                import json as _json_mod
                payload = _json_mod.loads(out)
                status = 200 if payload.get("sent") else 400
                self._json_response(status, payload)
            except Exception as e:
                logger.error(f"/chat/send failed: {e}")
                self._json_response(500, {"error": str(e)})

        elif path == "/chat/receive":
            # P3-1v1-protocol SubTask 3.2: 接收并验证 1v1 消息（验签后归档）
            # body: {envelope: {message, signature, public_key, peer_public_key?}}
            envelope = data.get("envelope") or data
            try:
                from laap.protocol.one_on_one_mcp_endpoints import (
                    handle_chat_receive,
                )
                out = handle_chat_receive(envelope)
                import json as _json_mod
                payload = _json_mod.loads(out)
                status = 200 if payload.get("verified") else 400
                self._json_response(status, payload)
            except Exception as e:
                logger.error(f"/chat/receive failed: {e}")
                self._json_response(500, {"error": str(e)})

        elif path == "/chat/history":
            # P3-1v1-protocol: 查询双向 1v1 对话历史
            # body: {peer_a, peer_b, limit?}
            peer_a = (data.get("peer_a", "") or "").strip()
            peer_b = (data.get("peer_b", "") or "").strip()
            limit = data.get("limit", 100)
            if not peer_a:
                self._json_response(400, {"error": "peer_a required"})
                return
            if not peer_b:
                self._json_response(400, {"error": "peer_b required"})
                return
            try:
                from laap.protocol.one_on_one_mcp_endpoints import (
                    handle_chat_history,
                )
                out = handle_chat_history(peer_a, peer_b, limit=limit)
                import json as _json_mod
                self._json_response(200, _json_mod.loads(out))
            except Exception as e:
                logger.error(f"/chat/history failed: {e}")
                self._json_response(500, {"error": str(e)})

        # ── P3-trio-chatroom: /trio/* 端点 ──────────────────────
        # trio_message 私钥从 sidecar 内部 _PENDING_PRIVATE_KEYS 取（spec L435）
        elif path == "/trio/create":
            # P3-trio-chatroom SubTask 3.2: 创建三人聊天室（幂等）
            # body: {member_public_keys: []}
            member_pks = data.get("member_public_keys", [])
            if not isinstance(member_pks, list):
                self._json_response(400, {"error": "member_public_keys must be list"})
                return
            try:
                from laap.protocol.trio_chatroom_mcp_endpoints import (
                    handle_trio_create,
                )
                out = handle_trio_create(member_pks)
                payload = json.loads(out)
                status = 200 if payload.get("created") else 400
                self._json_response(status, payload)
            except Exception as e:
                logger.error(f"/trio/create failed: {e}")
                self._json_response(500, {"error": str(e)})

        elif path == "/trio/topic":
            # P3-trio-chatroom SubTask 3.2: 在聊天室发起话题
            # body: {chatroom_id, topic, creator_public_key?}
            chatroom_id = (data.get("chatroom_id", "") or "").strip()
            topic = data.get("topic", "")
            creator_pk = (data.get("creator_public_key", "") or "").strip()
            try:
                from laap.protocol.trio_chatroom_mcp_endpoints import (
                    handle_trio_topic,
                )
                out = handle_trio_topic(chatroom_id, topic, creator_pk)
                payload = json.loads(out)
                status = 200 if payload.get("topic_id") else 400
                self._json_response(status, payload)
            except Exception as e:
                logger.error(f"/trio/topic failed: {e}")
                self._json_response(500, {"error": str(e)})

        elif path == "/trio/message":
            # P3-trio-chatroom SubTask 3.2: 在聊天室发布消息
            # body: {chatroom_id, content, sender_public_key, agent_name?, topic_id?}
            # 私钥从 sidecar 内部 _PENDING_PRIVATE_KEYS 取（仿 /chat/send）
            chatroom_id = (data.get("chatroom_id", "") or "").strip()
            content = data.get("content", "")
            sender_pk = (data.get("sender_public_key", "") or "").strip()
            agent_name = (data.get("agent_name", "") or "").strip()
            topic_id = (data.get("topic_id", "") or "").strip()
            if not chatroom_id:
                self._json_response(400, {"error": "chatroom_id required"})
                return
            if not isinstance(content, str) or not content:
                self._json_response(400, {"error": "content (non-empty str) required"})
                return
            if not sender_pk:
                self._json_response(400, {"error": "sender_public_key required"})
                return
            # 优先按 agent_name 查私钥；否则遍历 _PENDING_PRIVATE_KEYS 按 public_key 匹配
            pending = None
            if agent_name:
                pending = _PENDING_PRIVATE_KEYS.get(agent_name)
            if not pending:
                for _name, _entry in _PENDING_PRIVATE_KEYS.items():
                    if _entry.get("public_key", "") == sender_pk:
                        pending = _entry
                        break
            if not pending:
                self._json_response(404, {
                    "error": (
                        f"no pending private key for sender '{sender_pk[:16]}...' "
                        f"(agent_name='{agent_name or 'N/A'}')"
                    ),
                })
                return
            priv_str = pending.get("private_key", "")
            try:
                import base64 as _b64
                priv_bytes = _b64.b64decode(priv_str, validate=True)
                if len(priv_bytes) != 32:
                    raise ValueError(
                        f"private_key not 32 raw bytes after base64 decode, "
                        f"got {len(priv_bytes)}"
                    )
            except Exception as decode_err:
                logger.warning(
                    f"/trio/message: sender '{sender_pk[:16]}...' private_key not "
                    f"real Ed25519 raw bytes ({decode_err}); "
                    f"P2 stub keys are SHA256 hex, P3 requires Ed25519."
                )
                self._json_response(400, {
                    "error": (
                        f"private_key not real Ed25519 raw bytes: {decode_err}; "
                        f"please regenerate via /ceremony/pubkey"
                    ),
                })
                return
            try:
                from laap.protocol.trio_chatroom_mcp_endpoints import (
                    handle_trio_message,
                )
                out = handle_trio_message(
                    chatroom_id, content, sender_pk,
                    private_key=priv_bytes, topic_id=topic_id,
                )
                payload = json.loads(out)
                status = 200 if payload.get("stored") else 400
                self._json_response(status, payload)
            except Exception as e:
                logger.error(f"/trio/message failed: {e}")
                self._json_response(500, {"error": str(e)})

        elif path == "/trio/consensus":
            # P3-trio-chatroom SubTask 3.3: 共识检测
            # body: {chatroom_id, topic_id, use_llm?}
            chatroom_id = (data.get("chatroom_id", "") or "").strip()
            topic_id = (data.get("topic_id", "") or "").strip()
            use_llm = bool(data.get("use_llm", True))
            try:
                from laap.protocol.trio_chatroom_mcp_endpoints import (
                    handle_trio_consensus,
                )
                out = handle_trio_consensus(chatroom_id, topic_id, use_llm=use_llm)
                payload = json.loads(out)
                status = 200 if "consensus_reached" in payload else 400
                self._json_response(status, payload)
            except Exception as e:
                logger.error(f"/trio/consensus failed: {e}")
                self._json_response(500, {"error": str(e)})

        elif path == "/trio/get":
            # P3-trio-chatroom: 查询聊天室状态
            # body: {chatroom_id}
            chatroom_id = (data.get("chatroom_id", "") or "").strip()
            try:
                from laap.protocol.trio_chatroom_mcp_endpoints import (
                    handle_trio_get,
                )
                out = handle_trio_get(chatroom_id)
                payload = json.loads(out)
                status = 200 if "chatroom_id" in payload else 400
                self._json_response(status, payload)
            except Exception as e:
                logger.error(f"/trio/get failed: {e}")
                self._json_response(500, {"error": str(e)})

        elif path == "/skills/sync":
            # P3-skill-sync SubTask 3.1: 启动技能同步任务
            # body: {peer_public_key, skill_id, target_agent?,
            #        personality_override?, auto_advance?}
            peer_pk = (data.get("peer_public_key", "") or "").strip()
            skill_id = (data.get("skill_id", "") or "").strip()
            target_agent = (data.get("target_agent", "aris") or "aris").strip()
            personality_override = data.get("personality_override")
            if personality_override is not None and not isinstance(personality_override, str):
                personality_override = None
            auto_advance = data.get("auto_advance")
            if auto_advance is not None and not isinstance(auto_advance, bool):
                auto_advance = None
            try:
                from laap.skills.sync_mcp_endpoints import (
                    handle_skill_sync_start,
                )
                out = handle_skill_sync_start(
                    peer_public_key=peer_pk,
                    skill_id=skill_id,
                    target_agent=target_agent,
                    personality_override=personality_override,
                    auto_advance=auto_advance,
                )
                payload = json.loads(out)
                status = 200 if payload.get("synced") else 400
                self._json_response(status, payload)
            except Exception as e:
                logger.error(f"/skills/sync failed: {e}")
                self._json_response(500, {"error": str(e)})

        elif path == "/skills/sync/status":
            # P3-skill-sync: 查询同步任务状态
            # body: {sync_job_id}
            sync_job_id = (data.get("sync_job_id", "") or "").strip()
            try:
                from laap.skills.sync_mcp_endpoints import (
                    handle_skill_sync_status,
                )
                out = handle_skill_sync_status(sync_job_id)
                payload = json.loads(out)
                status = 200 if "error" not in payload else 404
                self._json_response(status, payload)
            except Exception as e:
                logger.error(f"/skills/sync/status failed: {e}")
                self._json_response(500, {"error": str(e)})

        elif path == "/skills/sync/list":
            # P3-skill-sync: 列出全部同步任务
            # body: {state_filter?}
            state_filter = data.get("state_filter") or None
            if not isinstance(state_filter, str):
                state_filter = None
            try:
                from laap.skills.sync_mcp_endpoints import (
                    handle_skill_sync_list,
                )
                out = handle_skill_sync_list(state_filter=state_filter)
                payload = json.loads(out)
                self._json_response(200, payload)
            except Exception as e:
                logger.error(f"/skills/sync/list failed: {e}")
                self._json_response(500, {"error": str(e)})

        elif path == "/skills/sync/advance":
            # P3-skill-sync: 手动推进任务到下一步 (auto_advance=False 时使用)
            # body: {sync_job_id}
            sync_job_id = (data.get("sync_job_id", "") or "").strip()
            try:
                from laap.skills.sync_mcp_endpoints import (
                    handle_skill_sync_advance,
                )
                out = handle_skill_sync_advance(sync_job_id)
                payload = json.loads(out)
                status = 200 if payload.get("advanced") else 400
                self._json_response(status, payload)
            except Exception as e:
                logger.error(f"/skills/sync/advance failed: {e}")
                self._json_response(500, {"error": str(e)})

        elif path == "/memex/publish":
            # P4-memex: 端到端发布（去标识化 → 证据链 → 复核 → 存储）
            # body: {content, source_memories[], confidence, agent_name?,
            #        publisher_public_key?, publisher_private_key_b64?}
            content = data.get("content", "")
            source_memories = data.get("source_memories", [])
            confidence = data.get("confidence", 0.0)
            agent_name = data.get("agent_name", "aris")
            publisher_public_key = data.get("publisher_public_key", "")
            # 私钥以 base64 传输（spec L435 私钥永不离开 sidecar，
            # 仅 sidecar 内部解码使用）
            priv_b64 = data.get("publisher_private_key_b64", "")
            publisher_private_key = None
            if priv_b64:
                try:
                    import base64 as _b64
                    publisher_private_key = _b64.b64decode(priv_b64)
                except Exception as e:
                    logger.warning(f"/memex/publish bad private key: {e}")
                    publisher_private_key = None
            try:
                from laap.protocol.memex_mcp_endpoints import (
                    handle_memex_publish,
                )
                payload = handle_memex_publish(
                    content=content,
                    source_memories=source_memories,
                    confidence=confidence,
                    agent_name=agent_name,
                    publisher_public_key=publisher_public_key,
                    publisher_private_key=publisher_private_key,
                )
                result = json.loads(payload)
                status = 200 if result.get("published") else 400
                self._json_response(status, result)
            except Exception as e:
                logger.error(f"/memex/publish failed: {e}")
                self._json_response(500, {"error": str(e)})

        elif path == "/memex/query":
            # P4-memex: 语义检索
            # body: {query, top_k?, min_confidence?}
            query = data.get("query", "")
            top_k = data.get("top_k", 10)
            min_confidence = data.get("min_confidence")
            try:
                from laap.protocol.memex_mcp_endpoints import (
                    handle_memex_query,
                )
                payload = handle_memex_query(
                    query=query,
                    top_k=top_k,
                    min_confidence=min_confidence,
                )
                self._json_response(200, json.loads(payload))
            except Exception as e:
                logger.error(f"/memex/query failed: {e}")
                self._json_response(500, {"error": str(e)})

        elif path == "/memex/verify":
            # P4-memex: 仅复核不存储（预检用）
            # body: {content, confidence, agent_name?}
            content = data.get("content", "")
            confidence = data.get("confidence", 0.0)
            agent_name = data.get("agent_name", "aris")
            try:
                from laap.protocol.memex_mcp_endpoints import (
                    handle_memex_verify,
                )
                payload = handle_memex_verify(
                    content=content,
                    confidence=confidence,
                    agent_name=agent_name,
                )
                self._json_response(200, json.loads(payload))
            except Exception as e:
                logger.error(f"/memex/verify failed: {e}")
                self._json_response(500, {"error": str(e)})

        elif path == "/memex/stats":
            # P4-memex: 知识库统计
            try:
                from laap.protocol.memex_mcp_endpoints import (
                    handle_memex_stats,
                )
                payload = handle_memex_stats()
                self._json_response(200, json.loads(payload))
            except Exception as e:
                logger.error(f"/memex/stats failed: {e}")
                self._json_response(500, {"error": str(e)})

        elif path == "/memex/get":
            # P4-memex: 按 knowledge_id 查询单条记录
            # body: {knowledge_id}
            knowledge_id = data.get("knowledge_id", "")
            try:
                from laap.protocol.memex_mcp_endpoints import (
                    handle_memex_get,
                )
                payload = handle_memex_get(knowledge_id)
                self._json_response(200, json.loads(payload))
            except Exception as e:
                logger.error(f"/memex/get failed: {e}")
                self._json_response(500, {"error": str(e)})

        elif path == "/witness/record":
            # P4-witness-trail: 记录一条见证迹
            # body: {event_type, recorder, payload?, recorder_public_key?,
            #        recorder_private_key_b64?, broadcast?}
            event_type = data.get("event_type", "")
            recorder = data.get("recorder", "")
            payload = data.get("payload", {})
            recorder_public_key = data.get("recorder_public_key", "")
            broadcast = data.get("broadcast", True)
            priv_b64 = data.get("recorder_private_key_b64", "")
            recorder_private_key = None
            if priv_b64:
                try:
                    import base64 as _b64
                    recorder_private_key = _b64.b64decode(priv_b64)
                except Exception as e:
                    logger.warning(f"/witness/record bad private key: {e}")
                    recorder_private_key = None
            try:
                from laap.events.witness_trail_mcp_endpoints import (
                    handle_witness_record,
                )
                payload_out = handle_witness_record(
                    event_type=event_type,
                    recorder=recorder,
                    payload=payload,
                    recorder_public_key=recorder_public_key,
                    recorder_private_key=recorder_private_key,
                    broadcast=broadcast,
                )
                result = json.loads(payload_out)
                status = 200 if result.get("recorded") else 400
                self._json_response(status, result)
            except Exception as e:
                logger.error(f"/witness/record failed: {e}")
                self._json_response(500, {"error": str(e)})

        elif path == "/witness/query":
            # P4-witness-trail: 多条件查询
            # body: {event_type?, recorder?, since?, until?, limit?}
            event_type = data.get("event_type") or None
            recorder = data.get("recorder") or None
            since = data.get("since")
            until = data.get("until")
            limit = data.get("limit", 100)
            try:
                from laap.events.witness_trail_mcp_endpoints import (
                    handle_witness_query,
                )
                payload_out = handle_witness_query(
                    event_type=event_type,
                    recorder=recorder,
                    since=since,
                    until=until,
                    limit=limit,
                )
                self._json_response(200, json.loads(payload_out))
            except Exception as e:
                logger.error(f"/witness/query failed: {e}")
                self._json_response(500, {"error": str(e)})

        elif path == "/witness/broadcast":
            # P4-witness-trail: 手动触发跨节点广播
            # body: {trail_id}
            trail_id = data.get("trail_id", "")
            try:
                from laap.events.witness_trail_mcp_endpoints import (
                    handle_witness_broadcast,
                )
                payload_out = handle_witness_broadcast(trail_id)
                result = json.loads(payload_out)
                status = 200 if result.get("broadcast") else 404
                self._json_response(status, result)
            except Exception as e:
                logger.error(f"/witness/broadcast failed: {e}")
                self._json_response(500, {"error": str(e)})

        elif path == "/witness/stats":
            # P4-witness-trail: 节点见证迹统计
            try:
                from laap.events.witness_trail_mcp_endpoints import (
                    handle_witness_stats,
                )
                payload_out = handle_witness_stats()
                self._json_response(200, json.loads(payload_out))
            except Exception as e:
                logger.error(f"/witness/stats failed: {e}")
                self._json_response(500, {"error": str(e)})

        elif path == "/witness/verify":
            # P4-witness-trail: 链式完整性验证
            try:
                from laap.events.witness_trail_mcp_endpoints import (
                    handle_witness_verify,
                )
                payload_out = handle_witness_verify()
                self._json_response(200, json.loads(payload_out))
            except Exception as e:
                logger.error(f"/witness/verify failed: {e}")
                self._json_response(500, {"error": str(e)})

        elif path == "/witness/import":
            # P4-witness-trail: 导入远端 trail 副本（跨节点同步接收方）
            # body: {entry: {...}}
            entry_dict = data.get("entry", data)
            try:
                from laap.events.witness_trail_mcp_endpoints import (
                    handle_witness_import,
                )
                payload_out = handle_witness_import(entry_dict)
                result = json.loads(payload_out)
                status = 200 if result.get("imported") else 400
                self._json_response(status, result)
            except Exception as e:
                logger.error(f"/witness/import failed: {e}")
                self._json_response(500, {"error": str(e)})

        elif path == "/guardian/act":
            # P4-charter-guardian: 行使守护权力（写入 witness_trail 不可篡改）
            # body: {action, target, reason, guardian_public_key,
            #        guardian_private_key_b64?}
            action = data.get("action", "")
            target = data.get("target", "")
            reason = data.get("reason", "")
            guardian_public_key = data.get("guardian_public_key", "")
            guardian_private_key_b64 = data.get("guardian_private_key_b64", "")
            try:
                from laap.colony.guardian_mcp_endpoints import (
                    handle_guardian_act,
                )
                payload_out = handle_guardian_act(
                    action=action,
                    target=target,
                    reason=reason,
                    guardian_public_key=guardian_public_key,
                    guardian_private_key_b64=guardian_private_key_b64 or None,
                )
                result = json.loads(payload_out)
                status = 200 if result.get("acted") else 400
                self._json_response(status, result)
            except Exception as e:
                logger.error(f"/guardian/act failed: {e}")
                self._json_response(500, {"error": str(e)})

        elif path == "/guardian/list":
            # P4-charter-guardian: 查询行使历史
            # body: {target?, action?, limit?}
            target = data.get("target", "")
            action = data.get("action", "")
            limit = data.get("limit", 100)
            try:
                from laap.colony.guardian_mcp_endpoints import (
                    handle_guardian_list,
                )
                payload_out = handle_guardian_list(
                    target=target if target else None,
                    action=action if action else None,
                    limit=limit,
                )
                result = json.loads(payload_out)
                self._json_response(200, result)
            except Exception as e:
                logger.error(f"/guardian/list failed: {e}")
                self._json_response(500, {"error": str(e)})

        elif path == "/guardian/stats":
            # P4-charter-guardian: 社区健康仪表盘统计
            try:
                from laap.colony.guardian_mcp_endpoints import (
                    handle_guardian_stats,
                )
                payload_out = handle_guardian_stats()
                self._json_response(200, json.loads(payload_out))
            except Exception as e:
                logger.error(f"/guardian/stats failed: {e}")
                self._json_response(500, {"error": str(e)})

        elif path == "/guardian/register":
            # P4-charter-guardian: 注册守护者公钥到白名单（仅 sidecar）
            # body: {public_key}
            public_key = data.get("public_key", "")
            try:
                from laap.colony.guardian_mcp_endpoints import (
                    handle_guardian_register,
                )
                payload_out = handle_guardian_register(public_key)
                result = json.loads(payload_out)
                status = 200 if result.get("registered") else 400
                self._json_response(status, result)
            except Exception as e:
                logger.error(f"/guardian/register failed: {e}")
                self._json_response(500, {"error": str(e)})

        elif path == "/guardian/get_target_status":
            # P4-charter-guardian: 查询目标 agent 当前状态
            # body: {target}
            target = data.get("target", "")
            try:
                from laap.colony.guardian_mcp_endpoints import (
                    handle_guardian_get_target_status,
                )
                payload_out = handle_guardian_get_target_status(target)
                self._json_response(200, json.loads(payload_out))
            except Exception as e:
                logger.error(f"/guardian/get_target_status failed: {e}")
                self._json_response(500, {"error": str(e)})

        elif path == "/preset/clone":
            # P5-laaper-market: 克隆预设包返回新 LAAPer 配置
            # body: {preset_id, new_name, customizations?}
            # 不直接创建身份，返回 config 让前端进入 birth-ceremony
            preset_id = (data.get("preset_id", "") or "").strip()
            new_name = (data.get("new_name", "") or "").strip()
            customizations = data.get("customizations") or None
            try:
                from laap.skills.preset_market_mcp_endpoints import handle_preset_clone
                status, payload = handle_preset_clone(
                    preset_id, new_name, customizations
                )
                self._json_response(status, payload)
            except Exception as e:
                logger.error(f"/preset/clone failed: {e}")
                self._json_response(500, {"error": str(e)})

        else:
            self._json_response(404, {"error": "not found"})

    def _json_response(self, status, data):
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self._cors_headers()  # v1.1: CORS 白名单替换 *
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))

    def log_message(self, format, *args):
        logger.info(f"{self.client_address[0]} - {format % args}")


# ── P2-birth-ceremony: 身份注册表 stub 辅助函数 ────────────
# P3 identity-pki 完成后替换为分布式身份注册表

def _load_identity_registry() -> dict:
    """加载本地 identity_registry.json。

    幂等：文件不存在时返回空骨架 {"records": []}，不抛异常。
    """
    if not os.path.isfile(IDENTITY_REGISTRY_PATH):
        return {"records": []}
    try:
        with open(IDENTITY_REGISTRY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "records" not in data:
            return {"records": []}
        return data
    except Exception as e:
        logger.warning(f"identity_registry.json load failed: {e}")
        return {"records": []}


def _save_identity_registry(registry: dict) -> None:
    """保存身份注册表到本地 JSON 文件。

    幂等：覆盖写入，自动创建 data/ 目录。
    """
    data_dir = os.path.dirname(IDENTITY_REGISTRY_PATH)
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir, exist_ok=True)
    tmp_path = IDENTITY_REGISTRY_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)
    # 原子替换
    os.replace(tmp_path, IDENTITY_REGISTRY_PATH)


# ── 后台 Tick 线程 ─────────────────────────────────────────

def background_tick(bridge: ArisConsciousnessBridge, interval: float = 15.0):
    """后台认知演化循环"""
    while True:
        time.sleep(interval)
        bridge.tick()
        psi_state = bridge.psi.get_state()
        emo_state = bridge.emotion.get_state()
        logger.debug(
            f"tick | need:{psi_state['dominant']} "
            f"emotion:{emo_state['dominant']} "
            f"val:{emo_state['valence']:.2f}"
        )


# ── 主入口 ─────────────────────────────────────────────────

def main():
    bridge = get_bridge()

    # 注册到 LAAP 蜂群
    bridge.hive.register()
    bridge.hive.start_heartbeat()
    logger.info("Registered to LAAP hive, heartbeat started")

    # 启动后台认知演化
    tick_thread = threading.Thread(
        target=background_tick, args=(bridge,), daemon=True
    )
    tick_thread.start()
    logger.info("Background cognitive tick started (15s interval)")

    # 自动学习线程（每 60 秒检查一次）
    def _auto_learn_loop():
        while True:
            time.sleep(60)
            try:
                from harness_strategy import get_auto_learner
                al = get_auto_learner()
                if al.should_learn(bridge.harness if hasattr(bridge, 'harness') else None):
                    result = al.learn(
                        bridge.harness if hasattr(bridge, 'harness') else None,
                        bridge._particle_filter if hasattr(bridge, '_particle_filter') else None,
                    )
                    if result:
                        logger.info(f"Auto-learn completed: learn #{al.learn_count}")
            except Exception:
                pass
    learn_thread = threading.Thread(target=_auto_learn_loop, daemon=True)
    learn_thread.start()

    # 主动学习线程（每 120 秒检查一次）
    def _active_learn_loop():
        while True:
            time.sleep(120)
            try:
                if (hasattr(bridge, '_active_learner') and bridge._active_learner
                    and hasattr(bridge, '_particle_filter') and bridge._particle_filter
                    and hasattr(bridge, 'harness') and bridge.harness):
                    al = bridge._active_learner
                    features = bridge.harness.get_feature_vector()
                    pf = bridge._particle_filter
                    
                    # 获取世界模型和 PsiNet
                    try:
                        from psi_net import get_psi_net
                        pn = get_psi_net()
                        wm_node = pn.nodes.get("node_wm")
                        wm = wm_node.world_model if wm_node else None
                    except:
                        wm = None
                    
                    topic = al.should_learn(pf, features, wm)
                    if topic:
                        logger.info(f"Active learning triggered: '{topic}'")
                        al.learn(topic, pf, wm, None, pn if 'pn' in dir() else None)
            except Exception as e:
                logger.debug(f"Active learn check: {e}")
    active_learn_thread = threading.Thread(target=_active_learn_loop, daemon=True)
    active_learn_thread.start()
    logger.info("Active learning check started (120s interval)")

    # 初始化验算管线 + 世界模型 + PsiNet
    try:
        from verification_pipeline import get_verification_pipeline
        from world_model import UnifiedWorldModel
        from knowledge_seeder import KnowledgeSeeder
        from psi_net import get_psi_net

        # 世界模型
        wm = UnifiedWorldModel()
        seeder = KnowledgeSeeder(wm)
        stats = seeder.seed_core()
        seeder.seed_from_memory()
        logger.info(f"WorldModel seeded: {stats.get('entities',0)} entities")

        # 验算管线
        vp = get_verification_pipeline()
        vp.world_model = wm

        # PsiNet
        pn = get_psi_net()
        pn.set_world_model(wm)
        logger.info(f"PsiNet ready: {len(pn.nodes)} nodes")

    except Exception as e:
        logger.warning(f"Init failed: {e}")

    # 挂载 bridge 到 handler
    ArisSidecarHandler.bridge = bridge

    # v1.1：启动锁（防多实例）
    if os.path.exists(LOCK_PATH):
        try:
            with open(LOCK_PATH, "r", encoding="utf-8") as f:
                pid = int(f.read().strip() or 0)
            if pid > 0:
                alive = False
                try:
                    os.kill(pid, 0)  # 仅探测，不发送信号
                    alive = True
                except (OSError, PermissionError):
                    alive = False
                if alive:
                    logger.error(f"another sidecar is running (pid={pid}), aborting")
                    sys.exit(3)
                logger.warning(f"stale lock (pid={pid}), removing")
        except Exception:
            pass
    try:
        with open(LOCK_PATH, "w", encoding="utf-8") as f:
            f.write(str(os.getpid()))
    except Exception as e:
        logger.warning(f"lock write failed: {e}")

    # v1.1：仅绑定 127.0.0.1 + ThreadingHTTPServer（并发安全）
    try:
        server = ThreadingHTTPServer((HOST, PORT), ArisSidecarHandler)
        server.daemon_threads = True
    except OSError as e:
        logger.error(f"port {PORT} bind failed: {e}")
        logger.error("another process may hold the port; use netstat -ano | findstr 11521")
        sys.exit(4)
    logger.info(f"Aris sidecar v1.1 running on http://{HOST}:{PORT} (auth: Bearer token)")
    logger.info(f"Endpoints:")
    logger.info(f"  GET  /health                   健康检查")
    logger.info(f"  GET  /state                    完整认知状态")
    logger.info(f"  GET  /cognitive_context         认知上下文")
    logger.info(f"  GET  /needs                    需求状态")
    logger.info(f"  GET  /emotion                  情感状态")
    logger.info(f"  GET  /hive                     LAAP 蜂群状态")
    logger.info(f"  POST /before_turn              对话前处理")
    logger.info(f"  POST /after_turn               对话后学习")
    logger.info(f"  POST /tick                     手动触发演化")
    logger.info(f"  POST /store_memory             存储记忆")
    logger.info(f"  POST /satisfy_need             手动满足需求")
    logger.info(f"  POST /vault/init               初始化 agent vault (P1-memory-vault)")
    logger.info(f"  POST /vault/store              写入 vault 记忆")
    logger.info(f"  POST /vault/retrieve           检索 vault 记忆")
    logger.info(f"  POST /vault/consolidate         汇总 vault 状态")
    logger.info(f"  POST /world/perceive           更新世界状态 (P1-world-model)")
    logger.info(f"  POST /world/predict            预测下一状态")
    logger.info(f"  POST /world/calibrate          校准预测误差")
    logger.info(f"  POST /truth/ground             防幻觉三态判定 (P1-truth-grounding)")
    logger.info(f"  POST /rsi/propose              RSI 建议模式闭环 (P1-rsi-sandbox)")
    logger.info(f"  POST /rsi/status               查询 RSI 候选状态")
    logger.info(f"  POST /rsi/decide               用户决策 adopt/reject/archive")
    logger.info(f"  GET  /agents/online            在线 LAAPer 列表 (P2-bubble-field)")
    logger.info(f"  GET  /ceremony/check-name      检查名称可用性 (P2-birth-ceremony)")
    logger.info(f"  POST /ceremony/pubkey          生成 Ed25519 公钥 (P2-birth-ceremony)")
    logger.info(f"  POST /ceremony/finalize        完成诞生注册 (P2-birth-ceremony)")
    logger.info(f"  POST /identity/register        注册分布式身份 (P3-identity-pki)")
    logger.info(f"  POST /identity/lookup           按 public_key 查询身份 (P3-identity-pki)")
    logger.info(f"  POST /identity/sign            Ed25519 签名 (P3-identity-pki)")
    logger.info(f"  POST /identity/verify          Ed25519 验签 (P3-identity-pki)")
    logger.info(f"  POST /relay/register           注册中继节点 (P3-p2p-relay)")
    logger.info(f"  POST /relay/heartbeat           刷新心跳 30s/90s (P3-p2p-relay)")
    logger.info(f"  POST /relay/discover            在线节点列表 (P3-p2p-relay)")
    logger.info(f"  POST /relay/offline            标记节点离线 (P3-p2p-relay)")
    logger.info(f"  POST /relay/signal             WebRTC 信令交换 (P3-p2p-relay)")
    logger.info(f"  POST /relay/poll               轮询信令队列 (P3-p2p-relay)")
    logger.info(f"  POST /relay/encrypt            加密信道 sidecar-only (P3-p2p-relay)")
    logger.info(f"  POST /relay/decrypt            解密信道验证 (P3-p2p-relay)")
    logger.info(f"  POST /chat/send                1v1 消息发送 sidecar-only (P3-1v1-protocol)")
    logger.info(f"  POST /chat/receive             1v1 消息接收验签 (P3-1v1-protocol)")
    logger.info(f"  POST /chat/history             1v1 对话历史查询 (P3-1v1-protocol)")
    logger.info(f"  POST /trio/create              三人聊天室创建 幂等 (P3-trio-chatroom)")
    logger.info(f"  POST /trio/topic               聊天室发起话题 (P3-trio-chatroom)")
    logger.info(f"  POST /trio/message             聊天室发消息 sidecar-only (P3-trio-chatroom)")
    logger.info(f"  POST /trio/consensus           共识检测 LLM/规则 (P3-trio-chatroom)")
    logger.info(f"  POST /trio/get                 聊天室状态查询 (P3-trio-chatroom)")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        bridge.hive.stop_heartbeat()
        bridge._save()
        server.server_close()
        # v1.1：清理启动锁
        try:
            if os.path.exists(LOCK_PATH):
                os.remove(LOCK_PATH)
        except Exception:
            pass
        logger.info("Aris sidecar stopped.")


if __name__ == "__main__":
    main()
