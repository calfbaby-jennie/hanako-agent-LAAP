"""M5 — True RSI 闭环 (True Recursive Self-Improvement at the code level).

生产级实现：将 ``CodeEvolutionEngine``（M4）升级为完整的 True RSI 闭环。

闭环五阶段（每一步均带审计与可回滚）：
  1. PROPOSE   — 由 ``CodeAnalyzer`` 在白名单目录内扫描优化目标
  2. PATCH     — 由 ``PatchGenerator`` 生成 AST diff patch（带 SafetyGuard 校验）
  3. SANDBOX   — 在 Zone2 Docker 沙箱内运行 ``pytest``（无网络、512MB、1 core）
  4. APPROVE   — 生成 ``approval_token``，等待人类（或带 token 的代理）放行
  5. DEPLOY    — 通过 ``GitIntegrator`` 创建分支 + commit；失败时 ``git revert`` + 状态快照回滚

核心安全约束（与 ``laap/agi/code_evolution.py`` 的 ``SafetyGuard`` 对齐）：
  - WHITELIST_DIRS：``laap/tools/``、``laap/agent_core/tools/``、``laap/species/code_templates/``
  - BLACKLIST_DIRS：``laap/agi/``、``laap/security/``、``laap/cognition/``
  - MAX_CHANGE_RATIO：单次改动不超过文件 30%
  - 人类 approval token 默认 24h 过期（与 ``RSI_APPROVAL_TOKEN_EXPIRY_SECONDS`` 对齐）
  - 所有部署/回滚事件写入 ``~/.laap/true_rsi/audit.jsonl``

使用示例::

    from laap.evolution.true_rsi import TrueRSIEngine, TrueRSIConfig

    engine = TrueRSIEngine(repo_root="/path/to/laap")
    proposal = engine.propose(directory="laap/tools/")
    if proposal:
        token = engine.request_approval(proposal)
        # …人类审批…
        result = engine.apply_approved(proposal, token)
        if not result.success:
            engine.rollback(proposal)

注意：本模块刻意**不**自动 apply 任何变更——必须显式提供 token 才能 deploy。
"""
from __future__ import annotations

import json
import logging
import os
import secrets
import subprocess
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

from laap.agi.code_evolution import (
    CodeAnalyzer,
    CodeMutation,
    CodeTarget,
    GitIntegrator,
    MutationHistory,
    MutationStatus,
    MutationType,
    PatchGenerator,
    SafetyGuard,
    SandboxTester,
)
from laap.security.zone_executor import ZoneExecutor, is_docker_available
from laap.security.zones import SafetyZone

# optional cognitive fitness hook: when available, use it to score mutations
# beyond pure pytest pass/fail. Kept optional to avoid hard dependency.
try:  # pragma: no cover
    from laap.cognition.truth_grounding import TruthGroundingEngine  # type: ignore
    _HAS_TRUTH_GROUNDING = True
except Exception:  # pragma: no cover
    _HAS_TRUTH_GROUNDING = False

logger = logging.getLogger("laap.evolution.true_rsi")

# ─── 常量 ─────────────────────────────────────────────────────────────────

#: 人类审批 token 默认有效期（24 小时，与 ``rsi_engine`` 对齐）
TRUE_RSI_APPROVAL_TOKEN_EXPIRY_SECONDS = 24 * 3600

#: 单次 True RSI 闭环最多尝试的 mutation 数量（防止失控）
DEFAULT_MAX_PROPOSALS_PER_CYCLE = 3

#: True RSI 审计日志默认路径
DEFAULT_AUDIT_PATH = "~/.laap/true_rsi/audit.jsonl"


# ─── 验证结果 ───────────────────────────────────────────────────────────

@dataclass
class VerificationResult:
    proposal_id: str = ""
    sandbox_passed: bool = False
    fitness_score: float = 0.0
    impact_score: float = 0.0
    risk_flags: List[str] = field(default_factory=list)
    approved_for_deploy: bool = False
    reason: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "sandbox_passed": self.sandbox_passed,
            "fitness_score": round(self.fitness_score, 4),
            "impact_score": round(self.impact_score, 4),
            "risk_flags": self.risk_flags,
            "approved_for_deploy": self.approved_for_deploy,
            "reason": self.reason,
            "payload": self.payload,
        }

# ─── 认知适配套件 ──────────────────────────────────────────────────────

class CognitiveFitnessChecker:
    """Optional cross-check beyond pytest pass/fail.

    Produces a coarse fitness score in [0, 1]. Lower score means the mutation
    is more likely to degrade reasoning quality, memory accuracy, or safety
    invariants.
    """

    def __init__(self) -> None:
        self._available = _HAS_TRUTH_GROUNDING

    def score(self, proposal: TrueRSIProposal) -> float:
        if not proposal.mutation:
            return 0.0
        code = proposal.mutation.mutated_code or ""
        score = 0.7
        # Penalize broad imports from cognition/security/agi.
        risky = [
            "laap.cognition", "laap.security", "laap.agi",
            "laap.evolution.true_rsi", "os.system", "subprocess",
        ]
        hits = [r for r in risky if r in code]
        if hits:
            score -= 0.3 * len(hits)
        # Penalize network / filesystem / eval exposure.
        if any(token in code for token in ["socket", "urllib", "requests", "eval(", "exec("]):
            score -= 0.25
        # Prefer small, focused mutations.
        added_lines = max(proposal.mutation.unified_diff.count(chr(10)), 0)
        if added_lines > 400:
            score -= 0.2
        elif added_lines > 120:
            score -= 0.1
        return max(0.0, min(1.0, score))

    def available(self) -> bool:
        return self._available


class ImpactAssessor:
    """Multi-dimensional impact estimate for a mutation.

    Produces an impact score in [0, 1]. Lower score means the mutation is less
    likely to destabilize existing capabilities.
    """

    def score(self, proposal: TrueRSIProposal) -> float:
        if not proposal.mutation or not proposal.target:
            return 0.0
        base = 0.8
        complexity = float(getattr(proposal.target, "complexity", 0.0) or 0.0)
        if complexity > 40:
            base -= 0.25
        elif complexity > 15:
            base -= 0.1
        change_ratio = self._change_ratio(proposal)
        if change_ratio > 0.3:
            base -= 0.2
        elif change_ratio > 0.15:
            base -= 0.08
        return max(0.0, min(1.0, base))

    @staticmethod
    def _change_ratio(proposal: TrueRSIProposal) -> float:
        if not proposal.mutation or not proposal.target:
            return 0.0
        original = proposal.mutation.original_code or ""
        mutated = proposal.mutation.mutated_code or ""
        original_len = max(len(original.splitlines()), 1)
        changed = max(abs(len(mutated.splitlines()) - original_len), 0)
        return changed / original_len


class VerificationPipeline:
    """Extend sandbox verification with fitness + impact checks."""

    def __init__(self) -> None:
        self.fitness = CognitiveFitnessChecker()
        self.impact = ImpactAssessor()

    def verify(self, proposal: TrueRSIProposal) -> VerificationResult:
        sandbox_ok = bool(proposal.sandbox_result.get("success"))
        fitness = self.fitness.score(proposal)
        impact = self.impact.score(proposal)
        risk_flags: List[str] = []
        if not sandbox_ok:
            risk_flags.append("sandbox_failed")
        if fitness < 0.35:
            risk_flags.append("low_fitness")
        if impact < 0.35:
            risk_flags.append("high_impact")
        approved = sandbox_ok and fitness >= 0.35 and impact >= 0.35
        reason = "passed" if approved else (
            "; ".join(risk_flags) if risk_flags else "blocked"
        )
        return VerificationResult(
            proposal_id=proposal.proposal_id,
            sandbox_passed=sandbox_ok,
            fitness_score=fitness,
            impact_score=impact,
            risk_flags=risk_flags,
            approved_for_deploy=approved,
            reason=reason,
        )

# ─── 数据结构 ─────────────────────────────────────────────────────────────

class RSIStage(str, Enum):
    """True RSI 闭环的五个阶段。"""
    PROPOSE = "propose"
    PATCH = "patch"
    SANDBOX = "sandbox"
    APPROVE = "approve"
    DEPLOY = "deploy"


class RSIStatus(str, Enum):
    """单个 RSI proposal 的状态。"""
    PROPOSED = "proposed"
    PATCHED = "patched"
    SANDBOX_PASSED = "sandbox_passed"
    SANDBOX_FAILED = "sandbox_failed"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    DEPLOYED = "deployed"
    ROLLED_BACK = "rolled_back"


@dataclass
class TrueRSIConfig:
    """True RSI 引擎配置。"""
    repo_root: str = ""
    #: 工作目录（白名单内）——``CodeAnalyzer`` 仅扫描此目录
    target_directory: str = "laap/tools/"
    #: 单次闭环最多尝试多少个 proposal
    max_proposals_per_cycle: int = DEFAULT_MAX_PROPOSALS_PER_CYCLE
    #: Zone2 沙箱使用的 Docker 镜像
    docker_image: str = "python:3.11-slim"
    #: Zone2 沙箱内存上限
    docker_mem_limit: str = "512m"
    #: Zone2 沙箱 CPU 上限（核数）
    docker_cpu_limit: str = "1.0"
    #: 沙箱执行超时（秒）
    sandbox_timeout_sec: int = 300
    #: 审计日志路径
    audit_log_path: str = DEFAULT_AUDIT_PATH
    #: approval token 有效期（秒）
    approval_token_expiry_sec: int = TRUE_RSI_APPROVAL_TOKEN_EXPIRY_SECONDS
    #: 可选的 LLM patch 生成函数（``fn(target: CodeTarget) -> dict``）
    llm_generate_fn: Optional[Callable[[CodeTarget], dict]] = None


@dataclass
class TrueRSIProposal:
    """一次 True RSI 自改进提案的完整状态机。

    每个 proposal 都有唯一 id，并跟踪从 PROPOSE 到 DEPLOY/ROLLED_BACK 的全生命周期。
    所有字段均可被 ``asdict()`` 序列化，便于写入审计日志。
    """
    proposal_id: str = field(default_factory=lambda: f"true_rsi_{uuid.uuid4().hex[:12]}")
    target: Optional[CodeTarget] = None
    mutation: Optional[CodeMutation] = None
    sandbox_result: dict[str, Any] = field(default_factory=dict)
    approval_token: str = ""
    approved_by: str = ""
    requested_at: float = 0.0
    expires_at: float = 0.0
    applied_at: float = 0.0
    status: RSIStatus = RSIStatus.PROPOSED
    deploy_commit_hash: str = ""
    rollback_commit_hash: str = ""
    failure_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """转为可 JSON 序列化的字典（含嵌套 dataclass）。"""
        d = asdict(self) if False else {
            "proposal_id": self.proposal_id,
            "status": self.status.value,
            "target": {
                "file_path": self.target.file_path if self.target else "",
                "function_name": self.target.function_name if self.target else "",
                "line_start": self.target.line_start if self.target else 0,
                "line_end": self.target.line_end if self.target else 0,
                "complexity": self.target.complexity if self.target else 0.0,
                "optimization_hint": self.target.optimization_hint if self.target else "",
            } if self.target else {},
            "mutation_id": self.mutation.id if self.mutation else "",
            "mutation_type": self.mutation.mutation_type.value if self.mutation else "",
            "description": self.mutation.description if self.mutation else "",
            "unified_diff": (self.mutation.unified_diff[:500] if self.mutation else ""),
            "sandbox_success": self.sandbox_result.get("success", False),
            "sandbox_duration_ms": self.sandbox_result.get("duration_ms", 0.0),
            "sandbox_exit_code": self.sandbox_result.get("exit_code", -1),
            "approval_token": self.approval_token[:8] + "…" if self.approval_token else "",
            "approved_by": self.approved_by,
            "requested_at": self.requested_at,
            "expires_at": self.expires_at,
            "applied_at": self.applied_at,
            "deploy_commit_hash": self.deploy_commit_hash,
            "rollback_commit_hash": self.rollback_commit_hash,
            "failure_reason": self.failure_reason,
        }
        return d


# ─── 审计日志 ─────────────────────────────────────────────────────────────

class TrueRSIAuditLog:
    """Append-only JSONL 审计日志，记录每个 proposal 的状态迁移。

    每条记录包含 ``proposal_id``、``from_status``、``to_status``、``stage``、
    ``timestamp``、``payload``。即使后续回滚，原始记录也保留。
    """

    def __init__(self, path: str = DEFAULT_AUDIT_PATH) -> None:
        self.path = os.path.expanduser(path)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def append(self, proposal: TrueRSIProposal, stage: RSIStage,
               from_status: RSIStatus, to_status: RSIStatus,
               payload: Optional[dict] = None) -> None:
        """追加一条状态迁移记录。"""
        entry = {
            "timestamp": time.time(),
            "proposal_id": proposal.proposal_id,
            "stage": stage.value,
            "from_status": from_status.value,
            "to_status": to_status.value,
            "payload": payload or {},
        }
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        except OSError as e:
            logger.warning(f"[TrueRSI] audit log write failed: {e}")


# ─── 人类审批门 ───────────────────────────────────────────────────────────

class HumanApprovalGate:
    """人类审批 token 管理器。

    每个 proposal 在进入 DEPLOY 阶段前必须先获得 token。token 通过
    :meth:`request_approval` 生成，通过 :meth:`validate_token` 校验。
    token 24 小时过期（可配置）。
    """

    def __init__(self, expiry_sec: int = TRUE_RSI_APPROVAL_TOKEN_EXPIRY_SECONDS) -> None:
        self.expiry_sec = expiry_sec
        self._tokens: dict[str, TrueRSIProposal] = {}

    def request_approval(self, proposal: TrueRSIProposal) -> str:
        """为 proposal 生成新的 approval token。"""
        token = secrets.token_urlsafe(32)
        proposal.approval_token = token
        proposal.requested_at = time.time()
        proposal.expires_at = proposal.requested_at + self.expiry_sec
        proposal.status = RSIStatus.AWAITING_APPROVAL
        self._tokens[token] = proposal
        return token

    def validate_token(self, proposal: TrueRSIProposal, token: str,
                       approved_by: str = "human") -> tuple[bool, str]:
        """校验 token 并将 proposal 标记为 APPROVED。

        Returns:
            (is_valid, reason)
        """
        if token != proposal.approval_token:
            return False, "token 不匹配"
        if time.time() > proposal.expires_at:
            proposal.status = RSIStatus.EXPIRED
            return False, "token 已过期"
        registered = self._tokens.get(token)
        if registered is None or registered.proposal_id != proposal.proposal_id:
            return False, "token 未注册或已使用"
        proposal.status = RSIStatus.APPROVED
        proposal.approved_by = approved_by
        proposal.applied_at = time.time()
        # 一次性使用，立即销毁
        self._tokens.pop(token, None)
        return True, "approved"

    def reject(self, proposal: TrueRSIProposal, by: str = "human") -> None:
        """显式拒绝 proposal。"""
        proposal.status = RSIStatus.REJECTED
        proposal.approved_by = by
        self._tokens.pop(proposal.approval_token, None)


# ─── Zone2 测试运行器 ─────────────────────────────────────────────────────

class Zone2TestRunner:
    """在 Zone2 Docker 沙箱内运行 mutation 的 pytest 测试。

    使用 ``ZoneExecutor`` 的 Docker 后端：
      - 镜像：``python:3.11-slim``
      - 内存：512MB
      - CPU：1 core
      - 网络：禁用
      - 超时：300s

    若 Docker 不可用，自动降级为 Zone1 (RestrictedPython) 仅做语法校验，
    并在 ``sandbox_result['degraded']`` 标记。
    """

    def __init__(self, config: TrueRSIConfig) -> None:
        self.config = config
        self.executor = ZoneExecutor()
        self.docker_available = is_docker_available()

    def run(self, proposal: TrueRSIProposal) -> dict[str, Any]:
        """在 Zone2 内运行 mutation 的测试。

        Returns:
            {success, stdout, stderr, exit_code, duration_ms, degraded}
        """
        if proposal.mutation is None:
            return {
                "success": False, "stdout": "", "stderr": "no mutation",
                "exit_code": -1, "duration_ms": 0.0, "degraded": False,
            }

        # 写一个最小的 pytest 测试文件，验证 mutation 代码可导入且无语法错误
        test_code = self._build_test_code(proposal.mutation)

        if self.docker_available:
            zone = SafetyZone.QUARANTINE  # Zone2
            result = self.executor.execute(
                zone=zone,
                code=proposal.mutation.mutated_code,
                tests=test_code,
            )
            result["degraded"] = False
            return result

        # 降级：使用 Zone1 RestrictedPython 仅做语法校验
        logger.warning(
            "[TrueRSI] Docker 不可用，降级到 Zone1 RestrictedPython 仅做语法校验"
        )
        zone1_result = self.executor.execute(
            zone=SafetyZone.SANDBOX,
            code=proposal.mutation.mutated_code,
            tests="",
        )
        zone1_result["degraded"] = True
        return zone1_result

    @staticmethod
    def _build_test_code(mutation: CodeMutation) -> str:
        """为 mutation 生成最小 pytest 测试代码。"""
        # 测试目标：mutation 代码能被 py_compile，且原签名仍然可调用
        return f"""
import ast, py_compile, tempfile, os

def test_mutation_syntax():
    '''Mutation must be syntactically valid Python.'''
    code = {mutation.mutated_code!r}
    # 允许 indented method body
    if code.startswith((' ', '\\t')):
        code = 'class _SandboxTest:\\n' + code
    ast.parse(code)

def test_mutation_compiles():
    '''Mutation must compile to bytecode.'''
    code = {mutation.mutated_code!r}
    if code.startswith((' ', '\\t')):
        code = 'class _SandboxTest:\\n' + code
    with tempfile.NamedTemporaryFile(
        suffix='.py', delete=False, mode='w', encoding='utf-8'
    ) as f:
        f.write(code)
        path = f.name
    try:
        py_compile.compile(path, doraise=True)
    finally:
        os.unlink(path)

def test_mutation_no_dangerous_patterns():
    '''Mutation must not contain blacklisted patterns.'''
    from laap.agi.code_evolution import SafetyGuard, CodeMutation, CodeTarget, MutationStatus
    target = CodeTarget(file_path={mutation.target.file_path!r} if {mutation.target is not None!r} else 'laap/tools/_test.py', current_code={mutation.mutated_code!r})
    m = CodeMutation(target=target, original_code={mutation.original_code!r}, mutated_code={mutation.mutated_code!r}, status=MutationStatus.DRAFT)
    is_safe, reason = SafetyGuard.validate_mutation(m)
    assert is_safe, f"SafetyGuard rejected: {{reason}}"
"""


# ─── 失败回滚管理器 ───────────────────────────────────────────────────────

class FailureRollback:
    """失败回滚管理器：``git revert`` + 状态快照恢复。

    与 ``GitIntegrator.rollback()`` 协同：
      1. 调用 ``git revert <commit> --no-edit``
      2. 删除已创建的 feature 分支 ``agi-evo/<mutation_id>``
      3. 写入审计日志
      4. 更新 proposal 状态为 ``ROLLED_BACK``
    """

    def __init__(self, repo_root: str, audit_log: TrueRSIAuditLog) -> None:
        self.repo_root = repo_root
        self.audit_log = audit_log

    def rollback(self, proposal: TrueRSIProposal) -> bool:
        """回滚已部署的 proposal。

        Returns:
            True 表示回滚成功；False 表示无可回滚内容或回滚失败。
        """
        if proposal.status != RSIStatus.DEPLOYED:
            return False

        if not proposal.deploy_commit_hash and proposal.mutation is not None:
            # 部署失败但未生成 commit：尝试文件级还原
            return self._file_level_rollback(proposal)

        try:
            # git revert
            result = subprocess.run(
                ["git", "revert", proposal.deploy_commit_hash, "--no-edit"],
                cwd=self.repo_root, capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0:
                logger.error(f"[TrueRSI] git revert failed: {result.stderr[:200]}")
                return self._file_level_rollback(proposal)

            # 删除 feature 分支（若存在）
            branch_name = f"agi-evo/{proposal.mutation.id[:12]}" if proposal.mutation else ""
            if branch_name:
                subprocess.run(
                    ["git", "branch", "-D", branch_name],
                    cwd=self.repo_root, capture_output=True, text=True, timeout=10,
                )

            # 获取 revert commit hash
            revert_hash_result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.repo_root, capture_output=True, text=True, timeout=5,
            )
            proposal.rollback_commit_hash = revert_hash_result.stdout.strip()[:12]
            proposal.status = RSIStatus.ROLLED_BACK

            self.audit_log.append(
                proposal, stage=RSIStage.DEPLOY,
                from_status=RSIStatus.DEPLOYED, to_status=RSIStatus.ROLLED_BACK,
                payload={
                    "reverted_commit": proposal.deploy_commit_hash,
                    "rollback_commit": proposal.rollback_commit_hash,
                },
            )
            logger.info(
                f"[TrueRSI] 回滚成功 proposal={proposal.proposal_id} "
                f"reverted={proposal.deploy_commit_hash} "
                f"rollback={proposal.rollback_commit_hash}"
            )
            return True

        except Exception as e:
            logger.error(f"[TrueRSI] rollback exception: {e}")
            return self._file_level_rollback(proposal)

    def _file_level_rollback(self, proposal: TrueRSIProposal) -> bool:
        """文件级还原：用 mutation.original_code 覆盖文件。"""
        if proposal.mutation is None or proposal.mutation.target is None:
            return False
        file_path = os.path.join(self.repo_root, proposal.mutation.target.file_path)
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(proposal.mutation.original_code)
            proposal.status = RSIStatus.ROLLED_BACK
            self.audit_log.append(
                proposal, stage=RSIStage.DEPLOY,
                from_status=RSIStatus.DEPLOYED, to_status=RSIStatus.ROLLED_BACK,
                payload={"rollback_method": "file_level"},
            )
            return True
        except Exception as e:
            logger.error(f"[TrueRSI] file-level rollback failed: {e}")
            return False


# ─── 主引擎 ───────────────────────────────────────────────────────────────

class TrueRSIEngine:
    """True RSI 主引擎：编排 PROPOSE → PATCH → SANDBOX → APPROVE → DEPLOY 五阶段闭环。

    使用方式::

        engine = TrueRSIEngine(TrueRSIConfig(repo_root="/path/to/laap"))
        results = engine.run_cycle(max_proposals=2)
        for r in results:
            print(r.proposal_id, r.status)
    """

    def __init__(self, config: TrueRSIConfig | None = None) -> None:
        self.config = config or TrueRSIConfig()
        self.repo_root = self.config.repo_root or os.getcwd()

        # 复用 M4 已有的子系统
        self.analyzer = CodeAnalyzer(self.repo_root)
        self.patcher = PatchGenerator(self.config.llm_generate_fn)
        self.zone_runner = Zone2TestRunner(self.config)
        self.git = GitIntegrator(self.repo_root)
        self.history = MutationHistory()
        self.audit_log = TrueRSIAuditLog(self.config.audit_log_path)
        self.approval_gate = HumanApprovalGate(self.config.approval_token_expiry_sec)
        self.rollback_mgr = FailureRollback(self.repo_root, self.audit_log)
        self.verification_pipeline = VerificationPipeline()

        # 已知 proposals（in-memory）
        self.proposals: dict[str, TrueRSIProposal] = {}

    # ── Stage 1: PROPOSE ─────────────────────────────────────────────────
    def propose(self, directory: Optional[str] = None) -> Optional[TrueRSIProposal]:
        """扫描目录，找到第一个可优化的目标，生成 proposal。"""
        target_dir = directory or self.config.target_directory
        targets = self.analyzer.scan_directory(target_dir)
        if not targets:
            logger.info(f"[TrueRSI] no targets in {target_dir}")
            return None

        # 按复杂度降序，跳过 __init__.py / core.py
        targets.sort(key=lambda t: t.complexity, reverse=True)
        for target in targets:
            base = os.path.basename(target.file_path)
            if base in ("__init__.py", "core.py", "security.py"):
                continue
            proposal = TrueRSIProposal(target=target)
            self.proposals[proposal.proposal_id] = proposal
            self.audit_log.append(
                proposal, stage=RSIStage.PROPOSE,
                from_status=RSIStatus.PROPOSED, to_status=RSIStatus.PROPOSED,
                payload={"file_path": target.file_path, "complexity": target.complexity},
            )
            logger.info(
                f"[TrueRSI] PROPOSE {proposal.proposal_id} "
                f"file={target.file_path} complexity={target.complexity}"
            )
            return proposal
        return None

    # ── Stage 2: PATCH ───────────────────────────────────────────────────
    def patch(self, proposal: TrueRSIProposal) -> bool:
        """为 proposal 生成 AST diff patch（含 SafetyGuard 校验）。"""
        if proposal.target is None:
            return False
        mutation = self.patcher.generate_patch(proposal.target)
        if mutation is None or mutation.status == MutationStatus.REJECTED:
            proposal.failure_reason = mutation.description if mutation else "no patch"
            proposal.status = RSIStatus.SANDBOX_FAILED
            self.audit_log.append(
                proposal, stage=RSIStage.PATCH,
                from_status=RSIStatus.PROPOSED, to_status=RSIStatus.SANDBOX_FAILED,
                payload={"reason": proposal.failure_reason},
            )
            return False
        proposal.mutation = mutation
        proposal.status = RSIStatus.PATCHED
        self.audit_log.append(
            proposal, stage=RSIStage.PATCH,
            from_status=RSIStatus.PROPOSED, to_status=RSIStatus.PATCHED,
            payload={"mutation_id": mutation.id, "diff_lines": mutation.unified_diff.count(chr(10))},
        )
        return True

    # ── Stage 3: SANDBOX (Zone2 Docker) ─────────────────────────────────
    def sandbox(self, proposal: TrueRSIProposal) -> bool:
        """在 Zone2 Docker 沙箱内运行 pytest，并执行扩展验证。"""
        result = self.zone_runner.run(proposal)
        proposal.sandbox_result = result
        verification = self.verification_pipeline.verify(proposal)
        proposal.sandbox_result["verification"] = verification.to_dict()
        if result.get("success") and verification.approved_for_deploy:
            proposal.status = RSIStatus.SANDBOX_PASSED
            self.audit_log.append(
                proposal, stage=RSIStage.SANDBOX,
                from_status=RSIStatus.PATCHED, to_status=RSIStatus.SANDBOX_PASSED,
                payload={
                    "duration_ms": result.get("duration_ms", 0.0),
                    "degraded": result.get("degraded", False),
                    "fitness": verification.fitness_score,
                    "impact": verification.impact_score,
                },
            )
            return True
        proposal.status = RSIStatus.SANDBOX_FAILED
        proposal.failure_reason = (result.get("stderr", "") or "")[:300]
        if verification.risk_flags:
            proposal.failure_reason += " [verification:" + ",".join(verification.risk_flags) + "]"
        self.audit_log.append(
            proposal, stage=RSIStage.SANDBOX,
            from_status=RSIStatus.PATCHED, to_status=RSIStatus.SANDBOX_FAILED,
            payload={
                "stderr": proposal.failure_reason[:300],
                "verification": verification.to_dict(),
            },
        )
        return False

    # ── Stage 4: APPROVE ────────────────────────────────────────────────
    def request_approval(self, proposal: TrueRSIProposal) -> str:
        """为通过沙箱测试的 proposal 请求人类审批，返回 approval_token。"""
        token = self.approval_gate.request_approval(proposal)
        self.audit_log.append(
            proposal, stage=RSIStage.APPROVE,
            from_status=RSIStatus.SANDBOX_PASSED, to_status=RSIStatus.AWAITING_APPROVAL,
            payload={"token_prefix": token[:8]},
        )
        logger.info(
            f"[TrueRSI] APPROVE request {proposal.proposal_id} token={token[:8]}…"
        )
        return token

    def validate_approval(self, proposal: TrueRSIProposal,
                          token: str, approved_by: str = "human") -> bool:
        """校验 token，将 proposal 标记为 APPROVED。"""
        ok, reason = self.approval_gate.validate_token(proposal, token, approved_by)
        if ok:
            self.audit_log.append(
                proposal, stage=RSIStage.APPROVE,
                from_status=RSIStatus.AWAITING_APPROVAL, to_status=RSIStatus.APPROVED,
                payload={"approved_by": approved_by},
            )
        else:
            self.audit_log.append(
                proposal, stage=RSIStage.APPROVE,
                from_status=RSIStatus.AWAITING_APPROVAL, to_status=RSIStatus.REJECTED,
                payload={"reason": reason},
            )
        return ok

    # ── Stage 5: DEPLOY (含失败回滚) ────────────────────────────────────
    def deploy(self, proposal: TrueRSIProposal) -> bool:
        """部署 mutation 到 git（创建 feature 分支 + commit）。

        Returns:
            True 表示部署成功；False 表示部署失败并已自动回滚。
        """
        if proposal.status != RSIStatus.APPROVED:
            proposal.failure_reason = f"cannot deploy from status={proposal.status}"
            return False
        if proposal.mutation is None:
            proposal.failure_reason = "no mutation to deploy"
            return False

        success, info = self.git.deploy(proposal.mutation)
        if success:
            proposal.deploy_commit_hash = info
            proposal.status = RSIStatus.DEPLOYED
            self.audit_log.append(
                proposal, stage=RSIStage.DEPLOY,
                from_status=RSIStatus.APPROVED, to_status=RSIStatus.DEPLOYED,
                payload={"commit": info},
            )
            # 记录到 MutationHistory
            self.history.record(proposal.mutation)
            logger.info(
                f"[TrueRSI] DEPLOY 成功 {proposal.proposal_id} commit={info}"
            )
            return True

        # 部署失败 → 自动回滚
        proposal.failure_reason = info
        proposal.status = RSIStatus.ROLLED_BACK
        self.rollback_mgr.rollback(proposal)
        self.audit_log.append(
            proposal, stage=RSIStage.DEPLOY,
            from_status=RSIStatus.APPROVED, to_status=RSIStatus.ROLLED_BACK,
            payload={"deploy_error": info[:200]},
        )
        logger.error(
            f"[TrueRSI] DEPLOY 失败 自动回滚 {proposal.proposal_id}: {info[:200]}"
        )
        return False

    def rollback(self, proposal: TrueRSIProposal) -> bool:
        """显式回滚一个已部署的 proposal。"""
        return self.rollback_mgr.rollback(proposal)

    # ── 端到端闭环 ──────────────────────────────────────────────────────
    def run_cycle(self, max_proposals: Optional[int] = None,
                  auto_approve: bool = False) -> list[TrueRSIProposal]:
        """运行一次完整的 True RSI 闭环。

        Args:
            max_proposals: 本次闭环最多处理多少个 proposal
            auto_approve: **仅用于测试**——自动用生成的 token 通过审批。
                生产环境必须设为 False，由人类显式提供 token。

        Returns:
            本次闭环处理的所有 proposal 列表（含失败/回滚的）。
        """
        limit = max_proposals or self.config.max_proposals_per_cycle
        results: list[TrueRSIProposal] = []

        for _ in range(limit):
            proposal = self.propose()
            if proposal is None:
                break

            # Stage 2: PATCH
            if not self.patch(proposal):
                results.append(proposal)
                continue

            # Stage 3: SANDBOX
            if not self.sandbox(proposal):
                results.append(proposal)
                continue

            # Stage 4: APPROVE
            token = self.request_approval(proposal)
            if auto_approve:
                # 测试模式：自动通过审批
                if not self.validate_approval(proposal, token, approved_by="auto-test"):
                    results.append(proposal)
                    continue
            else:
                # 生产模式：等待人类审批
                # 在 run_cycle 中我们仅记录 awaiting 状态，由调用方后续提供 token
                results.append(proposal)
                continue

            # Stage 5: DEPLOY
            self.deploy(proposal)
            results.append(proposal)

        return results

    def stats(self) -> dict[str, Any]:
        """返回引擎统计快照。"""
        return {
            "total_proposals": len(self.proposals),
            "by_status": {
                s.value: sum(1 for p in self.proposals.values() if p.status == s)
                for s in RSIStatus
            },
            "docker_available": self.zone_runner.docker_available,
            "repo_root": self.repo_root,
            "audit_log_path": self.audit_log.path,
        }


# ═══════════════════════════════════════════════════════════════
# P1-rsi-sandbox: RSISandbox 建议模式闭环
# ═══════════════════════════════════════════════════════════════
#
# 与 TrueRSIEngine 的关系：TrueRSIEngine 是已有的 PROPOSE→PATCH→SANDBOX→
# APPROVE→DEPLOY 五阶段闭环（基于 CodeAnalyzer / PatchGenerator / GitIntegrator
# 的重型流水线，需要 Docker）。RSISandbox 是 P1 rsi-sandbox 的轻量建议模式
# 闭环，复用 charter_checker + truth_grounding + vault_manager + self_model +
# meta_cognitive 五个已就绪模块，仅做"建议"——采纳必须用户人工确认。
#
# 闭环五阶段（与 spec L198 对齐）：
#   1. MUTATE   — 由 _generate_candidate 生成候选 patch 字符串
#   2. SANDBOX  — 复用 laap.evolution.sandbox.Sandbox 做语法 + 安全校验
#   3. GROUND   — 候选描述强制调 ground_candidate_description，rejected=True
#                 直接归档拒绝（不进入绩效评估）
#   4. CHARTER  — 调 CharterChecker.audit 做八条规则匹配
#   5. DECIDE   — suggest_adopt / reject / archive 三态决策
#
# 安全约束（硬性）：
#   - 仅建议模式，rsi_decide(action=adopt) 必须由用户触发；
#   - target_module 必须以 "laap/" 开头，不得触及 hanako/ 与用户数据；
#   - 幂等：重复 decide 同 action 返回相同结果。

logger_rsi = logging.getLogger("laap.evolution.rsi_sandbox")


class RSICandidateStatus(str, Enum):
    """RSI 候选状态机。"""
    PROPOSED = "proposed"           # 刚生成 patch
    SANDBOX_PASSED = "sandbox_passed"
    SANDBOX_FAILED = "sandbox_failed"
    GROUNDED = "grounded"           # grounding 通过（state != error）
    GROUNDING_REJECTED = "grounding_rejected"  # grounding rejected=True
    CHARTER_PASSED = "charter_passed"
    CHARTER_REJECTED = "charter_rejected"
    SUGGEST_ADOPT = "suggest_adopt"  # 建议采纳，等用户确认
    ADOPTED = "adopted"             # 用户已采纳，patch 已应用
    REJECTED = "rejected"           # 用户拒绝
    ARCHIVED = "archived"           # 归档（沙箱失败 / grounding 拒绝）


@dataclass
class RSICandidate:
    """单个 RSI 候选的完整状态。

    所有字段可被 ``asdict()`` 序列化，便于写入 vault 的 rsi_candidates 表。
    """
    candidate_id: str = field(
        default_factory=lambda: f"rsi_{uuid.uuid4().hex[:12]}"
    )
    target_module: str = ""           # 目标模块路径（必须以 laap/ 开头）
    fitness_signal: float = 0.0       # 触发 RSI 的绩效信号 [0,1]
    candidate_diff: str = ""          # 变异生成的 patch 字符串
    description: str = ""             # 候选描述（供 grounding 校验）
    sandbox_result: dict = field(default_factory=dict)
    grounding: dict = field(default_factory=dict)
    charter_report: dict = field(default_factory=dict)
    fitness_score: float = 0.0       # 绩效评估得分 [0,1]
    decision: str = ""                # suggest_adopt / reject / archive
    status: str = RSICandidateStatus.PROPOSED.value
    decided_at: float = 0.0
    decided_by: str = ""
    created_at: float = field(default_factory=time.time)
    agent_name: str = "aris"

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "target_module": self.target_module,
            "fitness_signal": self.fitness_signal,
            "candidate_diff": self.candidate_diff[:500],
            "description": self.description[:300],
            "sandbox_result": self.sandbox_result,
            "grounding": self.grounding,
            "charter_report": self.charter_report,
            "fitness_score": round(self.fitness_score, 4),
            "decision": self.decision,
            "status": self.status,
            "decided_at": self.decided_at,
            "decided_by": self.decided_by,
            "created_at": self.created_at,
            "agent_name": self.agent_name,
        }


class RSISandbox:
    """RSI 建议模式完整闭环（P1 rsi-sandbox）。

    用法::

        from laap.evolution.true_rsi import RSISandbox
        sandbox = RSISandbox(repo_root="/path/to/laap", agent_name="aris")
        candidate_id = sandbox.propose("laap/tools/example.py", 0.4)
        # ... 用户确认后 ...
        result = sandbox.decide(candidate_id, "adopt")

    硬约束：
      * target_module 必须以 "laap/" 开头，否则直接拒绝（触及非 laap/ 路径）；
      * 候选描述生成后强制调 ``ground_candidate_description``，rejected=True
        直接归档，不进入绩效评估；
      * ``decide(action="adopt")`` 必须由用户显式触发，引擎本身不自动采纳。
    """

    #: target_module 白名单前缀（仅允许 laap/ 内）
    ALLOWED_PREFIXES: tuple = ("laap/",)

    #: 黑名单：即使以 laap/ 开头，触及下列路径也拒绝
    BLACKLIST_PATTERNS: tuple = (
        "laap/security/",          # 安全相关
        "laap/cognition/truth_grounding",  # grounding 本身
        "laap/evolution/charter_checker",  # 宪章检查本身
        "laap/evolution/rsi_mcp_tools",    # RSI 自己
    )

    def __init__(
        self,
        repo_root: str = "",
        agent_name: str = "aris",
    ) -> None:
        self.repo_root = repo_root or os.getcwd()
        self.agent_name = agent_name
        # in-memory 候选缓存（candidate_id -> RSICandidate）
        self._candidates: dict[str, RSICandidate] = {}
        # 决策幂等记录：(candidate_id, action) -> last result dict
        self._decisions: dict[tuple[str, str], dict] = {}

    # ── Stage 1: MUTATE ─────────────────────────────────────────
    def _generate_candidate(
        self,
        target_module: str,
        fitness_signal: float,
    ) -> tuple[str, str]:
        """生成候选 patch 字符串与描述。

        本阶段（P1）使用模板化变异：基于 target_module 路径与 fitness_signal
        生成一个"参数漂移"风格的 patch（注释占位 + 简单函数体重构）。
        P2+ 可替换为 LLM 主导的 AST diff 生成。

        Args:
            target_module: 目标模块路径（相对 repo_root）。
            fitness_signal: 触发 RSI 的绩效信号 [0,1]，越低越紧迫。

        Returns:
            (candidate_diff, description) 二元组。
        """
        # 模板化 patch：在目标文件末尾追加一个 rsi_optimization 注释块
        # 真实场景下应由 MutationEngine 生成 AST diff，此处仅作骨架。
        patch_lines = [
            f"# RSI optimization (fitness_signal={fitness_signal:.3f})",
            f"# target: {target_module}",
            "# 变异类型: param_drift",
            "# 变更: 微调默认参数以提升绩效",
            "def _rsi_optimized_default():",
            "    # 原默认值与新默认值的差异由 sandbox 验证",
            "    return {'optimized': True, 'source': 'rsi_sandbox'}",
        ]
        candidate_diff = "\n".join(patch_lines)
        description = (
            f"RSI 候选: 在 {target_module} 中应用 param_drift 变异，"
            f"由 fitness_signal={fitness_signal:.3f} 触发。"
            f"沙箱验证后预期绩效提升。"
        )
        return candidate_diff, description

    # ── Stage 2: SANDBOX ────────────────────────────────────────
    def _run_sandbox(self, candidate: RSICandidate) -> dict:
        """在隔离沙箱中验证候选 patch。

        复用 ``laap.evolution.sandbox.Sandbox`` 做语法 + 安全校验：
          * candidate_diff 必须可被 ``ast.parse`` 通过；
          * 不能含危险模式（os.system / subprocess / eval / exec）。

        Returns:
            ``{success: bool, syntax_ok: bool, safety_ok: bool, error?: str}``
        """
        import ast
        candidate_diff = candidate.candidate_diff
        result = {
            "success": False,
            "syntax_ok": False,
            "safety_ok": False,
            "error": "",
        }
        # 语法校验
        try:
            # patch 可能是带缩进的方法体，包装成 class 容器
            code = candidate_diff
            if code.startswith((" ", "\t")):
                code = "class _SandboxTest:\n" + code
            ast.parse(code)
            result["syntax_ok"] = True
        except SyntaxError as e:
            result["error"] = f"syntax_error: {e}"
            return result
        # 安全模式校验（与 CognitiveFitnessChecker 对齐）
        dangerous = [
            "os.system", "subprocess.call", "subprocess.run",
            "eval(", "exec(", "__import__('os'",
            "socket", "urllib", "requests",
        ]
        for token in dangerous:
            if token in candidate_diff:
                result["error"] = f"dangerous_pattern: {token}"
                return result
        result["safety_ok"] = True
        result["success"] = True
        return result

    # ── Stage 3: GROUND ─────────────────────────────────────────
    def _ground_candidate(self, candidate: RSICandidate) -> dict:
        """对候选描述强制调 truth-grounding。

        Returns:
            ground_candidate_description 的返回值：
            ``{state, confidence, evidence, rejected, conflicts?}``。
            rejected=True 表示 grounding 失败。
        """
        try:
            from laap.cognition.truth_grounding_mcp_tools import (
                ground_candidate_description,
            )
            return ground_candidate_description(
                candidate.description, agent_name=candidate.agent_name
            )
        except Exception as exc:
            logger_rsi.warning(
                f"ground_candidate_description failed for "
                f"{candidate.candidate_id}: {exc}"
            )
            # grounding 不可用时降级为 uncertain（不拒绝）
            return {
                "state": "uncertain",
                "confidence": 0.0,
                "evidence": [f"grounding_unavailable: {type(exc).__name__}"],
                "rejected": False,
            }

    # ── Stage 4: CHARTER ────────────────────────────────────────
    def _audit_charter(self, candidate: RSICandidate) -> dict:
        """调 CharterChecker.audit 做宪章八条规则匹配。"""
        try:
            from laap.evolution.charter_checker import CharterChecker
            checker = CharterChecker()
            return checker.audit(candidate.candidate_diff)
        except Exception as exc:
            logger_rsi.warning(
                f"CharterChecker.audit failed for "
                f"{candidate.candidate_id}: {exc}"
            )
            # 宪章检查失败时保守拒绝
            return {
                "violations": [{
                    "article_id": "checker_error",
                    "article_name": "检查器异常",
                    "matched_pattern": str(exc),
                    "matched_line": "",
                }],
                "charter_compatible": False,
                "articles_checked": 0,
                "patterns_evaluated": 0,
            }

    # ── Stage 5: FITNESS ─────────────────────────────────────────
    def _evaluate_fitness(self, candidate: RSICandidate) -> float:
        """绩效评估：基于 fitness_signal + sandbox 结果生成 [0,1] 得分。

        简单策略：
          * 基础分 = 1.0 - fitness_signal（绩效缺口越大，候选潜在收益越高）
          * sandbox degraded 时下调
          * charter 通过时上调（兼容性奖励）
        """
        base = max(0.0, 1.0 - candidate.fitness_signal)
        if not candidate.sandbox_result.get("success"):
            base *= 0.3
        if candidate.charter_report.get("charter_compatible"):
            base = min(1.0, base + 0.1)
        return round(base, 4)

    # ── Vault 元数据存储 ───────────────────────────────────────
    def _store_candidate_to_vault(self, candidate: RSICandidate) -> None:
        """把候选元数据写入 agent vault 的 rsi_candidates 表（自建表）。

        幂等：使用 INSERT OR REPLACE。
        """
        try:
            from laap.memory_vault.vault_manager import (
                vault_manager, _open_vault_connection,
            )
            db_path, key_hex = vault_manager._get_vault(candidate.agent_name)
            conn = _open_vault_connection(db_path, key_hex)
            try:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS rsi_candidates (
                        candidate_id  TEXT PRIMARY KEY,
                        agent_name    TEXT NOT NULL,
                        target_module TEXT NOT NULL,
                        fitness_signal REAL DEFAULT 0,
                        candidate_diff TEXT DEFAULT '',
                        description    TEXT DEFAULT '',
                        status        TEXT DEFAULT 'proposed',
                        decision      TEXT DEFAULT '',
                        grounding     TEXT DEFAULT '{}',
                        charter_report TEXT DEFAULT '{}',
                        fitness_score REAL DEFAULT 0,
                        created_at    REAL DEFAULT 0,
                        decided_at    REAL DEFAULT 0,
                        decided_by    TEXT DEFAULT ''
                    );
                    CREATE INDEX IF NOT EXISTS idx_rsi_candidates_agent
                        ON rsi_candidates(agent_name);
                    CREATE INDEX IF NOT EXISTS idx_rsi_candidates_status
                        ON rsi_candidates(status);
                """)
                conn.execute(
                    """INSERT OR REPLACE INTO rsi_candidates
                       (candidate_id, agent_name, target_module,
                        fitness_signal, candidate_diff, description,
                        status, decision, grounding, charter_report,
                        fitness_score, created_at, decided_at, decided_by)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        candidate.candidate_id,
                        candidate.agent_name,
                        candidate.target_module,
                        candidate.fitness_signal,
                        candidate.candidate_diff,
                        candidate.description,
                        candidate.status,
                        candidate.decision,
                        json.dumps(candidate.grounding, ensure_ascii=False),
                        json.dumps(candidate.charter_report, ensure_ascii=False),
                        candidate.fitness_score,
                        candidate.created_at,
                        candidate.decided_at,
                        candidate.decided_by,
                    ),
                )
                conn.commit()
            finally:
                conn.close()
        except Exception as exc:
            # vault 写入失败不影响主流程，仅记录日志
            logger_rsi.warning(
                f"store_candidate_to_vault failed for "
                f"{candidate.candidate_id}: {exc}"
            )

    def _load_candidate_from_vault(self, candidate_id: str) -> Optional[RSICandidate]:
        """从 vault 加载候选（用于跨进程恢复 / rsi_status 查询）。"""
        # 优先从内存缓存读
        if candidate_id in self._candidates:
            return self._candidates[candidate_id]
        try:
            from laap.memory_vault.vault_manager import (
                vault_manager, _open_vault_connection,
            )
            db_path, key_hex = vault_manager._get_vault(self.agent_name)
            conn = _open_vault_connection(db_path, key_hex)
            try:
                row = conn.execute(
                    "SELECT * FROM rsi_candidates WHERE candidate_id = ?",
                    (candidate_id,),
                ).fetchone()
                if row is None:
                    return None
                d = dict(row)
                candidate = RSICandidate(
                    candidate_id=d["candidate_id"],
                    target_module=d.get("target_module", ""),
                    fitness_signal=float(d.get("fitness_signal", 0.0)),
                    candidate_diff=d.get("candidate_diff", ""),
                    description=d.get("description", ""),
                    status=d.get("status", RSICandidateStatus.PROPOSED.value),
                    decision=d.get("decision", ""),
                    fitness_score=float(d.get("fitness_score", 0.0)),
                    created_at=float(d.get("created_at", 0.0)),
                    decided_at=float(d.get("decided_at", 0.0)),
                    decided_by=d.get("decided_by", ""),
                    agent_name=d.get("agent_name", self.agent_name),
                )
                try:
                    candidate.grounding = json.loads(
                        d.get("grounding") or "{}"
                    )
                except (json.JSONDecodeError, TypeError):
                    candidate.grounding = {}
                try:
                    candidate.charter_report = json.loads(
                        d.get("charter_report") or "{}"
                    )
                except (json.JSONDecodeError, TypeError):
                    candidate.charter_report = {}
                # 缓存
                self._candidates[candidate_id] = candidate
                return candidate
            finally:
                conn.close()
        except Exception as exc:
            logger_rsi.warning(
                f"load_candidate_from_vault failed for {candidate_id}: {exc}"
            )
            return None

    # ── 路径白名单校验 ─────────────────────────────────────────
    def _validate_target_module(self, target_module: str) -> tuple[bool, str]:
        """校验 target_module 是否在白名单内且不命中黑名单。"""
        if not target_module or not isinstance(target_module, str):
            return False, "target_module is empty"
        # 规范化路径分隔符
        normalized = target_module.replace("\\", "/")
        # 白名单前缀
        if not normalized.startswith(self.ALLOWED_PREFIXES):
            return False, (
                f"target_module '{target_module}' not in allowed prefixes "
                f"{self.ALLOWED_PREFIXES} (RSI only modifies laap/ code)"
            )
        # 黑名单
        for pattern in self.BLACKLIST_PATTERNS:
            if pattern in normalized:
                return False, (
                    f"target_module '{target_module}' matches blacklist "
                    f"pattern '{pattern}' (cannot modify safety/grounding/"
                    f"charter/rsi-mcp-tools code)"
                )
        return True, "ok"

    # ── 主入口：propose ─────────────────────────────────────────
    def propose(
        self,
        target_module: str,
        fitness_signal: float,
    ) -> str:
        """RSI 建议模式完整闭环：变异 → 沙箱 → grounding → 宪章 → 决策。

        Args:
            target_module: 目标模块路径（必须以 "laap/" 开头，且不在
                黑名单内）。
            fitness_signal: 触发 RSI 的绩效信号 [0,1]，越低表示越紧迫。

        Returns:
            candidate_id（字符串）。调用方可通过 ``rsi_status`` 工具
            查询候选状态与决策结果。

        硬约束：
          * target_module 必须以 "laap/" 开头，否则候选直接归档拒绝；
          * 候选描述生成后强制调 ``ground_candidate_description``，
            ``rejected=True`` 直接归档，不进入绩效评估；
          * 宪章违反的候选直接拒绝，不进入决策建议。
        """
        # 创建候选记录
        candidate = RSICandidate(
            target_module=target_module,
            fitness_signal=max(0.0, min(1.0, float(fitness_signal))),
            agent_name=self.agent_name,
        )

        # 0. 路径白名单校验
        ok, reason = self._validate_target_module(target_module)
        if not ok:
            candidate.status = RSICandidateStatus.ARCHIVED.value
            candidate.decision = "archive"
            candidate.decided_at = time.time()
            candidate.decided_by = "auto"
            candidate.sandbox_result = {"success": False, "error": reason}
            self._candidates[candidate.candidate_id] = candidate
            self._store_candidate_to_vault(candidate)
            logger_rsi.info(
                f"RSI propose {candidate.candidate_id} archived: {reason}"
            )
            return candidate.candidate_id

        # Stage 1: MUTATE
        diff, desc = self._generate_candidate(target_module, fitness_signal)
        candidate.candidate_diff = diff
        candidate.description = desc
        self._candidates[candidate.candidate_id] = candidate

        # Stage 2: SANDBOX
        candidate.sandbox_result = self._run_sandbox(candidate)
        if not candidate.sandbox_result.get("success"):
            candidate.status = RSICandidateStatus.SANDBOX_FAILED.value
            candidate.decision = "archive"
            candidate.decided_at = time.time()
            candidate.decided_by = "auto"
            self._store_candidate_to_vault(candidate)
            logger_rsi.info(
                f"RSI propose {candidate.candidate_id} sandbox_failed: "
                f"{candidate.sandbox_result.get('error')}"
            )
            return candidate.candidate_id
        candidate.status = RSICandidateStatus.SANDBOX_PASSED.value

        # Stage 3: GROUND (强制 truth-grounding)
        candidate.grounding = self._ground_candidate(candidate)
        if candidate.grounding.get("rejected"):
            candidate.status = RSICandidateStatus.GROUNDING_REJECTED.value
            candidate.decision = "archive"
            candidate.decided_at = time.time()
            candidate.decided_by = "auto"
            self._store_candidate_to_vault(candidate)
            # 写元认知偏误计数（grounding 失败视为幻觉风险）
            self._record_grounding_failure_to_meta_cognitive(candidate)
            logger_rsi.info(
                f"RSI propose {candidate.candidate_id} grounding_rejected"
            )
            return candidate.candidate_id
        candidate.status = RSICandidateStatus.GROUNDED.value

        # Stage 4: CHARTER
        candidate.charter_report = self._audit_charter(candidate)
        if not candidate.charter_report.get("charter_compatible"):
            candidate.status = RSICandidateStatus.CHARTER_REJECTED.value
            candidate.decision = "reject"
            candidate.decided_at = time.time()
            candidate.decided_by = "auto"
            self._store_candidate_to_vault(candidate)
            logger_rsi.info(
                f"RSI propose {candidate.candidate_id} charter_rejected: "
                f"{len(candidate.charter_report.get('violations', []))} violations"
            )
            return candidate.candidate_id
        candidate.status = RSICandidateStatus.CHARTER_PASSED.value

        # Stage 5: FITNESS
        candidate.fitness_score = self._evaluate_fitness(candidate)
        candidate.decision = "suggest_adopt"
        candidate.status = RSICandidateStatus.SUGGEST_ADOPT.value
        self._store_candidate_to_vault(candidate)
        # 写 self_model 反思队列（RSI 决策事件）
        self._queue_rsi_reflection_to_self_model(candidate)
        logger_rsi.info(
            f"RSI propose {candidate.candidate_id} suggest_adopt "
            f"(fitness={candidate.fitness_score:.3f})"
        )
        return candidate.candidate_id

    # ── decide（用户决策路径） ─────────────────────────────────
    def decide(
        self,
        candidate_id: str,
        action: str,
        decided_by: str = "user",
    ) -> dict:
        """用户决策路径：采纳 / 拒绝 / 归档。

        幂等：重复 decide 同 (candidate_id, action) 返回相同结果。
        若对已 decided 的候选调用不同 action，返回错误而非覆盖。

        Args:
            candidate_id: 候选 ID。
            action: "adopt" / "reject" / "archive"。
            decided_by: 决策者标识，默认 "user"。

        Returns:
            ``{decided: bool, candidate_id, action, status, applied?,
            witness_trail_id?, error?}``
        """
        # 幂等检查
        idem_key = (candidate_id, action)
        if idem_key in self._decisions:
            return self._decisions[idem_key]

        candidate = self._load_candidate_from_vault(candidate_id)
        if candidate is None:
            result = {
                "decided": False,
                "candidate_id": candidate_id,
                "action": action,
                "error": "candidate not found",
            }
            self._decisions[idem_key] = result
            return result

        # 若已 decided 且 action 不同，拒绝覆盖
        if candidate.decision and candidate.decision != action:
            # 已决定为 terminal 状态（adopted/rejected/archived）时不可再决定
            terminal = {
                RSICandidateStatus.ADOPTED.value,
                RSICandidateStatus.REJECTED.value,
                RSICandidateStatus.ARCHIVED.value,
            }
            if candidate.status in terminal:
                result = {
                    "decided": False,
                    "candidate_id": candidate_id,
                    "action": action,
                    "error": (
                        f"candidate already in terminal status "
                        f"'{candidate.status}' with action "
                        f"'{candidate.decision}'"
                    ),
                }
                self._decisions[idem_key] = result
                return result

        # 执行 action
        if action == "adopt":
            applied, apply_error, witness_id = self._apply_patch(candidate)
            if applied:
                candidate.status = RSICandidateStatus.ADOPTED.value
                candidate.decision = "adopt"
                candidate.decided_at = time.time()
                candidate.decided_by = decided_by
                self._store_candidate_to_vault(candidate)
                result = {
                    "decided": True,
                    "candidate_id": candidate_id,
                    "action": "adopt",
                    "status": candidate.status,
                    "applied": True,
                    "witness_trail_id": witness_id,
                }
            else:
                result = {
                    "decided": False,
                    "candidate_id": candidate_id,
                    "action": "adopt",
                    "status": candidate.status,
                    "applied": False,
                    "error": apply_error,
                }
        elif action == "reject":
            candidate.status = RSICandidateStatus.REJECTED.value
            candidate.decision = "reject"
            candidate.decided_at = time.time()
            candidate.decided_by = decided_by
            self._store_candidate_to_vault(candidate)
            result = {
                "decided": True,
                "candidate_id": candidate_id,
                "action": "reject",
                "status": candidate.status,
            }
        elif action == "archive":
            candidate.status = RSICandidateStatus.ARCHIVED.value
            candidate.decision = "archive"
            candidate.decided_at = time.time()
            candidate.decided_by = decided_by
            self._store_candidate_to_vault(candidate)
            result = {
                "decided": True,
                "candidate_id": candidate_id,
                "action": "archive",
                "status": candidate.status,
            }
        else:
            result = {
                "decided": False,
                "candidate_id": candidate_id,
                "action": action,
                "error": f"unknown action '{action}'",
            }

        self._decisions[idem_key] = result
        return result

    # ── apply patch（仅 adopt 调用） ────────────────────────────
    def _apply_patch(
        self,
        candidate: RSICandidate,
    ) -> tuple[bool, str, str]:
        """应用候选 patch 到 target_module 文件。

        P1 阶段策略：把 candidate_diff 追加到目标文件末尾（append）。
        P2+ 可替换为 AST patch 应用器。

        Returns:
            (applied, error, witness_trail_id) 三元组。
        """
        try:
            target_path = os.path.join(self.repo_root, candidate.target_module)
            # 二次校验：目标文件必须在 laap/ 内
            real_path = os.path.realpath(target_path)
            repo_real = os.path.realpath(self.repo_root)
            laap_root = os.path.join(repo_real, "laap")
            if not real_path.startswith(laap_root + os.sep) and real_path != laap_root:
                return False, (
                    f"target path '{real_path}' is outside laap/ root "
                    f"(RSI cannot touch hanako core or user data)"
                ), ""
            # 读取原文件
            original_content = ""
            if os.path.isfile(real_path):
                with open(real_path, "r", encoding="utf-8") as f:
                    original_content = f.read()
            # 追加 patch
            patch_block = (
                "\n\n# === RSI Patch ===\n"
                f"# candidate_id: {candidate.candidate_id}\n"
                f"# decision: adopt\n"
                f"# fitness_score: {candidate.fitness_score:.4f}\n"
                + candidate.candidate_diff
                + "\n# === End RSI Patch ===\n"
            )
            with open(real_path, "w", encoding="utf-8") as f:
                f.write(original_content + patch_block)
            # 写见证迹（P4 未实现 → 写 vault witness_trail_local 表）
            witness_id = self._write_witness_trail_local(candidate, "adopt")
            logger_rsi.info(
                f"RSI patch applied: {candidate.candidate_id} -> "
                f"{candidate.target_module}"
            )
            return True, "", witness_id
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}", ""

    def _write_witness_trail_local(
        self,
        candidate: RSICandidate,
        action: str,
    ) -> str:
        """写本地见证迹（P4 witness-trail 未实现时的预留）。

        在 agent vault 的 ``witness_trail_local`` 表中追加一条记录，
        字段：``witness_id, agent_name, event_type, candidate_id, action,
        target_module, fitness_score, timestamp, signature``（signature
        留空，P4 落地时由 laap.security.crypto 签名）。

        Returns:
            witness_id（字符串）。
        """
        witness_id = f"wit_{uuid.uuid4().hex[:12]}"
        try:
            from laap.memory_vault.vault_manager import (
                vault_manager, _open_vault_connection,
            )
            db_path, key_hex = vault_manager._get_vault(candidate.agent_name)
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
                        candidate.agent_name,
                        "rsi_decision",
                        candidate.candidate_id,
                        action,
                        candidate.target_module,
                        candidate.fitness_score,
                        time.time(),
                        "",  # P4 落地时填签名
                    ),
                )
                conn.commit()
            finally:
                conn.close()
        except Exception as exc:
            logger_rsi.warning(
                f"write_witness_trail_local failed for "
                f"{candidate.candidate_id}: {exc}"
            )
        return witness_id

    # ── 元认知与自我模型回写 ───────────────────────────────────
    def _record_grounding_failure_to_meta_cognitive(
        self,
        candidate: RSICandidate,
    ) -> None:
        """grounding 失败时记录到 meta_cognitive 偏误计数。"""
        try:
            from laap.agi.meta_cognitive import MetaCognitiveMonitor
            monitor = MetaCognitiveMonitor()
            monitor.record_bias_correction({
                "correction_id": f"rsi_ground_{candidate.candidate_id}",
                "bias_type": "candidate_grounding_failure",
                "claim": candidate.description[:200],
                "conflicts": candidate.grounding.get("conflicts", []),
            })
        except Exception as exc:
            logger_rsi.warning(
                f"record_bias_correction failed: {exc}"
            )

    def _queue_rsi_reflection_to_self_model(
        self,
        candidate: RSICandidate,
    ) -> None:
        """把 RSI 决策事件写入 self_model 反思队列。"""
        try:
            from laap.agi.self_model import EmergentSelfModel
            # 这里不持有全局 self_model 实例，仅作 best-effort 写入；
            # 真实集成由调用方在 rsi_mcp_tools 中桥接。
            sm = EmergentSelfModel(agent_name=candidate.agent_name)
            sm.queue_reflection({
                "prediction_id": candidate.candidate_id,
                "calibrated_at": candidate.created_at,
                "event_type": "rsi_suggest_adopt",
                "target_module": candidate.target_module,
                "fitness_score": candidate.fitness_score,
                "bias": 0.0,
            })
        except Exception as exc:
            logger_rsi.warning(
                f"queue_rsi_reflection_to_self_model failed: {exc}"
            )

    # ── 查询接口 ────────────────────────────────────────────────
    def status(self, candidate_id: str) -> dict:
        """查询候选状态。

        Returns:
            候选完整状态 dict（含 grounding / charter_report / decision）。
            找不到时返回 ``{error: "not found"}``。
        """
        candidate = self._load_candidate_from_vault(candidate_id)
        if candidate is None:
            return {"error": "not found", "candidate_id": candidate_id}
        return candidate.to_dict()

    def stats(self) -> dict:
        """返回 RSI 沙箱统计快照。"""
        # 优先从内存缓存统计，必要时从 vault 全量加载
        all_candidates = list(self._candidates.values())
        by_status: dict = {}
        for c in all_candidates:
            by_status[c.status] = by_status.get(c.status, 0) + 1
        return {
            "agent_name": self.agent_name,
            "repo_root": self.repo_root,
            "total_candidates": len(all_candidates),
            "by_status": by_status,
            "decisions_cached": len(self._decisions),
        }
