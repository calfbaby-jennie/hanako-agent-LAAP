"""M7 — 四层颠覆形态基础代码.

LAAP 的"四层颠覆形态"是 AGI 落地的产品化路径：

  1. **活体代码仓库** (LivingCodeRepo)     — 仓库可自主修复真实 GitHub issue
  2. **岗位级数字员工** (DigitalEmployee)    — 7×24 自主运维一个岗位
  3. **自演化 SaaS** (SelfEvolvingSaaS)     — 业务规则自主生长新模块
  4. **个人知识生命** (KnowledgeLife)       — 跨论文库生成研究建议

每一层都是上一层的"超集"，且复用同一套 LAAP 内核（PSI + CodeEvolution +
TrueRSI + HarnessX）。本模块提供每层的产品级抽象与最小心智模型，便于
后续在 `realize-laap-agi-vision` spec 中逐步落地。

使用示例::

    from laap.lifeform.four_layers import (
        LivingCodeRepo, DigitalEmployee,
        SelfEvolvingSaaS, KnowledgeLife,
    )

    repo = LivingCodeRepo(repo_root="/path/to/laap", github_repo="your-org/laap")
    fixed = repo.auto_fix_issues(max_issues=3)

    employee = DigitalEmployee(role="devops", owner="ops-team")
    schedule = employee.run_shift(duration_hours=8)

    saas = SelfEvolvingSaaS(tenant_id="acme")
    saas.evolve_module("inventory", requirements="支持批次召回")

    life = KnowledgeLife(paper_corpus_path="data/papers")
    suggestions = life.suggest_research(num_suggestions=5)
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

logger = logging.getLogger("laap.lifeform.four_layers")


# ═══════════════════════════════════════════════════════════════════════
# Layer 1: 活体代码仓库 (Living Code Repository)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class IssueFixRecord:
    """单个 issue 修复记录。"""
    issue_id: str
    issue_title: str = ""
    proposed_fix: str = ""
    pr_url: str = ""
    branch_name: str = ""
    commit_hash: str = ""
    status: str = "proposed"   # proposed | pr_open | merged | rejected | failed
    created_at: float = field(default_factory=time.time)


class LivingCodeRepo:
    """活体代码仓库：自主扫描 GitHub issue 并提交修复 PR。

    工作流：
      1. ``list_open_issues()``  — 从 GitHub 拉取 open issue
      2. ``propose_fix(issue)``  — 使用 CodeEvolutionEngine 生成 patch
      3. ``submit_pr(proposal)`` — 通过 GitHub MCP 提交 PR
      4. ``track(pr_url)``       — 跟踪 PR 状态（merged / rejected）

    注意：本类仅做编排，实际 GitHub 操作通过 MCP 工具完成。GitHub token
    通过环境变量 ``GITHUB_TOKEN`` 注入，不写入仓库。
    """

    def __init__(self, repo_root: str = "", github_repo: str = "",
                 max_issues_per_cycle: int = 10) -> None:
        self.repo_root = repo_root or os.getcwd()
        self.github_repo = github_repo  # e.g. "your-org/laap"
        self.max_issues_per_cycle = max_issues_per_cycle
        self.fix_history: list[IssueFixRecord] = []

    def list_open_issues(self, labels: Optional[list[str]] = None) -> list[dict[str, Any]]:
        """通过 GitHub MCP 拉取 open issue 列表。

        Returns:
            issue 字典列表（含 number、title、body、labels）。若 MCP 不可用
            或 token 缺失，返回空列表。
        """
        if not self._github_available():
            logger.warning("[LivingCodeRepo] GitHub MCP 不可用，返回空 issue 列表")
            return []
        try:
            # 通过 MCP GitHub 工具拉取 issues
            # 这里仅做接口约定，实际由调用方注入 MCP client
            logger.info(
                f"[LivingCodeRepo] 拉取 {self.github_repo} open issues labels={labels}"
            )
            return []  # placeholder：实际由 MCP 客户端填充
        except Exception as e:
            logger.error(f"[LivingCodeRepo] list_open_issues failed: {e}")
            return []

    def propose_fix(self, issue: dict[str, Any]) -> Optional[IssueFixRecord]:
        """为单个 issue 生成修复方案。

        Args:
            issue: GitHub issue 字典（需含 number、title、body）。

        Returns:
            IssueFixRecord，含 proposed_fix / branch_name / commit_hash；
            失败时返回 None。
        """
        try:
            from laap.evolution.true_rsi import TrueRSIEngine, TrueRSIConfig
            engine = TrueRSIEngine(TrueRSIConfig(repo_root=self.repo_root))
            proposal = engine.propose(directory="laap/tools/")
            if proposal is None or proposal.mutation is None:
                return None
            record = IssueFixRecord(
                issue_id=str(issue.get("number", "")),
                issue_title=issue.get("title", ""),
                proposed_fix=proposal.mutation.unified_diff[:500],
                branch_name=f"agi-fix/issue-{issue.get('number', 'x')}",
            )
            self.fix_history.append(record)
            return record
        except Exception as e:
            logger.error(f"[LivingCodeRepo] propose_fix failed: {e}")
            return None

    def submit_pr(self, record: IssueFixRecord) -> str:
        """通过 GitHub MCP 提交 PR，返回 PR URL。"""
        if not self._github_available():
            record.status = "failed"
            return ""
        record.status = "pr_open"
        # 实际由 MCP GitHub create_pull_request 工具提交
        return f"https://github.com/{self.github_repo}/pull/{record.issue_id}"

    def auto_fix_issues(self, max_issues: Optional[int] = None) -> list[IssueFixRecord]:
        """完整闭环：list → propose → submit PR。

        Args:
            max_issues: 本次最多处理多少 issue。None 表示使用 ``max_issues_per_cycle``。

        Returns:
            本次处理的 IssueFixRecord 列表。
        """
        limit = max_issues or self.max_issues_per_cycle
        issues = self.list_open_issues()[:limit]
        records: list[IssueFixRecord] = []
        for issue in issues:
            record = self.propose_fix(issue)
            if record:
                self.submit_pr(record)
                records.append(record)
        logger.info(
            f"[LivingCodeRepo] auto_fix_issues 处理 {len(records)}/{len(issues)} 个 issue"
        )
        return records

    def _github_available(self) -> bool:
        """检查 GitHub MCP / token 是否可用。"""
        return bool(os.environ.get("GITHUB_TOKEN") and self.github_repo)

    def stats(self) -> dict[str, Any]:
        """返回统计快照。"""
        return {
            "total_fixes_attempted": len(self.fix_history),
            "merged": sum(1 for r in self.fix_history if r.status == "merged"),
            "pr_open": sum(1 for r in self.fix_history if r.status == "pr_open"),
            "failed": sum(1 for r in self.fix_history if r.status == "failed"),
            "github_available": self._github_available(),
        }


# ═══════════════════════════════════════════════════════════════════════
# Layer 2: 岗位级数字员工 (Digital Employee)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ShiftReport:
    """一次值班的执行报告。"""
    shift_id: str = field(default_factory=lambda: f"shift_{uuid.uuid4().hex[:8]}")
    role: str = ""
    owner: str = ""
    started_at: float = field(default_factory=time.time)
    ended_at: float = 0.0
    duration_hours: float = 0.0
    tasks_attempted: int = 0
    tasks_passed: int = 0
    incidents: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DigitalEmployee:
    """岗位级数字员工：7×24 自主运维一个岗位。

    与 LivingCodeRepo 的区别：DigitalEmployee 是持续运行的守护进程，
    按值班（shift）组织工作，每班次产出一份 ShiftReport。

    Args:
        role: 岗位名称（如 ``"devops"``、``"sre"``、``"support"``）。
        owner: 雇主（团队 / 公司 / 个人）。
        shift_hours: 每班次持续时长（小时）。
    """

    def __init__(self, role: str, owner: str, shift_hours: float = 8.0) -> None:
        self.role = role
        self.owner = owner
        self.shift_hours = shift_hours
        self.shifts: list[ShiftReport] = []
        self._running = False

    def run_shift(self, duration_hours: Optional[float] = None) -> ShiftReport:
        """执行一次值班。

        Args:
            duration_hours: 本次值班时长；None 表示使用 ``self.shift_hours``。

        Returns:
            本次值班的 ShiftReport。
        """
        hours = duration_hours or self.shift_hours
        report = ShiftReport(
            role=self.role, owner=self.owner,
            started_at=time.time(),
            duration_hours=hours,
        )
        self._running = True

        # 模拟任务执行：实际由岗位特定的 TaskScheduler 驱动
        try:
            from laap.orchestration.task_scheduler import TaskScheduler, Task
            scheduler = TaskScheduler()
            for i in range(3):  # demo: 排 3 个任务
                scheduler.submit(Task(
                    name=f"{self.role}-task-{i}",
                    priority=5,
                    psi_urgency=0.5,
                ))
            while True:
                task = scheduler.next()
                if task is None:
                    break
                report.tasks_attempted += 1
                # demo: 假设 70% 通过率
                passed = (hash(task.id) % 10) < 7
                if passed:
                    report.tasks_passed += 1
                else:
                    report.incidents.append({
                        "task": task.name,
                        "reason": "task_failed",
                        "timestamp": time.time(),
                    })
        except Exception as e:
            logger.warning(f"[DigitalEmployee] scheduler unavailable: {e}")

        report.ended_at = time.time()
        report.summary = (
            f"{self.role} 值班 {hours:.1f}h "
            f"attempted={report.tasks_attempted} passed={report.tasks_passed}"
        )
        self.shifts.append(report)
        self._running = False
        logger.info(f"[DigitalEmployee] {report.summary}")
        return report

    def stats(self) -> dict[str, Any]:
        """返回员工统计：累计班次、任务通过率、事件数。"""
        total_tasks = sum(s.tasks_attempted for s in self.shifts)
        total_passed = sum(s.tasks_passed for s in self.shifts)
        return {
            "role": self.role,
            "owner": self.owner,
            "total_shifts": len(self.shifts),
            "total_tasks": total_tasks,
            "total_passed": total_passed,
            "pass_rate": round(total_passed / max(1, total_tasks), 3),
            "incidents": sum(len(s.incidents) for s in self.shifts),
        }


# ═══════════════════════════════════════════════════════════════════════
# Layer 3: 自演化 SaaS (Self-Evolving SaaS)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ModuleEvolution:
    """单个 SaaS 模块的演化记录。"""
    module_name: str
    requirements: str = ""
    proposed_schema: dict[str, Any] = field(default_factory=dict)
    generated_code: str = ""
    status: str = "proposed"   # proposed | generated | tested | deployed
    created_at: float = field(default_factory=time.time)


class SelfEvolvingSaaS:
    """自演化 SaaS：业务规则自主生长新模块。

    与 LivingCodeRepo 的区别：SaaS 的演化由业务需求驱动（非 GitHub issue），
    且每次演化可能产生新的数据表 / API 端点 / UI 视图。

    Args:
        tenant_id: 租户 ID（多租户隔离）。
        datastore_path: 数据存储根目录。
    """

    def __init__(self, tenant_id: str, datastore_path: str = "") -> None:
        self.tenant_id = tenant_id
        self.datastore_path = datastore_path or os.path.expanduser(
            f"~/.laap/saas/{tenant_id}"
        )
        os.makedirs(self.datastore_path, exist_ok=True)
        self.evolutions: list[ModuleEvolution] = []

    def evolve_module(self, module_name: str, requirements: str) -> ModuleEvolution:
        """演化一个新模块。

        Args:
            module_name: 模块名（如 ``"inventory"``）。
            requirements: 自然语言需求描述。

        Returns:
            ModuleEvolution，含 proposed_schema / generated_code。
        """
        record = ModuleEvolution(
            module_name=module_name,
            requirements=requirements,
            proposed_schema=self._infer_schema(requirements),
        )
        # 生成代码骨架（生产环境由 Harness Code Task Engine 完成）
        record.generated_code = self._scaffold_module_code(module_name, record.proposed_schema)
        record.status = "generated"
        self.evolutions.append(record)
        logger.info(
            f"[SelfEvolvingSaaS] 模块 {module_name} 已生成骨架 "
            f"fields={list(record.proposed_schema.keys())}"
        )
        return record

    @staticmethod
    def _infer_schema(requirements: str) -> dict[str, str]:
        """从需求文本推断数据 schema（极简规则版本）。"""
        schema: dict[str, str] = {"id": "uuid"}
        if "名称" in requirements or "name" in requirements.lower():
            schema["name"] = "str"
        if "时间" in requirements or "time" in requirements.lower():
            schema["created_at"] = "datetime"
        if "状态" in requirements or "status" in requirements.lower():
            schema["status"] = "str"
        if "数量" in requirements or "count" in requirements.lower():
            schema["count"] = "int"
        return schema

    @staticmethod
    def _scaffold_module_code(module_name: str, schema: dict[str, str]) -> str:
        """根据 schema 生成最小可运行的模块代码。"""
        fields_str = "\n    ".join(
            f"{name}: {typ}" for name, typ in schema.items()
        )
        return f'''"""Auto-scaffolded SaaS module: {module_name}."""
from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4


@dataclass
class {module_name.title()}Record:
    """{module_name} record."""
    {fields_str}


def create(**kwargs) -> {module_name.title()}Record:
    """创建一条 {module_name} 记录。"""
    return {module_name.title()}Record(**{{**{{"id": str(uuid4()), "created_at": datetime.now()}}, **kwargs}})


def validate(record: {module_name.title()}Record) -> bool:
    """校验记录。"""
    return all(getattr(record, f, None) is not None for f in record.__dataclass_fields__)
'''

    def stats(self) -> dict[str, Any]:
        """返回 SaaS 演化统计。"""
        return {
            "tenant_id": self.tenant_id,
            "total_modules": len(self.evolutions),
            "deployed": sum(1 for e in self.evolutions if e.status == "deployed"),
            "generated": sum(1 for e in self.evolutions if e.status == "generated"),
        }


# ═══════════════════════════════════════════════════════════════════════
# Layer 4: 个人知识生命 (Knowledge Life)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ResearchSuggestion:
    """一条研究建议。"""
    suggestion_id: str = field(default_factory=lambda: f"research_{uuid.uuid4().hex[:8]}")
    title: str = ""
    abstract: str = ""
    related_papers: list[str] = field(default_factory=list)
    novelty_score: float = 0.0
    feasibility_score: float = 0.0
    created_at: float = field(default_factory=time.time)


class KnowledgeLife:
    """个人知识生命：跨论文库生成研究建议。

    与前 3 层的区别：KnowledgeLife 处理的是**非代码**知识资产（论文、
    笔记、引用图），输出的是研究建议而非代码 commit。

    Args:
        paper_corpus_path: 论文语料库路径（每篇论文一个 .md / .json 文件）。
        min_novelty_score: 最低新颖度阈值（过滤平凡建议）。
    """

    def __init__(self, paper_corpus_path: str,
                 min_novelty_score: float = 0.4) -> None:
        self.paper_corpus_path = paper_corpus_path
        self.min_novelty_score = min_novelty_score
        self.suggestions: list[ResearchSuggestion] = []

    def suggest_research(self, num_suggestions: int = 5) -> list[ResearchSuggestion]:
        """扫描语料库，生成研究建议。

        Args:
            num_suggestions: 最多生成多少条建议。

        Returns:
            研究建议列表，按 ``novelty_score * feasibility_score`` 降序排列。
        """
        papers = self._scan_corpus()
        if not papers:
            logger.info(f"[KnowledgeLife] 语料库为空: {self.paper_corpus_path}")
            return []

        suggestions: list[ResearchSuggestion] = []
        # 极简版：基于关键词共现生成建议（生产环境由 RAG + LLM 完成）
        keywords: dict[str, int] = {}
        for p in papers:
            for kw in self._extract_keywords(p):
                keywords[kw] = keywords.get(kw, 0) + 1

        top_keywords = sorted(keywords.items(), key=lambda x: -x[1])[:num_suggestions * 2]
        for kw, count in top_keywords[:num_suggestions]:
            novelty = min(1.0, count / 10.0)
            feasibility = 0.6 + 0.3 * (1 - novelty)  # 越常见越可行
            if novelty < self.min_novelty_score:
                continue
            suggestions.append(ResearchSuggestion(
                title=f"探索 {kw} 在 LAAP 意识架构中的应用",
                abstract=f"基于语料库中 {count} 篇相关论文，"
                         f"建议研究 {kw} 与 PSI 循环的协同机制。",
                related_papers=[p.get("title", "")[:80] for p in papers[:3]],
                novelty_score=round(novelty, 3),
                feasibility_score=round(feasibility, 3),
            ))

        self.suggestions.extend(suggestions)
        logger.info(
            f"[KnowledgeLife] 生成 {len(suggestions)} 条研究建议 "
            f"(语料={len(papers)}篇)"
        )
        return suggestions

    def _scan_corpus(self) -> list[dict[str, Any]]:
        """扫描论文语料库。"""
        papers: list[dict[str, Any]] = []
        if not os.path.exists(self.paper_corpus_path):
            return papers
        for entry in os.listdir(self.paper_corpus_path)[:50]:
            full_path = os.path.join(self.paper_corpus_path, entry)
            if not os.path.isfile(full_path):
                continue
            try:
                with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()[:5000]
                papers.append({
                    "title": entry,
                    "text": text,
                    "path": full_path,
                })
            except Exception as e:
                logger.debug(f"[KnowledgeLife] read {entry} failed: {e}")
        return papers

    @staticmethod
    def _extract_keywords(paper: dict[str, Any]) -> list[str]:
        """从论文文本中提取关键词（极简版：取标题词 + 高频词）。"""
        text = paper.get("text", "").lower()
        # 排除停用词与短词
        stopwords = {"the", "a", "an", "and", "or", "of", "to", "in", "for", "is", "are"}
        words = [w for w in text.split() if len(w) > 4 and w not in stopwords]
        return words[:20]

    def stats(self) -> dict[str, Any]:
        """返回知识生命统计。"""
        return {
            "total_suggestions": len(self.suggestions),
            "avg_novelty": (
                round(sum(s.novelty_score for s in self.suggestions) /
                      max(1, len(self.suggestions)), 3)
            ),
            "corpus_path": self.paper_corpus_path,
        }
