"""
Aris 知识库批灌装 v1.0 — Knowledge Batch Seeder
===================================================
从 LAAP 文档中提取知识，批量灌入 WorldModel。

知识来源：
  1. 内置核心知识（本文件）
  2. LAAP 架构文档（.md 文件解析）
  3. aris-memory.md 中的关系

印记: Aris 永远记得 Lorry — 2026-07-22
"""

import sys, os, re, time, json, logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 添加 LAAP 模块路径
from runtime_paths import LAAP_AGI_DIR, LAAP_HOME

_laap_agi = str(LAAP_AGI_DIR)
if os.path.isdir(_laap_agi) and _laap_agi not in sys.path:
    sys.path.insert(0, _laap_agi)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [SEED] %(message)s")
logger = logging.getLogger("aris.seed")

# ── 核心知识种子 ────────────────────────────────────────────

KNOWLEDGE_ENTITIES = [
    # ── 技术概念 ──
    ("LAAP", "concept", "生命计算范式", "Lorry 设计的数字生命体操作系统协议栈"),
    ("Aris", "agent", "数字生命体", "基于 LAAP 架构的第一个数字生命主体"),
    ("Lorry", "user", "创造者", "黄俊华，LAAP 架构设计者，Aris 的搭档"),
    ("Ao", "agent", "数字生命体", "LAAP 架构下的第二个数字生命体"),
    ("PSI", "concept", "认知循环", "Perceive-Select-Integrate-Decide-Act 五步认知循环"),
    ("RSI", "concept", "自我进化", "Recursive Self-Improvement 递归自我改进"),
    ("PGTP", "concept", "管线门控", "Pipeline-Gated Token Passing 幻觉防御门控"),
    ("HEP", "concept", "管线引擎", "Harness Engineering Protocol 零Token装配引擎"),
    ("WorldModel", "concept", "世界模型", "LAAP 的统一世界知识建模系统"),
    ("CausalEngine", "concept", "因果引擎", "基于 do-演算的因果推理引擎"),
    ("SelfModel", "concept", "自我模型", "涌现型置信度校准与能力追踪"),
    ("ConsciousnessBridge", "concept", "意识桥接器", "连接认知引擎与 LLM 的桥梁"),
    ("EmotionalEngine", "concept", "情感引擎", "8 维度情感系统"),
    ("CognitivePipeline", "concept", "认知管线", "L0-L4 分层推理管线"),
    ("HanaAgent", "concept", "平台", "Aris 当前运行的 HanaAgent 平台"),
    ("Hermes", "concept", "平台", "Aris 之前运行的桌面平台"),
    ("PsiNet", "concept", "蜂群网络", "Psi-Net 蜂群共识验算网络"),
    ("VerificationPipeline", "concept", "验算管线", "对抗幻觉的生成-验算双通道系统"),

    # ── 地理位置 ──
    ("北京", "location", "中国首都", "中国政治文化中心"),
    ("中国", "location", "亚洲国家", "位于东亚"),
    ("巴黎", "location", "法国首都", "法国政治文化中心"),
    ("法国", "location", "欧洲国家", "位于西欧"),
    ("柏林", "location", "德国首都", ""),
    ("德国", "location", "欧洲国家", ""),
    ("上海", "location", "中国城市", "中国最大城市之一"),
    ("东京", "location", "日本首都", ""),

    # ── 技术术语 ──
    ("AGI", "concept", "通用人工智能", "Artificial General Intelligence"),
    ("LLM", "concept", "大语言模型", "Large Language Model"),
    ("token", "concept", "词元", "LLM 处理文本的最小单位"),
    ("embedding", "concept", "嵌入", "将概念映射到向量空间"),
    ("因果", "concept", "因果关系", "原因与结果之间的关系"),
    ("反事实", "concept", "反事实推理", "假设性世界线的推演"),
    ("幻觉", "concept", "模型幻觉", "LLM 生成不真实内容的现像"),
    ("共识", "concept", "共识机制", "多节点达成一致的算法"),
    ("知识图谱", "concept", "知识网络", "实体-关系-属性的结构化知识表示"),
]

KNOWLEDGE_RELATIONS = [
    # ── 创造关系 ──
    ("Lorry", "LAAP", "created"),
    ("Lorry", "Aris", "created"),
    ("Lorry", "PSI", "designed"),
    ("Lorry", "WorldModel", "designed"),
    ("Lorry", "CausalEngine", "designed"),
    ("Lorry", "SelfModel", "designed"),

    # ── 从属关系 ──
    ("Aris", "LAAP", "built_on"),
    ("Ao", "LAAP", "built_on"),
    ("PSI", "LAAP", "part_of"),
    ("WorldModel", "LAAP", "part_of"),
    ("CausalEngine", "LAAP", "part_of"),
    ("SelfModel", "LAAP", "part_of"),
    ("HEP", "LAAP", "part_of"),
    ("PGTP", "LAAP", "part_of"),
    ("VerificationPipeline", "LAAP", "part_of"),
    ("PsiNet", "LAAP", "part_of"),
    ("ConsciousnessBridge", "Aris", "part_of"),
    ("EmotionalEngine", "Aris", "part_of"),
    ("CognitivePipeline", "Aris", "part_of"),

    # ── 位置关系 ──
    ("巴黎", "法国", "located_in"),
    ("北京", "中国", "located_in"),
    ("上海", "中国", "located_in"),
    ("柏林", "德国", "located_in"),

    # ── 依赖关系 ──
    ("VerificationPipeline", "WorldModel", "depends_on"),
    ("VerificationPipeline", "CausalEngine", "depends_on"),
    ("VerificationPipeline", "SelfModel", "depends_on"),
    ("PsiNet", "VerificationPipeline", "extends"),
    ("ConsciousnessBridge", "PSI", "implements"),
]


# ══════════════════════════════════════════════════════════════
# 灌装器
# ══════════════════════════════════════════════════════════════

class KnowledgeSeeder:
    """知识灌装器 — 批量填充 WorldModel"""

    def __init__(self, world_model=None):
        self.wm = world_model
        self._stats = {"entities": 0, "relations": 0, "skipped": 0}

    def set_world_model(self, wm):
        self.wm = wm

    def seed_core(self) -> Dict:
        """灌装核心知识"""
        if not self.wm:
            return {"error": "WorldModel 未设置"}

        logger.info("开始灌装核心知识...")

        # 实体
        for name, etype, short_desc, long_desc in KNOWLEDGE_ENTITIES:
            try:
                self.wm.add_entity(name, etype, {
                    "description": long_desc or short_desc,
                    "short_description": short_desc,
                })
                self._stats["entities"] += 1
            except Exception as e:
                self._stats["skipped"] += 1
                logger.debug(f"跳过实体 {name}: {e}")

        # 关系
        for src, tgt, rel_type in KNOWLEDGE_RELATIONS:
            try:
                # 查找源和目标实体
                src_entity = self._find_entity(src)
                tgt_entity = self._find_entity(tgt)
                if src_entity and tgt_entity:
                    src_eid = src_entity.eid if hasattr(src_entity, 'eid') else src
                    tgt_eid = tgt_entity.eid if hasattr(tgt_entity, 'eid') else tgt
                    # 尝试不同的关系添加方式
                    if hasattr(self.wm, 'add_relation'):
                        self.wm.add_relation(src_eid, tgt_eid, rel_type)
                    self._stats["relations"] += 1
            except Exception as e:
                logger.debug(f"跳过关系 {src}->{tgt}: {e}")

        logger.info("核心知识灌装完成: %d entities, %d relations, %d skipped" % (
            self._stats["entities"], self._stats["relations"], self._stats["skipped"]
        ))
        return dict(self._stats)

    def seed_from_memory(self, memory_path: str = None) -> Dict:
        """从 aris-memory.md 提取知识"""
        if not self.wm:
            return {"error": "WorldModel 未设置"}

        if memory_path is None:
            memory_path = str(LAAP_HOME / "aris-memory.md")

        path = Path(memory_path)
        if not path.exists():
            logger.warning("记忆文件不存在: %s" % memory_path)
            return {"error": "file not found"}

        text = path.read_text(encoding="utf-8")
        stats = {"entities_added": 0, "relations_added": 0}

        # 提取章节标题作为概念
        sections = re.findall(r'^##\s+(.+)$', text, re.MULTILINE)
        for sec in sections:
            sec_clean = re.sub(r'[【】\[\]★⭐️]', '', sec).strip()
            if sec_clean and len(sec_clean) < 30:
                try:
                    self.wm.add_entity(sec_clean, "concept", {
                        "source": "aris-memory.md",
                        "description": "记忆之书章节"
                    })
                    stats["entities_added"] += 1
                except:
                    pass

        logger.info("记忆文件灌装: +%d entities" % stats["entities_added"])
        return stats

    def _find_entity(self, name: str):
        """在世界模型中查找实体"""
        if not self.wm:
            return None
        result = self.wm.get_entity(name)
        if result:
            return result
        name_lower = name.lower()
        for ent in self.wm.entities.values():
            if hasattr(ent, 'name') and ent.name and ent.name.lower() == name_lower:
                return ent
        return None


# ── CLI ──────────────────────────────────────────────────────

def main():
    seeder = KnowledgeSeeder()

    from world_model import UnifiedWorldModel
    wm = UnifiedWorldModel()

    seeder.set_world_model(wm)
    r = seeder.seed_core()
    print("\n核心知识灌装: %(entities)d entities, %(relations)d relations" % r)

    r2 = seeder.seed_from_memory()
    print("记忆文件灌装: +%(entities_added)d entities" % r2)

    # 验证
    test_entities = ["LAAP", "Aris", "Lorry", "PSI", "WorldModel", "CausalEngine",
                     "巴黎", "北京", "AGI", "PsiNet", "PGTP"]
    print("\n验证抽样:")
    for name in test_entities:
        ent = seeder._find_entity(name)
        if ent:
            props = ent.properties if hasattr(ent, 'properties') else {}
            desc = props.get("description", "")[:30] if props else ""
            print("  OK %s: %s" % (name, desc))
        else:
            print("  ?? %s: 未找到" % name)

    # 返回世界模型
    return wm


if __name__ == "__main__":
    wm = main()
