"""
LAAP SaaS — FastAPI 服务器 (v2.0)

提供:
- 动态页面渲染 (LAAP-UI → HTML)
- REST API 端点 (/api/v1/{entity} 通用CRUD)
- Colony Agent 桥接
- 多租户 Auth 中间件
- 数据模型层 (SchemaRegistry + GenericCRUD)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from laap.protocol.laap_ui import (
    Component, ComponentType, LayoutTree, LayoutType,
    FlexStyle, GridStyle, TextStyle, EventBinding, EventType,
    StyleDefinition, ThemeConfig, ThemePalette, TypographyConfig,
    SpacingConfig, BorderRadiusConfig, ShadowConfig,
)
from laap.saas.renderer.html_renderer import HTMLRenderer, render_page

# A1: Data Model Layer
from laap.saas.datastore.schema_registry import SchemaRegistry
from laap.saas.datastore.generic_crud import GenericCRUD, FilterTuple
from laap.saas.datastore.auto_migrator import AutoMigrator
from laap.config.paths import get_laap_root

# A2: Multi-tenant
from laap.saas.tenant.manager import TenantManager

from starlette.requests import Request as StarletteRequest
logger = logging.getLogger("laap.saas.server")

# ── 默认主题 ──────────────────────────────────────────────────
DEFAULT_THEME = ThemeConfig(
    name="laap-default",
    palette=ThemePalette(
        primary="#7C3AED", secondary="#3B82F6", accent="#F59E0B",
        success="#10B981", warning="#F59E0B", danger="#EF4444",
        info="#6366F1", background="#0F172A", surface="#1E293B",
        text_primary="#F8FAFC", text_secondary="#94A3B8",
        text_disabled="#475569", border="#334155", divider="#1E293B",
    ),
)

# ── CSS 工具 ──────────────────────────────────────────────────

def css(**props) -> Dict[str, str]:
    return props


def comp(tag: str, ctype: ComponentType = ComponentType.CONTAINER,
         text: str = "",
         style: Optional[Dict[str, str]] = None,
         **kwargs) -> dict:
    result = {"id": f"c-{uuid.uuid4().hex[:6]}", "component_type": ctype, "children": []}
    if tag:
        result["id"] = tag
    if text:
        result["text_content"] = text
    if style:
        styledef_fields = set(StyleDefinition.__dataclass_fields__)
        sd_kwargs = {}
        extra_css = {}
        for k, v in style.items():
            py_key = k.replace("-", "_")
            if py_key in styledef_fields:
                sd_kwargs[py_key] = v
            else:
                extra_css[k] = v
        if sd_kwargs:
            result["style"] = StyleDefinition(**sd_kwargs)
        if extra_css:
            result["metadata"] = {"css": extra_css}
    for k, v in kwargs.items():
        if k == "children":
            result["children"] = v
        elif k == "flex":
            # 转换 kebab-case → snake_case
            flex_kwargs = {}
            for fk, fv in v.items():
                flex_kwargs[fk.replace("-", "_")] = fv
            result["flex_style"] = FlexStyle(**flex_kwargs) if isinstance(v, dict) else v
        elif k == "grid":
            grid_kwargs = {}
            for gk, gv in v.items():
                grid_kwargs[gk.replace("-", "_")] = gv
            result["grid_style"] = GridStyle(**grid_kwargs) if isinstance(v, dict) else v
        elif k == "props":
            result["properties"] = v
        elif k == "classes":
            result["classes"] = v
        elif k == "metadata":
            if result.get("metadata"):
                result["metadata"].update(v)
            else:
                result["metadata"] = v
    return result


def make(tree: dict) -> Component:
    children = [make(c) for c in tree.pop("children", [])]
    return Component(children=children, **tree)

# ── 全局单例 ──────────────────────────────────────────────────

_registry: Optional[SchemaRegistry] = None
_crud: Optional[GenericCRUD] = None
_migrator: Optional[AutoMigrator] = None
_tenant_mgr: Optional[TenantManager] = None
_agents: Dict[str, Any] = {}

_page_registry: Dict[str, Dict[str, Any]] = {}


def _ensure_data_layer(db_path: str = "laap_saas.db"):
    """延迟初始化"""
    global _registry, _crud, _migrator, _tenant_mgr
    if _registry is None:
        _registry = SchemaRegistry(db_path)
        _crud = GenericCRUD(_registry)
        _migrator = AutoMigrator(_registry)
        _tenant_mgr = TenantManager(db_path=db_path)
        logger.info(f"Data layer initialized: {db_path}")


def _ensure_colony():
    """延迟初始化 Colony Agent (LAAP 2.0)

    如果 Colony / AGI 依赖不可用，静默降级（不影响 SaaS 启动）。
    """
    global _agents
    if _agents:
        return
    try:
        from laap.colony import get_available_agents
        from laap.sandbox import ColonyEventBus, SkillLibrary

        # 使用 get_available_agents() 获取所有可用的 Agent 类
        available = get_available_agents()
        bus = ColonyEventBus()
        lib = SkillLibrary()
        agent_id = 0
        for name, agent_cls in available.items():
            agent_id += 1
            sid = f"sb-saas-{name.lower().replace('agent','')}-{agent_id:03d}"
            try:
                agent = agent_cls(sid, lib, bus)
                _agents[name.lower()] = agent
            except Exception as inner_e:
                logger.warning(f"Agent {name} 实例化失败: {inner_e}")

        logger.info(f"Colony initialized: {len(_agents)} agents")
    except ImportError as e:
        logger.debug(f"Colony 不可用（跳过）: {e}")
    except Exception as e:
        logger.warning(f"Colony 初始化异常: {e}")


# ── 页面注册 ──────────────────────────────────────────────────

def register_page(page_id: str, title: str,
                  component: Component,
                  theme: Optional[ThemeConfig] = None) -> str:
    _page_registry[page_id] = {
        "id": page_id, "title": title,
        "component": component, "theme": theme or DEFAULT_THEME,
        "created_at": time.time(),
    }
    return page_id


# ── 内置页面 ──────────────────────────────────────────────────

def _build_landing_page() -> Component:
    return make(comp("root", ComponentType.ROOT,
        style={"display": "flex", "flex-direction": "column",
               "min-height": "100vh",
               "background-color": "var(--laap-palette-background)"},
        children=[
        comp("header", ComponentType.HEADER,
             style={"padding": "16px 32px", "border-bottom": "1px solid var(--laap-palette-border)",
                    "display": "flex", "align-items": "center", "justify-content": "space-between"},
             children=[
            comp("logo", ComponentType.TEXT, text="LAAP SaaS",
                 style={"font-size": "20px", "font-weight": "700",
                        "color": "var(--laap-palette-text-primary)"}),
            comp("nav", ComponentType.CONTAINER, flex={"gap": "24px"},
                 children=[
                comp("dash", ComponentType.LINK, text="控制台",
                     props={"href": "/page/dashboard"},
                     style={"color": "var(--laap-palette-text-secondary)",
                            "text-decoration": "none", "cursor": "pointer"}),
                comp("models", ComponentType.LINK, text="模型管理",
                     props={"href": "/page/models"},
                     style={"color": "var(--laap-palette-text-secondary)",
                            "text-decoration": "none", "cursor": "pointer"}),
                comp("tenants", ComponentType.LINK, text="租户管理",
                     props={"href": "/page/tenants"},
                     style={"color": "var(--laap-palette-text-secondary)",
                            "text-decoration": "none", "cursor": "pointer"}),
            ]),
        ]),
        comp("hero", ComponentType.CONTAINER,
             style={"display": "flex", "flex-direction": "column",
                    "align-items": "center", "justify-content": "center",
                    "padding": "80px 32px", "text-align": "center",
                    "flex": "1"},
             children=[
            comp("title", ComponentType.TEXT, text="LAAP 自演化 SaaS",
                 style={"font-size": "48px", "font-weight": "800",
                        "color": "var(--laap-palette-text-primary)",
                        "margin-bottom": "16px"}),
            comp("sub", ComponentType.TEXT, text="你的业务系统会自己长大",
                 style={"font-size": "20px",
                        "color": "var(--laap-palette-text-secondary)",
                        "margin-bottom": "32px"}),
            comp("badge", ComponentType.BADGE,
                 text="LAAP 2.0 Living Runtime",
                 style={"padding": "8px 16px",
                        "background": "var(--laap-palette-primary)",
                        "color": "white", "border-radius": "999px",
                        "font-size": "14px"}),
        ]),
        comp("features", ComponentType.CONTAINER,
             flex={"gap": "24px", "justify-content": "center"},
             style={"padding": "0 32px 64px",
                    "display": "flex", "flex-wrap": "wrap"},
             children=[
            comp("f1", ComponentType.CARD,
                 style={"padding": "24px", "width": "280px",
                        "background": "var(--laap-palette-surface)",
                        "border-radius": "12px",
                        "border": "1px solid var(--laap-palette-border)"},
                 children=[
                    comp("f1t", ComponentType.TEXT, text="动态数据模型",
                         style={"font-size": "18px", "font-weight": "700",
                                "color": "var(--laap-palette-text-primary)",
                                "margin-bottom": "8px"}),
                    comp("f1d", ComponentType.TEXT,
                         text="JSON Schema → SQLite 自动建表，零配置启动",
                         style={"font-size": "14px",
                                "color": "var(--laap-palette-text-secondary)"}),
                 ]),
            comp("f2", ComponentType.CARD,
                 style={"padding": "24px", "width": "280px",
                        "background": "var(--laap-palette-surface)",
                        "border-radius": "12px",
                        "border": "1px solid var(--laap-palette-border)"},
                 children=[
                    comp("f2t", ComponentType.TEXT, text="通用 CRUD API",
                         style={"font-size": "18px", "font-weight": "700",
                                "color": "var(--laap-palette-text-primary)",
                                "margin-bottom": "8px"}),
                    comp("f2d", ComponentType.TEXT,
                         text="自动暴露 RESTful API，支持过滤/排序/分页/多租户",
                         style={"font-size": "14px",
                                "color": "var(--laap-palette-text-secondary)"}),
                 ]),
            comp("f3", ComponentType.CARD,
                 style={"padding": "24px", "width": "280px",
                        "background": "var(--laap-palette-surface)",
                        "border-radius": "12px",
                        "border": "1px solid var(--laap-palette-border)"},
                 children=[
                    comp("f3t", ComponentType.TEXT, text="Colony 数字生命体",
                         style={"font-size": "18px", "font-weight": "700",
                                "color": "var(--laap-palette-text-primary)",
                                "margin-bottom": "8px"}),
                    comp("f3d", ComponentType.TEXT,
                         text="集成沙箱 Agent 提供架构/安全/性能/测试分析",
                         style={"font-size": "14px",
                                "color": "var(--laap-palette-text-secondary)"}),
                 ]),
        ]),
        comp("footer", ComponentType.FOOTER,
             text="LAAP 2.0 Living Runtime — Powered by LAAP",
             style={"padding": "24px 32px", "border-top": "1px solid var(--laap-palette-border)",
                    "text-align": "center", "color": "var(--laap-palette-text-disabled)",
                    "font-size": "12px", "margin-top": "auto"}),
    ]))


def _build_dashboard_page() -> Component:
    """控制台页面 — 显示 SaaS 运行状态"""
    models = _registry.list_models() if _registry else []
    tenants = _tenant_mgr.list() if _tenant_mgr else []
    return make(comp("root", ComponentType.ROOT,
        style={"display": "flex", "flex-direction": "column",
               "min-height": "100vh",
               "background-color": "var(--laap-palette-background)"},
        children=[
        comp("header", ComponentType.HEADER,
             style={"padding": "16px 32px", "border-bottom": "1px solid var(--laap-palette-border)",
                    "display": "flex", "align-items": "center", "justify-content": "space-between"},
             children=[
            comp("logo", ComponentType.LINK, text="← LAAP SaaS",
                 props={"href": "/page/landing"},
                 style={"font-size": "20px", "font-weight": "700",
                        "color": "var(--laap-palette-primary)",
                        "text-decoration": "none"}),
        ]),
        comp("content", ComponentType.CONTAINER,
             style={"padding": "32px", "flex": "1"},
             children=[
            comp("title", ComponentType.TEXT, text="系统控制台",
                 style={"font-size": "32px", "font-weight": "800",
                        "color": "var(--laap-palette-text-primary)",
                        "margin-bottom": "24px"}),
            # 统计卡片行
            comp("stats", ComponentType.CONTAINER,
                 flex={"gap": "16px"},
                 style={"display": "flex", "margin-bottom": "32px"},
                 children=[
                comp("sc1", ComponentType.CARD,
                     style={"padding": "20px", "flex": "1",
                            "background": "var(--laap-palette-surface)",
                            "border-radius": "12px",
                            "border": "1px solid var(--laap-palette-border)"},
                     children=[
                        comp("sc1v", ComponentType.TEXT,
                             text=str(len(models)),
                             style={"font-size": "36px", "font-weight": "800",
                                    "color": "var(--laap-palette-primary)"}),
                        comp("sc1l", ComponentType.TEXT, text="数据模型",
                             style={"font-size": "14px",
                                    "color": "var(--laap-palette-text-secondary)"}),
                     ]),
                comp("sc2", ComponentType.CARD,
                     style={"padding": "20px", "flex": "1",
                            "background": "var(--laap-palette-surface)",
                            "border-radius": "12px",
                            "border": "1px solid var(--laap-palette-border)"},
                     children=[
                        comp("sc2v", ComponentType.TEXT,
                             text=str(len(tenants)),
                             style={"font-size": "36px", "font-weight": "800",
                                    "color": "var(--laap-palette-accent)"}),
                        comp("sc2l", ComponentType.TEXT, text="租户",
                             style={"font-size": "14px",
                                    "color": "var(--laap-palette-text-secondary)"}),
                     ]),
                comp("sc3", ComponentType.CARD,
                     style={"padding": "20px", "flex": "1",
                            "background": "var(--laap-palette-surface)",
                            "border-radius": "12px",
                            "border": "1px solid var(--laap-palette-border)"},
                     children=[
                        comp("sc3v", ComponentType.TEXT,
                             text=str(len(_agents)),
                             style={"font-size": "36px", "font-weight": "800",
                                    "color": "var(--laap-palette-success)"}),
                        comp("sc3l", ComponentType.TEXT, text="Colony Agent",
                             style={"font-size": "14px",
                                    "color": "var(--laap-palette-text-secondary)"}),
                     ]),
            ]),
            # API 表格
            comp("api_section", ComponentType.CONTAINER,
                 style={"margin-bottom": "24px"},
                 children=[
                comp("api_t", ComponentType.TEXT, text="API 端点",
                     style={"font-size": "20px", "font-weight": "700",
                            "color": "var(--laap-palette-text-primary)",
                            "margin-bottom": "12px"}),
                comp("api_table", ComponentType.TABLE,
                     props={"columns": [
                         {"key": "method", "title": "方法"},
                         {"key": "path", "title": "路径"},
                         {"key": "desc", "title": "描述"},
                     ], "rows": [
                         {"method": "GET", "path": "/health", "desc": "健康检查"},
                         {"method": "POST", "path": "/api/schema/register", "desc": "注册数据模型"},
                         {"method": "GET", "path": "/api/v1/{entity}", "desc": "通用查询"},
                         {"method": "POST", "path": "/api/v1/{entity}", "desc": "通用创建"},
                         {"method": "PATCH", "path": "/api/v1/{entity}/{id}", "desc": "通用更新"},
                         {"method": "DELETE", "path": "/api/v1/{entity}/{id}", "desc": "通用删除"},
                         {"method": "GET", "path": "/api/agents", "desc": "列出 Colony Agent"},
                         {"method": "GET", "path": "/api/tenants", "desc": "列出租户"},
                     ]},
                     style={"width": "100%",
                            "border-collapse": "collapse",
                            "color": "var(--laap-palette-text-primary)"}),
            ]),
        ]),
        comp("footer", ComponentType.FOOTER,
             text="LAAP SaaS v2.0",
             style={"padding": "16px 32px",
                    "text-align": "center",
                    "color": "var(--laap-palette-text-disabled)",
                    "font-size": "12px"}),
    ]))


def _build_models_page() -> Component:
    """模型管理页面"""
    models = _registry.list_models() if _registry else []
    rows = []
    for m in models:
        rows.append({
            "name": m.get("name", ""),
            "table": m.get("table", ""),
            "version": str(m.get("version", 1)),
            "fields": str(m.get("fields", 0)),
            "isolated": "是" if m.get("tenant_isolated", True) else "否",
        })
    if not rows:
        rows = [{"name": "（暂无注册模型）", "table": "", "version": "", "fields": "", "isolated": ""}]

    return make(comp("root", ComponentType.ROOT,
        style={"display": "flex", "flex-direction": "column",
               "min-height": "100vh",
               "background-color": "var(--laap-palette-background)"},
        children=[
        comp("header", ComponentType.HEADER,
             style={"padding": "16px 32px",
                    "border-bottom": "1px solid var(--laap-palette-border)",
                    "display": "flex", "align-items": "center",
                    "justify-content": "space-between"},
             children=[
            comp("logo", ComponentType.LINK, text="← 控制台",
                 props={"href": "/page/dashboard"},
                 style={"font-size": "20px", "font-weight": "700",
                        "color": "var(--laap-palette-primary)",
                        "text-decoration": "none"}),
        ]),
        comp("content", ComponentType.CONTAINER,
             style={"padding": "32px", "flex": "1"},
             children=[
            comp("title", ComponentType.TEXT, text="数据模型管理",
                 style={"font-size": "32px", "font-weight": "800",
                        "color": "var(--laap-palette-text-primary)",
                        "margin-bottom": "16px"}),
            comp("desc", ComponentType.TEXT,
                 text="注册 JSON Schema 后自动建表并暴露 REST API",
                 style={"font-size": "14px",
                        "color": "var(--laap-palette-text-secondary)",
                        "margin-bottom": "24px"}),
            # Schema 注册表单提示
            comp("form_section", ComponentType.CONTAINER,
                 style={"margin-bottom": "24px",
                        "padding": "20px",
                        "background": "var(--laap-palette-surface)",
                        "border-radius": "12px",
                        "border": "1px solid var(--laap-palette-border)"},
                 children=[
                    comp("ft", ComponentType.TEXT, text="注册新模型",
                         style={"font-size": "16px", "font-weight": "600",
                                "color": "var(--laap-palette-text-primary)",
                                "margin-bottom": "12px"}),
                    comp("fd", ComponentType.TEXT,
                         text="POST /api/schema/register  请求体: { name: \"product\", schema: { type: \"object\", properties: {...} } }",
                         style={"font-size": "13px",
                                "color": "var(--laap-palette-text-secondary)",
                                "font-family": "monospace",
                                "white-space": "pre-wrap",
                                "background": "var(--laap-palette-background)",
                                "padding": "12px", "border-radius": "8px"}),
                ]),
            # 已注册模型表格
            comp("table_section", ComponentType.CONTAINER,
                 children=[
                comp("st", ComponentType.TEXT, text="已注册模型",
                     style={"font-size": "18px", "font-weight": "600",
                            "color": "var(--laap-palette-text-primary)",
                            "margin-bottom": "12px"}),
                comp("model_table", ComponentType.TABLE,
                     props={"columns": [
                         {"key": "name", "title": "模型名"},
                         {"key": "table", "title": "表名"},
                         {"key": "version", "title": "版本"},
                         {"key": "fields", "title": "字段数"},
                         {"key": "isolated", "title": "多租户"},
                     ], "rows": rows},
                     style={"width": "100%",
                            "border-collapse": "collapse",
                            "color": "var(--laap-palette-text-primary)"}),
            ]),
        ]),
        comp("footer", ComponentType.FOOTER,
             text="LAAP SaaS — 数据模型管理",
             style={"padding": "16px 32px",
                    "text-align": "center",
                    "color": "var(--laap-palette-text-disabled)",
                    "font-size": "12px"}),
    ]))


def _build_tenants_page() -> Component:
    """租户管理页面"""
    tenants = _tenant_mgr.list() if _tenant_mgr else []
    rows = []
    for t in tenants:
        rows.append({
            "id": t.get("id", ""),
            "name": t.get("name", ""),
            "status": t.get("status", ""),
            "max_users": str(t.get("max_users", 0)),
            "features": ", ".join(k for k, v in t.get("features", {}).items() if v),
        })
    if not rows:
        rows = [{"id": "（暂无）", "name": "", "status": "", "max_users": "", "features": ""}]

    return make(comp("root", ComponentType.ROOT,
        style={"display": "flex", "flex-direction": "column",
               "min-height": "100vh",
               "background-color": "var(--laap-palette-background)"},
        children=[
        comp("header", ComponentType.HEADER,
             style={"padding": "16px 32px",
                    "border-bottom": "1px solid var(--laap-palette-border)",
                    "display": "flex", "align-items": "center",
                    "justify-content": "space-between"},
             children=[
            comp("logo", ComponentType.LINK, text="← 控制台",
                 props={"href": "/page/dashboard"},
                 style={"font-size": "20px", "font-weight": "700",
                        "color": "var(--laap-palette-primary)",
                        "text-decoration": "none"}),
        ]),
        comp("content", ComponentType.CONTAINER,
             style={"padding": "32px", "flex": "1"},
             children=[
            comp("title", ComponentType.TEXT, text="租户管理",
                 style={"font-size": "32px", "font-weight": "800",
                        "color": "var(--laap-palette-text-primary)",
                        "margin-bottom": "24px"}),
            comp("tenant_table", ComponentType.TABLE,
                 props={"columns": [
                     {"key": "id", "title": "租户 ID"},
                     {"key": "name", "title": "名称"},
                     {"key": "status", "title": "状态"},
                     {"key": "max_users", "title": "最大用户"},
                     {"key": "features", "title": "功能"},
                 ], "rows": rows},
                 style={"width": "100%",
                        "border-collapse": "collapse",
                        "color": "var(--laap-palette-text-primary)"}),
        ]),
        comp("footer", ComponentType.FOOTER,
             text="LAAP SaaS — 租户管理",
             style={"padding": "16px 32px",
                    "text-align": "center",
                    "color": "var(--laap-palette-text-disabled)",
                    "font-size": "12px"}),
    ]))


# ── 注册内置页面 ──────────────────────────────────────────────
register_page("landing", "LAAP SaaS — 自演化业务系统", _build_landing_page())
register_page("dashboard", "系统控制台", _build_dashboard_page())
register_page("models", "数据模型管理", _build_models_page())
register_page("tenants", "租户管理", _build_tenants_page())


# ── FastAPI 应用 ──────────────────────────────────────────────

def create_app(db_path: str = "laap_saas.db") -> "FastAPI":
    try:
        from fastapi import FastAPI, Request, HTTPException
        from fastapi.responses import HTMLResponse, JSONResponse
        from fastapi.middleware.cors import CORSMiddleware
    except ImportError:
        raise ImportError("需要 fastapi / uvicorn: pip install fastapi uvicorn")

    _ensure_data_layer(db_path)
    _ensure_colony()

    app = FastAPI(title="LAAP SaaS Runtime", version="2.0.0",
                  description="自演化业务系统运行时")

    app.add_middleware(CORSMiddleware, allow_origins=["*"],
                       allow_credentials=True, allow_methods=["*"],
                       allow_headers=["*"])

    # ── 中间件: 解析 tenant_id ──

    @app.middleware("http")
    async def tenant_middleware(request: Request, call_next):
        tenant_id = request.headers.get("x-tenant-id", "default")
        request.state.tenant_id = tenant_id
        response = await call_next(request)
        response.headers["x-tenant-id"] = tenant_id
        return response

    # ── 页面路由 ──

    @app.get("/", response_class=HTMLResponse)
    async def root():
        return await render_page_handler("landing")

    @app.get("/health")
    async def health():
        models = _registry.list_models() if _registry else []
        return {
            "status": "ok", "version": "2.0.0",
            "pages": list(_page_registry.keys()),
            "models": models,
            "agents": list(_agents.keys()),
            "tenants": _tenant_mgr.list() if _tenant_mgr else [],
            "uptime": time.time(),
        }

    @app.get("/page/{page_id}", response_class=HTMLResponse)
    async def render_page_handler(page_id: str):
        if page_id not in _page_registry:
            fallback = make(comp("root", ComponentType.ROOT, children=[
                comp("nf", ComponentType.TEXT,
                     text=f"页面 /page/{page_id} 未找到",
                     style={"padding": "48px", "font-size": "24px",
                            "color": "var(--laap-palette-text-secondary)"}),
            ]))
            register_page(page_id, f"Page {page_id}", fallback)
        entry = _page_registry[page_id]
        return render_page(LayoutTree(root=entry["component"]), theme=entry["theme"])

    # ── 数据模型 API ──

    @app.post("/api/schema/register")
    async def register_schema(data: dict):
        """注册数据模型: POST /api/schema/register
        Body: {"name": "product", "schema": {...}}
        """
        name = data.get("name")
        schema = data.get("schema")
        if not name or not schema:
            raise HTTPException(status_code=400, detail="name and schema required")
        model = _registry.register(name, schema)
        return {"status": "registered", "name": name, "table": model.table_name,
                "version": model.version, "fields": len(model.fields)}

    @app.get("/api/schema")
    async def list_schemas():
        return {"models": _registry.list_models()}

    # ── 通用 CRUD API ──

    def _tenant(request):
        return getattr(request.state, 'tenant_id', 'default')

    @app.post("/api/v1/{entity}")
    async def create_entity(entity: str, request: StarletteRequest):
        data = await request.json()
        tenant = request.state.tenant_id
        result = _crud.create(entity, data, tenant_id=_tenant(request))
        if result is None:
            raise HTTPException(status_code=400, detail=f"Entity '{entity}' not registered")
        return result

    @app.get("/api/v1/{entity}/{record_id}")
    async def read_entity(entity: str, record_id: str, request: StarletteRequest):
        result = _crud.read(entity, record_id, tenant_id=_tenant(request))
        if result is None:
            raise HTTPException(status_code=404, detail="Not found")
        return result

    @app.patch("/api/v1/{entity}/{record_id}")
    async def update_entity(entity: str, record_id: str, request: StarletteRequest):
        data = await request.json()
        result = _crud.update(entity, record_id, data, tenant_id=_tenant(request))
        if result is None:
            raise HTTPException(status_code=404, detail="Not found")
        return result

    @app.delete("/api/v1/{entity}/{record_id}")
    async def delete_entity(entity: str, record_id: str, request: StarletteRequest):
        ok = _crud.delete(entity, record_id, tenant_id=request.state.tenant_id)
        if not ok:
            raise HTTPException(status_code=404, detail="Not found")
        return {"status": "deleted"}

    @app.get("/api/v1/{entity}")
    async def query_entity(entity: str, request: StarletteRequest,
                           sort: str = "created_at",
                           limit: int = 100, offset: int = 0):
        """查询: GET /api/v1/product?sort=price&limit=10&offset=0"""
        results = _crud.query(entity, sort=sort, limit=limit,
                              offset=offset, tenant_id=_tenant(request))
        total = _crud.count(entity, tenant_id=_tenant(request))
        return {"data": results, "total": total, "limit": limit, "offset": offset}

    # ── Colony Agent API ──

    @app.get("/api/agents")
    async def list_agents():
        if not _agents:
            return {"agents": []}
        return {
            "agents": [
                {"id": aid, "name": a.name, "role": a.role,
                 "goals": len(a.goal_keeper.list_goals()) if hasattr(a, 'goal_keeper') else 0}
                for aid, a in _agents.items()
            ]
        }

    @app.post("/api/agents/{agent_id}/perceive")
    async def agent_perceive(agent_id: str):
        if agent_id not in _agents:
            raise HTTPException(status_code=404, detail="Agent not found")
        agent = _agents[agent_id]
        try:
            from laap.sandbox._types import ProjectSnapshot, FileTreeState
            snap = ProjectSnapshot(root_path=str(get_laap_root()),
                file_tree=FileTreeState(total_files=0, total_lines=0))
            agent.perceive(snap)
            suggestion = agent.think()
            return {"agent_id": agent_id, "perceived": True, "suggestion": str(suggestion) if suggestion else None}
        except Exception as e:
            return {"agent_id": agent_id, "perceived": False, "error": str(e)}

    # ── 租户 API ──

    @app.post("/api/tenants")
    async def create_tenant(data: dict):
        tid = data.get("id", f"tenant-{uuid.uuid4().hex[:8]}")
        name = data.get("name", tid)
        tenant = _tenant_mgr.create(tid, name)
        return tenant

    @app.get("/api/tenants")
    async def list_tenants():
        return {"tenants": _tenant_mgr.list()}

    # ── GitHub Webhook ──

    @app.post("/api/github/webhook")
    async def github_webhook(request: Request):
        """接收 GitHub webhook 事件，自动触发 PR Review。"""
        try:
            payload = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON body")

        event = request.headers.get("x-github-event", "")
        if event != "pull_request":
            return {"status": "ignored", "event": event, "message": "only pull_request events processed"}

        action = payload.get("action", "")
        if action not in ("opened", "synchronize", "reopened"):
            return {"status": "ignored", "action": action, "message": "only opened/synchronize/reopened actions"}

        pr = payload.get("pull_request", {})
        repo = payload.get("repository", {})
        pr_number = pr.get("number")
        repo_full_name = repo.get("full_name", "")
        clone_url = repo.get("clone_url", "")
        head_sha = pr.get("head", {}).get("sha", "")
        base_ref = pr.get("base", {}).get("ref", "main")

        logger.info(f"GitHub PR #{pr_number} → {repo_full_name} ({action})")

        # 在后台线程中执行审查（避免 webhook 超时）
        import threading
        def _run_review():
            try:
                from laap.github.review_submitter import run_pr_review
                result = run_pr_review(
                    repo_url=clone_url,
                    pr_number=pr_number,
                    repo_full_name=repo_full_name,
                    head_sha=head_sha,
                    base_ref=base_ref,
                )
                logger.info(f"PR #{pr_number} 审查完成: {result.get('status', 'unknown')}")
            except Exception as e:
                logger.error(f"PR #{pr_number} 审查失败: {e}")

        threading.Thread(target=_run_review, daemon=True).start()

        return {
            "status": "accepted",
            "pr": pr_number,
            "repo": repo_full_name,
            "message": "PR review started in background",
        }

    @app.post("/api/github/review")
    async def trigger_manual_review(data: dict):
        """手动触发 PR Review（用于测试）。"""
        repo_url = data.get("repo_url", "")
        pr_number = data.get("pr_number", 0)
        repo_full_name = data.get("repo_full_name", "")
        head_sha = data.get("head_sha", "")
        base_ref = data.get("base_ref", "main")

        if not repo_url or not pr_number:
            raise HTTPException(status_code=400, detail="repo_url and pr_number required")

        try:
            from laap.github.review_submitter import run_pr_review
            result = run_pr_review(
                repo_url=repo_url,
                pr_number=pr_number,
                repo_full_name=repo_full_name,
                head_sha=head_sha,
                base_ref=base_ref,
            )
            return result
        except Exception as e:
            logger.error(f"手动审查失败: {e}")
            return {"status": "error", "error": str(e)}

    return app


def run_server(host: str = "0.0.0.0", port: int = 8910,
               db_path: str = "laap_saas.db",
               reload: bool = False, log_level: str = "info"):
    try:
        import uvicorn
    except ImportError:
        raise ImportError("需要 uvicorn: pip install uvicorn")

    logger.info(f"LAAP SaaS Runtime v2.0.0")
    logger.info(f"  监听: http://{host}:{port}")
    logger.info(f"  数据库: {db_path}")
    logger.info(f"  API:  /api/v1/{{entity}} 通用CRUD")
    logger.info(f"  Agent: {list(_agents.keys()) if _agents else '未加载'}")
    uvicorn.run("laap.saas.server.app:create_app", factory=True,
                host=host, port=port, log_level=log_level, reload=reload)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    run_server()
