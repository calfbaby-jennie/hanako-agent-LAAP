"""Tests for laap.config.paths."""

from __future__ import annotations

from pathlib import Path

import pytest

from laap.config.paths import (
    get_aris_brain_dir,
    get_cache_dir,
    get_hana_home,
    get_hermes_root,
    get_laap_home,
    get_laap_root,
    get_logs_dir,
    get_models_dir,
    get_state_dir,
    get_video_dir,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """每个测试前清理可能影响路径解析的环境变量。"""
    for key in (
        "LAAP_ROOT",
        "LAAP_HOME",
        "HANA_HOME",
        "ARIS_BRAIN_DIR",
        "HERMES_ROOT",
        "LAAP_STATE_DIR",
        "LAAP_CACHE_DIR",
        "LAAP_MODELS_DIR",
        "LAAP_LOGS_DIR",
        "LAAP_VIDEO_DIR",
        "XDG_STATE_HOME",
        "XDG_CACHE_HOME",
    ):
        monkeypatch.delenv(key, raising=False)


def test_all_functions_return_path_objects(tmp_path, monkeypatch):
    """所有目录函数在正常情境下都应返回 pathlib.Path 对象。"""
    monkeypatch.setenv("LAAP_ROOT", str(tmp_path / "laap"))
    monkeypatch.setenv("HERMES_ROOT", str(tmp_path / "hermes"))
    monkeypatch.setenv("LAAP_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("LAAP_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("LAAP_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("LAAP_LOGS_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("LAAP_VIDEO_DIR", str(tmp_path / "video"))

    assert isinstance(get_laap_root(), Path)
    assert isinstance(get_laap_home(), Path)
    assert isinstance(get_hana_home(), Path)
    assert isinstance(get_aris_brain_dir(), Path)
    assert isinstance(get_hermes_root(), Path)
    assert isinstance(get_state_dir(), Path)
    assert isinstance(get_cache_dir(), Path)
    assert isinstance(get_models_dir(), Path)
    assert isinstance(get_logs_dir(), Path)
    assert isinstance(get_video_dir(), Path)


def test_env_variables_override_defaults(tmp_path, monkeypatch):
    """环境变量能够覆盖默认路径。"""
    overrides = {
        "LAAP_ROOT": tmp_path / "laap",
        "HERMES_ROOT": tmp_path / "hermes",
        "LAAP_STATE_DIR": tmp_path / "state",
        "LAAP_CACHE_DIR": tmp_path / "cache",
        "LAAP_MODELS_DIR": tmp_path / "models",
        "LAAP_LOGS_DIR": tmp_path / "logs",
        "LAAP_VIDEO_DIR": tmp_path / "video",
    }
    for key, value in overrides.items():
        monkeypatch.setenv(key, str(value))

    assert get_laap_root() == overrides["LAAP_ROOT"]
    assert get_hermes_root() == overrides["HERMES_ROOT"]
    assert get_state_dir() == overrides["LAAP_STATE_DIR"]
    assert get_cache_dir() == overrides["LAAP_CACHE_DIR"]
    assert get_models_dir() == overrides["LAAP_MODELS_DIR"]
    assert get_logs_dir() == overrides["LAAP_LOGS_DIR"]
    assert get_video_dir() == overrides["LAAP_VIDEO_DIR"]


def test_runtime_home_overrides(tmp_path, monkeypatch):
    laap_home = tmp_path / "laap-home"
    hana_home = tmp_path / "hana-home"
    monkeypatch.setenv("LAAP_HOME", str(laap_home))
    monkeypatch.setenv("HANA_HOME", str(hana_home))

    assert get_laap_home() == laap_home
    assert get_hana_home() == hana_home
    assert get_state_dir() == laap_home / "state"
    assert get_cache_dir() == laap_home / "cache"
    assert get_logs_dir() == laap_home / "logs"
    assert get_aris_brain_dir() == laap_home / "aris_brain"


def test_xdg_env_overrides_platform_defaults(tmp_path, monkeypatch):
    """XDG_STATE_HOME / XDG_CACHE_HOME 优先于平台默认目录。"""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg_state"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg_cache"))

    assert get_state_dir() == tmp_path / "xdg_state" / "laap"
    assert get_cache_dir() == tmp_path / "xdg_cache" / "laap"


def test_get_laap_root_derived_from_file_location(monkeypatch):
    """未设置 LAAP_ROOT 时，应能从当前文件位置推导出项目根目录。"""
    monkeypatch.delenv("LAAP_ROOT", raising=False)
    root = get_laap_root()
    assert isinstance(root, Path)
    assert root == Path(__file__).resolve().parent.parent.parent


def test_get_hermes_root_from_env(tmp_path, monkeypatch):
    """HERMES_ROOT 环境变量可直接指定 Hermes 根目录。"""
    hermes = tmp_path / "custom_hermes"
    monkeypatch.setenv("HERMES_ROOT", str(hermes))
    assert get_hermes_root() == hermes


def test_get_hermes_root_returns_none_when_not_found(tmp_path, monkeypatch):
    """未找到 Hermes 时返回 None 而不是抛出异常。"""
    monkeypatch.delenv("HERMES_ROOT", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    # 确保默认子目录也不存在
    monkeypatch.setenv("LAAP_ROOT", str(tmp_path / "no_hermes_here"))
    assert get_hermes_root() is None


def test_default_models_dir(tmp_path, monkeypatch):
    """默认模型目录位于项目根目录的 public/models 下。"""
    monkeypatch.setenv("LAAP_ROOT", str(tmp_path / "laap"))
    assert get_models_dir() == tmp_path / "laap" / "public" / "models"


def test_default_logs_dir(tmp_path, monkeypatch):
    """默认日志目录位于用户 LAAP_HOME 下。"""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    assert get_logs_dir() == tmp_path / ".laap" / "logs"


def test_default_video_dir(tmp_path, monkeypatch):
    """默认 TUI 视频目录位于项目根目录的 .trae/specs/... 下。"""
    monkeypatch.setenv("LAAP_ROOT", str(tmp_path / "laap"))
    assert get_video_dir() == (
        tmp_path / "laap" / ".trae" / "specs" / "integrate-aether-orchestration-into-laap"
    )
