"""
GuidedGenerator — structured output generation with constraint enforcement.

Main entry point for Aris cognitive control path 2.
Forces LLM output to conform to JSON Schema or GBNF grammar constraints,
with validation and automatic retry.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .constraints.json_schema import SchemaConstraintBuilder, PRESET_SCHEMAS
from .constraints.grammar_bnf import GrammarConstraintBuilder
from .constraints.memory_ref import MemoryRefConstraintBuilder
from .constraints.chain_of_thought import ChainOfThoughtConstraintBuilder
from .validators.schema_validator import (
    SchemaValidator,
    ValidationResult as SchemaValidationResult,
)
from .validators.format_validator import FormatValidator, FormatValidationResult
from .validators.content_validator import ContentValidator, ContentValidationResult

logger = logging.getLogger("guided_generator")

# ── Optional: try to import CognitiveBusState from the AGI module ──
try:
    import sys
    from laap.config.paths import get_laap_root
    sys.path.insert(0, str(get_laap_root() / "laap"))
    from agi.cognitive_bus import CognitiveBus, CognitiveStateSnapshot
    _COG_AVAILABLE = True
except ImportError:
    _COG_AVAILABLE = False
    CognitiveStateSnapshot = Any  # type: ignore


# ════════════════════════════════════════════════════════════
# Public Types
# ════════════════════════════════════════════════════════════

@dataclass
class ValidationResult:
    """Aggregated result from all validation steps."""
    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    schema_checks: SchemaValidationResult = field(
        default_factory=lambda: SchemaValidationResult(is_valid=True)
    )
    format_checks: FormatValidationResult = field(
        default_factory=lambda: FormatValidationResult(is_valid=True)
    )
    content_checks: ContentValidationResult = field(
        default_factory=lambda: ContentValidationResult(is_valid=True)
    )


@dataclass
class GenerationResult:
    """Complete result of a guided generation call."""
    text: str = ""
    parsed: Optional[Any] = None
    constraint_used: Dict[str, Any] = field(default_factory=dict)
    constraint_type: str = "none"
    validation: ValidationResult = field(default_factory=ValidationResult)
    retries: int = 0
    success: bool = False
    raw_response: Optional[Dict[str, Any]] = None


# ════════════════════════════════════════════════════════════
# Main Generator
# ════════════════════════════════════════════════════════════

class GuidedGenerator:
    """
    Generate LLM output with structural constraint enforcement.

    Supports three constraint modes:
      - "json":         Output must be valid JSON conforming to a schema
      - "memory_ref":   Output must include memory reference markers [...]
      - "reasoning":    Output must follow 思考→分析→结论 structure
      - "grammar":      Output must conform to a custom GBNF grammar
      - "none":         No constraints (pass-through)
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self._config: Dict[str, Any] = {}
        if config_path:
            self._load_config(config_path)

        self.content_validator = ContentValidator()
        logger.info(
            f"GuidedGenerator initialized"
            f"{' (cognitive bus available)' if _COG_AVAILABLE else ''}"
        )

    def _load_config(self, path: str) -> None:
        """Load configuration from a JSON file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                self._config = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to load config from {path}: {e}")
            self._config = {}

    # ════════════════════════════════════════════════════════
    # Main generation method
    # ════════════════════════════════════════════════════════

    def generate(
        self,
        prompt: str,
        constraint_type: str = "json",
        cognitive_state: Optional[Any] = None,
        llama_url: str = "http://localhost:8082",
        max_retries: int = 3,
        temperature: float = 0.3,
        n_predict: int = 512,
        **llama_kwargs,
    ) -> GenerationResult:
        """
        Generate text with format constraint enforcement.

        Args:
            prompt: The input prompt for the LLM.
            constraint_type: One of "json", "memory_ref", "reasoning",
                           "grammar", or "none".
            cognitive_state: Optional CognitiveStateSnapshot for state-aware
                           constraint selection.
            llama_url: URL of the llama.cpp completion endpoint.
            max_retries: Max generation retries on validation failure.
            temperature: Sampling temperature.
            n_predict: Max tokens to generate.
            **llama_kwargs: Additional kwargs passed to the LLM API.

        Returns:
            GenerationResult with generated text, parsed data, and validation.
        """
        # 1. Build constraint based on type and optional cognitive state
        constraint = self.build_constraint(constraint_type, cognitive_state)

        # 2. Build the request payload
        payload = {
            "prompt": prompt,
            "n_predict": n_predict,
            "temperature": temperature,
        }
        payload.update(constraint)
        payload.update(llama_kwargs)

        logger.debug(
            f"Generating with constraint_type='{constraint_type}', "
            f"constraint keys: {list(constraint.keys())}"
        )

        # 3. Generate with retry
        last_result = GenerationResult(
            constraint_used=constraint,
            constraint_type=constraint_type,
        )

        for attempt in range(max_retries + 1):
            # Call the LLM
            try:
                raw = self._call_llama(llama_url, payload)
            except Exception as e:
                logger.error(f"LLM call failed (attempt {attempt + 1}): {e}")
                last_result.validation.errors.append(f"LLM call failed: {e}")
                if attempt < max_retries:
                    time.sleep(1.0)
                    continue
                else:
                    last_result.success = False
                    return last_result

            text = raw.get("content", "")
            last_result.raw_response = raw
            last_result.text = text
            last_result.retries = attempt

            # Try to parse as JSON if applicable
            parsed = None
            if constraint_type == "json":
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError:
                    pass
            last_result.parsed = parsed

            # 4. Validate
            validation = self.validate(text, constraint, constraint_type)
            last_result.validation = validation

            if validation.is_valid:
                last_result.success = True
                logger.debug(
                    f"Generation succeeded on attempt {attempt + 1}"
                )
                return last_result

            logger.debug(
                f"Validation failed (attempt {attempt + 1}): "
                f"{validation.errors}"
            )

            if attempt < max_retries:
                time.sleep(0.5)

        # All retries exhausted
        last_result.success = False
        logger.warning(
            f"Generation failed after {max_retries + 1} attempts"
        )
        return last_result

    # ════════════════════════════════════════════════════════
    # Constraint building
    # ════════════════════════════════════════════════════════

    def build_constraint(
        self,
        dtype: str,
        state: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Build constraint parameters for llama.cpp.

        Args:
            dtype: Constraint type.
            state: Optional cognitive state for dynamic constraint selection.

        Returns:
            Dict with "grammar" or "json_schema" key, or empty dict.
        """
        if dtype == "json":
            return self._build_json_constraint(state)
        elif dtype == "memory_ref":
            return MemoryRefConstraintBuilder("optional").build()
        elif dtype == "reasoning":
            return ChainOfThoughtConstraintBuilder("basic").build()
        elif dtype == "grammar":
            return self._build_grammar_constraint(state)
        elif dtype == "none":
            return {}
        else:
            raise ValueError(
                f"Unknown constraint_type '{dtype}'. "
                f"Valid: json, memory_ref, reasoning, grammar, none"
            )

    def _build_json_constraint(
        self, state: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Select and build a JSON Schema constraint, optionally based on
        the cognitive state's attention focus.
        """
        schema_name = self._infer_schema_from_state(state)
        if schema_name:
            return SchemaConstraintBuilder.from_preset(schema_name).build()
        # Default to cognitive_report
        return SchemaConstraintBuilder.from_preset("cognitive_report").build()

    def _build_grammar_constraint(
        self, state: Optional[Any] = None
    ) -> Dict[str, str]:
        """
        Build a grammar constraint. Uses state to pick a grammar template.
        """
        # Default to a simple text grammar
        return GrammarConstraintBuilder.enum(["yes", "no"]).build()

    # ════════════════════════════════════════════════════════
    # State-aware schema inference
    # ════════════════════════════════════════════════════════

    def infer_constraint_type(
        self, state: Optional[Any] = None
    ) -> str:
        """
        Infer the appropriate constraint type from cognitive state.

        Maps attention focus → constraint type:
          - MEMORY     → "memory_ref"
          - PLANNING   → "json" (action_plan schema)
          - SELF       → "json" (self_report schema)
          - TASK       → "reasoning"
          - LEARNING   → "reasoning"
          - USER       → "none" (free-form)
          - ENVIRONMENT → "json"
          - IDLE       → "none"
        """
        if state is None:
            return "json"

        focus = None
        if hasattr(state, "attention") and hasattr(state.attention, "focus"):
            focus = str(state.attention.focus.value) if hasattr(
                state.attention.focus, "value"
            ) else str(state.attention.focus)
        elif isinstance(state, dict):
            attention = state.get("attention", {})
            if isinstance(attention, dict):
                focus = attention.get("focus", "")

        mapping = {
            "memory": "memory_ref",
            "planning": "json",
            "self": "json",
            "task": "reasoning",
            "learning": "reasoning",
            "user": "none",
            "environment": "json",
            "idle": "none",
        }
        return mapping.get(str(focus).lower() if focus else "", "json")

    def _infer_schema_from_state(
        self, state: Optional[Any] = None
    ) -> Optional[str]:
        """
        Infer the best JSON Schema preset from cognitive state.
        """
        if state is None:
            return "cognitive_report"

        focus = None
        if hasattr(state, "attention") and hasattr(state.attention, "focus"):
            focus = str(state.attention.focus.value) if hasattr(
                state.attention.focus, "value"
            ) else str(state.attention.focus)
        elif isinstance(state, dict):
            attention = state.get("attention", {})
            if isinstance(attention, dict):
                focus = attention.get("focus", "")

        schema_map = {
            "planning": "action_plan",
            "self": "self_model",
            "memory": "memory_retrieval",
            "environment": "cognitive_report",
            "task": "cognitive_report",
            "learning": "cognitive_report",
            "user": "cognitive_report",
        }
        return schema_map.get(str(focus).lower() if focus else "", None)

    # ════════════════════════════════════════════════════════
    # Validation
    # ════════════════════════════════════════════════════════

    def validate(
        self,
        output: str,
        constraint: Dict[str, Any],
        constraint_type: str,
    ) -> ValidationResult:
        """
        Validate generated output against the applied constraint.

        Runs up to three checks depending on constraint_type:
          1. Schema validation (for "json")
          2. Format validation (for "memory_ref", "reasoning", "grammar")
          3. Content validation (always)

        Returns:
            ValidationResult with per-check results and aggregated validity.
        """
        result = ValidationResult()

        # 1. Content validation (always)
        content_result = self.content_validator.validate(output)
        result.content_checks = content_result
        if not content_result.is_valid:
            result.errors.extend(content_result.errors)
            result.is_valid = False

        # 2. Schema or format validation
        if constraint_type == "json":
            schema = constraint.get("json_schema")
            schema_result = self._validate_json(output, schema)
            result.schema_checks = schema_result
            if not schema_result.is_valid:
                result.errors.extend(schema_result.errors)
                result.is_valid = False

        elif constraint_type == "memory_ref":
            fmt = FormatValidator.with_memory_ref_check()
            fmt_result = fmt.validate(output)
            result.format_checks = fmt_result
            if not fmt_result.is_valid:
                result.errors.extend(fmt_result.errors)
                result.is_valid = False

        elif constraint_type == "reasoning":
            fmt = FormatValidator.with_reasoning_structure()
            fmt_result = fmt.validate(output)
            result.format_checks = fmt_result
            if not fmt_result.is_valid:
                result.errors.extend(fmt_result.errors)
                result.is_valid = False

        # For "grammar" and "none", we rely on content validation only
        # since the LLM server enforces the grammar server-side.

        return result

    def _validate_json(
        self, output: str, schema: Optional[Dict[str, Any]]
    ) -> SchemaValidationResult:
        """
        Validate output as JSON, optionally against a schema.
        """
        return SchemaValidator(schema).validate(output)

    # ════════════════════════════════════════════════════════
    # LLM API call
    # ════════════════════════════════════════════════════════

    def _call_llama(
        self, url: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Call the llama.cpp completion API.

        Args:
            url: Server URL (e.g. http://localhost:8082).
            payload: JSON payload for /completion.

        Returns:
            Parsed response dict with at least "content" key.
        """
        import requests

        endpoint = f"{url.rstrip('/')}/completion"

        # Ensure we don't stream
        payload.setdefault("stream", False)

        resp = requests.post(
            endpoint,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()


# ════════════════════════════════════════════════════════════
# Convenience Functions
# ════════════════════════════════════════════════════════════

_DEFAULT_GENERATOR: Optional[GuidedGenerator] = None


def get_generator(config_path: Optional[str] = None) -> GuidedGenerator:
    """Get or create the default GuidedGenerator singleton."""
    global _DEFAULT_GENERATOR
    if _DEFAULT_GENERATOR is None:
        _DEFAULT_GENERATOR = GuidedGenerator(config_path)
    return _DEFAULT_GENERATOR


def generate_json(
    prompt: str,
    schema_name: str = "cognitive_report",
    llama_url: str = "http://localhost:8082",
    **kwargs,
) -> GenerationResult:
    """Quick helper: generate JSON constrained by a preset schema."""
    gen = get_generator()
    return gen.generate(
        prompt=prompt,
        constraint_type="json",
        llama_url=llama_url,
        **kwargs,
    )


def generate_reasoning(
    prompt: str,
    llama_url: str = "http://localhost:8082",
    **kwargs,
) -> GenerationResult:
    """Quick helper: generate structured reasoning output."""
    gen = get_generator()
    return gen.generate(
        prompt=prompt,
        constraint_type="reasoning",
        llama_url=llama_url,
        **kwargs,
    )
