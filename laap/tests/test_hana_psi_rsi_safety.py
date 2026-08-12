from laap.agi.code_evolution import CodeMutation, CodeTarget, SafetyGuard


def mutation(path: str) -> CodeMutation:
    code = "def f():\n    return 1\n"
    return CodeMutation(
        target=CodeTarget(file_path=path, function_name="f", current_code=code),
        original_code=code,
        mutated_code=code,
    )


def test_only_hana_psi_runtime_is_whitelisted_for_integration_rsi():
    allowed, reason = SafetyGuard.validate_mutation(
        mutation("laap/integrations/hana_psi_runtime.py")
    )
    assert allowed is True, reason


def test_acceptance_oracle_cannot_be_self_modified():
    allowed, _ = SafetyGuard.validate_mutation(
        mutation("laap/integrations/hana_psi_contract.py")
    )
    assert allowed is False


def test_neighboring_integration_files_remain_outside_rsi_scope():
    allowed, _ = SafetyGuard.validate_mutation(
        mutation("laap/integrations/other_adapter.py")
    )
    assert allowed is False
