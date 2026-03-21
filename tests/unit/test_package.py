"""Smoke tests for package structure."""


def test_package_import() -> None:
    import ml_sfa

    assert isinstance(ml_sfa.__version__, str)
    assert len(ml_sfa.__version__) > 0


def test_submodule_imports() -> None:
    import ml_sfa.data
    import ml_sfa.evaluation
    import ml_sfa.models
    import ml_sfa.utils

    assert ml_sfa.models is not None
    assert ml_sfa.data is not None
    assert ml_sfa.evaluation is not None
    assert ml_sfa.utils is not None
