"""The explanation layer is a product feature, so it gets tested like one."""

from vpa import explain


def test_every_reported_metric_has_an_explanation():
    reported = ["opening_2s", "closing_2s", "peak_value", "trough_value",
                "swing", "variability", "decay", "sustained_above"]
    for name in reported:
        assert name in explain.METRIC_GLOSSARY, f"{name} is reported but never explained"
        assert len(explain.METRIC_GLOSSARY[name]) > 30


def test_limits_section_states_what_tribe_cannot_do():
    text = explain.WHAT_IT_IS_NOT.lower()
    for claim in ("attention", "watch time", "sales"):
        assert claim in text


def test_llm_context_carries_the_caveats():
    ctx = explain.llm_context().lower()
    assert "fmri" in ctx
    assert "artefact" in ctx or "cut" in ctx


def test_full_text_covers_all_sections():
    text = explain.full_text()
    for _, (title, _) in explain.SECTIONS.items():
        assert title in text
