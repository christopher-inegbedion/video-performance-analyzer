from vpa import learn


def test_views_only_is_not_zero_engagement():
    """Unrecorded interactions must not read as observed zeros.

    A views-only row is the normal shape of a first metrics entry. Scoring it
    0.0 asserts nobody engaged, which is a real claim the data never made, and
    it biases every correlation fitted afterwards.
    """
    assert learn.engagement_rate(
        {"views": 779, "likes": None, "comments": None, "shares": None, "saves": None}
    ) is None


def test_genuine_zero_engagement_is_kept():
    """A recorded zero is real data and must survive."""
    assert learn.engagement_rate(
        {"views": 779, "likes": 0, "comments": 0, "shares": 0, "saves": 0}
    ) == 0.0


def test_partial_interactions_still_score():
    """Likes recorded, the rest absent: usable, treating the absent ones as 0."""
    assert learn.engagement_rate(
        {"views": 100, "likes": 5, "comments": None, "shares": None, "saves": None}
    ) == 0.05


def test_pending_metrics_are_reported_as_pending_not_absent():
    """Metrics recorded but not yet joinable must not read as 'none recorded'.

    A views-only entry, or one whose evaluation has not finished, sits in this
    state. Telling the user nothing was recorded sends them to re-enter data
    they already entered.
    """
    model = learn.LearnedModel(0, caveat="", usable=False, pending=1)
    text = model.summary_text()
    assert "not yet usable" in text
    assert "No published performance recorded yet" not in text
