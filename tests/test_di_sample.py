from scripts.di_sample import FilteredSuite


def test_filtered_suite_drops_only_declared_catalogs():
    rows = [{"id": str(i), "_evaluation": {"catalog_id": i}} for i in (1, 45, 57, 61)]
    class Suite:
        def rows(self, **kwargs):
            assert kwargs == {"apply_exclusions": True}
            yield from rows
    filtered = list(FilteredSuite(Suite(), {45, 61}).rows(apply_exclusions=True))
    assert [r["id"] for r in filtered] == ["1", "57"]
    assert len(rows) == 4
    assert list(FilteredSuite(Suite(), set()).rows(apply_exclusions=True)) == rows
