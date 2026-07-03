import pandas as pd

from volpulse.data.gaps import find_gaps, suspicious_gaps


def test_continuous_bdays_have_no_gaps():
    idx = pd.bdate_range("2023-01-02", periods=20)
    assert find_gaps(idx) == []


def test_weekends_are_not_gaps():
    # Friday to Monday: calendar gap, but no missing business day.
    idx = pd.DatetimeIndex(["2023-01-05", "2023-01-06", "2023-01-09"])
    assert find_gaps(idx) == []


def test_single_holiday_is_small_gap():
    idx = pd.bdate_range("2023-01-02", periods=10)
    holiday = idx[4]
    gaps = find_gaps(idx.drop(holiday))
    assert len(gaps) == 1
    assert gaps[0].n_missing == 1
    assert gaps[0].start == holiday
    # A 1-day gap is a normal holiday, not suspicious.
    assert suspicious_gaps(idx.drop(holiday), max_gap_bdays=1) == []


def test_missing_week_is_suspicious():
    idx = pd.bdate_range("2023-01-02", periods=20)
    missing = idx[5:10]
    remaining = idx.drop(missing)

    gaps = suspicious_gaps(remaining, max_gap_bdays=1)
    assert len(gaps) == 1
    assert gaps[0].n_missing == 5
    assert gaps[0].start == missing[0]
    assert gaps[0].end == missing[-1]


def test_multiple_distinct_gaps():
    idx = pd.bdate_range("2023-01-02", periods=30)
    remaining = idx.drop(idx[3:5]).drop(idx[20:23])
    gaps = find_gaps(remaining)
    assert [g.n_missing for g in gaps] == [2, 3]


def test_short_index_has_no_gaps():
    assert find_gaps(pd.DatetimeIndex([])) == []
    assert find_gaps(pd.DatetimeIndex(["2023-01-02"])) == []
