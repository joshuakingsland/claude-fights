import io
import json
import unittest
import urllib.error
from unittest import mock

import pandas as pd

import fetch_odds
import historical_odds as ho
import http_retry


def _http_error(code, retry_after=None):
    headers = {"Retry-After": retry_after} if retry_after else {}
    return urllib.error.HTTPError(
        "http://x", code, "boom", headers, io.BytesIO(b"detail"))


class _Response:
    """Minimal stand-in for the urlopen context manager."""

    headers = {"x-requests-remaining": "4000000", "x-requests-used": "1"}

    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class BackoffTests(unittest.TestCase):
    def test_delay_doubles_with_each_attempt(self):
        self.assertEqual([http_retry.backoff_seconds(i) for i in range(4)],
                         [2.0, 4.0, 8.0, 16.0])

    def test_delay_is_capped(self):
        self.assertLessEqual(http_retry.backoff_seconds(20), 120.0)

    def test_the_servers_retry_after_wins(self):
        self.assertEqual(http_retry.backoff_seconds(0, retry_after="30"), 30.0)

    def test_a_retry_after_above_the_cap_is_still_capped(self):
        self.assertEqual(http_retry.backoff_seconds(0, retry_after="9999"), 120.0)

    def test_an_unparseable_retry_after_falls_back_to_exponential(self):
        self.assertEqual(http_retry.backoff_seconds(1, retry_after="soon"), 4.0)


class RequestRetryTests(unittest.TestCase):
    """The key is shared with another repository, so 429s are expected.

    Before this, the first collision raised and killed the whole run. That was
    never a correctness problem - the manifest only records what actually
    returned, so a resume retried anything unattempted - but a backfill of tens
    of thousands of requests dying partway is expensive in job time.
    """

    WHEN = pd.Timestamp("2024-01-01T00:00:00Z")

    def _call(self, side_effect, attempts=6):
        slept = []
        with mock.patch("urllib.request.urlopen", side_effect=side_effect):
            payload, headers = ho._request(
                "k", self.WHEN, "us", attempts=attempts, sleep=slept.append)
        return payload, headers, slept

    def test_a_rate_limit_is_waited_out_and_the_request_succeeds(self):
        payload, _, slept = self._call(
            [_http_error(429), _http_error(429), _Response({"data": []})])
        self.assertEqual(payload, {"data": []})
        self.assertEqual(slept, [2.0, 4.0])

    def test_the_servers_retry_after_is_obeyed(self):
        _, _, slept = self._call(
            [_http_error(429, retry_after="7"), _Response({"data": []})])
        self.assertEqual(slept, [7.0])

    def test_a_permanent_error_is_not_retried(self):
        # A bad key or a date outside the plan will never succeed; retrying it
        # just burns wall clock on every remaining request.
        with self.assertRaises(RuntimeError) as caught:
            self._call([_http_error(401)])
        self.assertIn("401", str(caught.exception))

    def test_a_permanent_error_sleeps_zero_times(self):
        slept = []
        with mock.patch("urllib.request.urlopen", side_effect=[_http_error(422)]):
            with self.assertRaises(RuntimeError):
                ho._request("k", self.WHEN, "us", sleep=slept.append)
        self.assertEqual(slept, [])

    def test_server_errors_are_retried_too(self):
        payload, _, slept = self._call([_http_error(503), _Response({"data": []})])
        self.assertEqual(payload, {"data": []})
        self.assertEqual(len(slept), 1)

    def test_a_dropped_connection_is_retried(self):
        payload, _, slept = self._call(
            [urllib.error.URLError("connection reset"), _Response({"data": []})])
        self.assertEqual(payload, {"data": []})
        self.assertEqual(len(slept), 1)

    def test_giving_up_raises_the_last_failure_rather_than_returning_junk(self):
        # Raising is deliberate. The manifest treats a recorded result as
        # terminal, so writing a rate-limit response into it would turn a
        # transient collision into a permanent hole no resume would refill.
        with self.assertRaises(RuntimeError) as caught:
            self._call([_http_error(429)] * 3, attempts=3)
        self.assertIn("429", str(caught.exception))

    def test_it_stops_after_the_configured_number_of_attempts(self):
        slept = []
        with mock.patch("urllib.request.urlopen", side_effect=[_http_error(429)] * 9):
            with self.assertRaises(RuntimeError):
                ho._request("k", self.WHEN, "us", attempts=4, sleep=slept.append)
        self.assertEqual(len(slept), 3)  # sleeps between attempts, not after the last


class LiveFetchRetryTests(unittest.TestCase):
    """The scheduled capture shares the key with the backfill.

    fetch_odds had no error handling at all - a bare urlopen. That was
    survivable while this repo was the only heavy consumer of the key.
    Backfilling tens of thousands of historical requests against the same key
    makes a collision on the six-hourly run likely, and a collision there means
    a failed workflow and a stale card.
    """

    def test_a_rate_limit_on_the_live_endpoint_is_waited_out(self):
        calls = [_http_error(429), _Response({"ok": True})]
        with mock.patch("urllib.request.urlopen", side_effect=calls):
            with mock.patch("http_retry.time.sleep"):
                got = fetch_odds.fetch_region("k", "us")
        self.assertEqual(got, {"ok": True})

    def test_a_permanent_error_on_the_live_endpoint_still_raises(self):
        with mock.patch("urllib.request.urlopen", side_effect=[_http_error(401)]):
            with self.assertRaises(RuntimeError):
                fetch_odds.fetch_region("k", "us")


if __name__ == "__main__":
    unittest.main()
