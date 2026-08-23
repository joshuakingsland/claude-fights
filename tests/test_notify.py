import json
import os
import smtplib
import unittest
import urllib.error
from unittest import mock

import notify_email

BASE = {"RESEND_API_KEY": "", "NOTIFY_WEBHOOK_URL": "", "SMTP_USER": "",
        "SMTP_PASSWORD": "", "BET_EMAIL_TO": "", "NOTIFY_FROM": "",
        "SMTP_HOST": "", "SMTP_PORT": ""}


def _env(**overrides):
    return mock.patch.dict(os.environ, {**BASE, **overrides}, clear=False)


class ChannelSelectionTests(unittest.TestCase):
    """Whichever secret is set decides the channel, in a fixed order."""

    def test_resend_is_preferred_when_configured(self):
        with _env(RESEND_API_KEY="re_123", BET_EMAIL_TO="me@example.com",
                  NOTIFY_WEBHOOK_URL="https://ntfy.sh/x", SMTP_USER="u",
                  SMTP_PASSWORD="p"):
            with mock.patch.object(notify_email, "_post", return_value=200):
                self.assertEqual(notify_email.deliver("s", "b"), "resend")

    def test_a_webhook_is_used_when_there_is_no_resend_key(self):
        with _env(NOTIFY_WEBHOOK_URL="https://ntfy.sh/x", SMTP_USER="u",
                  SMTP_PASSWORD="p", BET_EMAIL_TO="me@example.com"):
            with mock.patch.object(notify_email, "_post", return_value=200):
                self.assertEqual(notify_email.deliver("s", "b"), "webhook")

    def test_smtp_is_the_last_resort(self):
        with _env(SMTP_USER="u", SMTP_PASSWORD="p", BET_EMAIL_TO="me@example.com"):
            with mock.patch.object(notify_email, "send_smtp", return_value=200):
                self.assertEqual(notify_email.deliver("s", "b"), "smtp")

    def test_nothing_configured_selects_no_channel(self):
        with _env():
            self.assertIsNone(notify_email.deliver("s", "b"))

    def test_a_resend_key_without_a_recipient_is_not_used(self):
        # Resend needs somewhere to send. Falling through is better than
        # posting a request that will be rejected.
        with _env(RESEND_API_KEY="re_123"):
            self.assertIsNone(notify_email.deliver("s", "b"))


class ResendTests(unittest.TestCase):
    def _capture(self, **overrides):
        seen = {}

        def fake(url, payload, headers):
            seen.update(url=url, payload=payload, headers=headers)
            return 200

        with _env(RESEND_API_KEY="re_123", BET_EMAIL_TO="me@example.com",
                  **overrides):
            with mock.patch.object(notify_email, "_post", side_effect=fake):
                notify_email.deliver("the subject", "the body")
        return seen

    def test_it_posts_to_resend_with_a_bearer_key(self):
        seen = self._capture()
        self.assertEqual(seen["url"], notify_email.RESEND_ENDPOINT)
        self.assertEqual(seen["headers"]["Authorization"], "Bearer re_123")

    def test_the_default_sender_needs_no_domain_setup(self):
        # The whole point of recommending Resend: one secret, no DNS.
        self.assertEqual(self._capture()["payload"]["from"],
                         notify_email.DEFAULT_RESEND_SENDER)

    def test_a_custom_sender_is_honoured(self):
        seen = self._capture(NOTIFY_FROM="alerts@example.com")
        self.assertEqual(seen["payload"]["from"], "alerts@example.com")

    def test_the_recipient_and_text_are_carried(self):
        payload = self._capture()["payload"]
        self.assertEqual(payload["to"], ["me@example.com"])
        self.assertEqual(payload["subject"], "the subject")
        self.assertEqual(payload["text"], "the body")


class WebhookTests(unittest.TestCase):
    def test_it_sends_keys_all_three_services_can_read(self):
        seen = {}
        with _env(NOTIFY_WEBHOOK_URL="https://ntfy.sh/topic"):
            with mock.patch.object(
                    notify_email, "_post",
                    side_effect=lambda u, p, h: seen.update(url=u, payload=p) or 200):
                notify_email.deliver("subj", "body")
        self.assertEqual(seen["url"], "https://ntfy.sh/topic")
        for key in ("title", "text", "content", "message"):
            self.assertIn(key, seen["payload"])

    def test_the_payload_is_serialisable(self):
        # _post json-encodes it; an unserialisable value would only surface
        # at runtime inside the failure handler.
        json.dumps({"title": "s", "text": "t", "content": "c", "message": "m"})


class FailureTests(unittest.TestCase):
    """The notifier runs as the `if: failure()` step and must never raise."""

    def _run_with(self, side_effect):
        with _env(RESEND_API_KEY="re_123", BET_EMAIL_TO="me@example.com"):
            with mock.patch.object(notify_email, "_post", side_effect=side_effect):
                return notify_email.main()

    def test_an_http_error_does_not_mask_the_real_failure(self):
        self.assertEqual(
            self._run_with(urllib.error.URLError("connection refused")), 0)

    def test_a_rejected_api_key_does_not_mask_the_real_failure(self):
        self.assertEqual(self._run_with(
            urllib.error.HTTPError("u", 401, "unauthorized", {}, None)), 0)

    def test_a_dead_smtp_password_does_not_mask_the_real_failure(self):
        # The original Gmail failure, kept as a regression test.
        with _env(SMTP_USER="u", SMTP_PASSWORD="p", BET_EMAIL_TO="me@example.com"):
            with mock.patch.object(
                    notify_email, "send_smtp",
                    side_effect=smtplib.SMTPAuthenticationError(
                        535, b"BadCredentials")):
                self.assertEqual(notify_email.main(), 0)

    def test_no_channel_configured_exits_clean(self):
        with _env():
            self.assertEqual(notify_email.main(), 0)

    def test_a_bad_smtp_port_does_not_raise(self):
        with _env(SMTP_USER="u", SMTP_PASSWORD="p",
                  BET_EMAIL_TO="me@example.com", SMTP_PORT="not-a-port"):
            self.assertEqual(notify_email.main(), 0)




class ErrorDetailTests(unittest.TestCase):
    """A provider's explanation must survive into the message.

    urllib's HTTPError stringifies to "HTTP Error 403: Forbidden" and throws
    the body away, but the body is where Resend explains the sending
    restriction. Without it a live failure is unguessable.
    """

    def _raise_http(self, code, body):
        import io
        return urllib.error.HTTPError(
            "https://api.resend.com/emails", code, "Forbidden", {},
            io.BytesIO(body.encode()))

    def test_the_response_body_is_carried_into_the_error(self):
        with mock.patch("urllib.request.urlopen",
                        side_effect=self._raise_http(
                            403, '{"message":"only send to your own address"}')):
            with self.assertRaises(urllib.error.URLError) as caught:
                notify_email._post("https://api.resend.com/emails", {}, {})
        self.assertIn("403", str(caught.exception))
        self.assertIn("only send to your own address", str(caught.exception))

    def test_an_empty_body_falls_back_to_the_reason(self):
        with mock.patch("urllib.request.urlopen",
                        side_effect=self._raise_http(403, "")):
            with self.assertRaises(urllib.error.URLError) as caught:
                notify_email._post("https://api.resend.com/emails", {}, {})
        self.assertIn("Forbidden", str(caught.exception))

    def test_main_still_exits_clean_on_an_http_error(self):
        with _env(RESEND_API_KEY="re_1", BET_EMAIL_TO="me@example.com"):
            with mock.patch("urllib.request.urlopen",
                            side_effect=self._raise_http(403, "nope")):
                self.assertEqual(notify_email.main(), 0)


if __name__ == "__main__":
    unittest.main()
