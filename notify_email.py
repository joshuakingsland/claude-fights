"""Send workflow failure notices, by whichever channel is configured.

Gmail was the original and the wrong choice for unattended automation. An app
password needs 2FA enabled, is invisible once created, expires or is revoked
without warning, and when it dies the only symptom is a second red step in a
workflow that was already failing. That happened here twice.

So the channel is chosen by whichever secret is set, in this order:

1. `RESEND_API_KEY` - Resend's HTTP API. The recommended replacement. One key,
   no SMTP, no app password, no 2FA dance, and the free tier is 3,000 emails a
   month against the handful this repository sends. It needs no domain and no
   DNS: leaving `NOTIFY_FROM` unset uses `onboarding@resend.dev`, which Resend
   allows without verification as long as you send to your own account address.
2. `NOTIFY_WEBHOOK_URL` - a plain JSON POST. Covers ntfy.sh, Slack and Discord
   incoming webhooks. Not email, but it is the zero-signup option.
3. `SMTP_USER` + `SMTP_PASSWORD` - the original path, kept because it works
   with any provider. Point `SMTP_HOST` at something that is not Gmail.

Nothing configured means the notice is printed and skipped, which is fine:
GitHub already emails the repository owner when a scheduled workflow fails, so
that is the floor rather than silence.

This runs as the `if: failure()` step. It never raises. A broken notifier must
not replace the error somebody needs to read with a stack trace about
notifications - which is exactly what the Gmail failure did.
"""

import json
import os
import smtplib
import sys
import urllib.error
import urllib.request
from email.message import EmailMessage

RESEND_ENDPOINT = "https://api.resend.com/emails"
# Resend accepts this sender with no domain set up, delivering only to the
# address that owns the account. That makes the whole thing one secret.
DEFAULT_RESEND_SENDER = "onboarding@resend.dev"
TIMEOUT_SECONDS = 30


def _env(name, default=""):
    """An environment value, treating blank as absent.

    A GitHub secret that exists but is empty arrives as "", which os.environ
    hands back in preference to the default - so `_env("SMTP_PORT", "587")`
    returned "" and int() raised inside the failure handler. Blank and unset
    mean the same thing here.
    """
    value = os.environ.get(name, "").strip()
    return value if value else default


def _post(url, payload, headers):
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        return response.status


def send_resend(subject, body, api_key, to_address, sender=None):
    """Email over Resend's HTTP API. No SMTP, no app password."""
    return _post(
        RESEND_ENDPOINT,
        {"from": sender or DEFAULT_RESEND_SENDER, "to": [to_address],
         "subject": subject, "text": body},
        {"Authorization": f"Bearer {api_key}"},
    )


def send_webhook(subject, body, url):
    """A JSON POST that ntfy, Slack and Discord all accept.

    The three of them read different keys out of the same object, so all three
    are sent rather than branching on which service the URL belongs to.
    """
    return _post(url, {"title": subject, "text": f"{subject}\n\n{body}",
                       "content": f"**{subject}**\n{body}", "message": body}, {})


def send_smtp(subject, body, host, port, user, password, to_address):
    """The original path, for any provider that is not Gmail."""
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = user
    message["To"] = to_address
    message.set_content(body)
    with smtplib.SMTP(host, port, timeout=TIMEOUT_SECONDS) as smtp:
        smtp.starttls()
        smtp.login(user, password)
        smtp.send_message(message)
    return 200


def deliver(subject, body):
    """Try the configured channel. Returns the channel used, or None."""
    to_address = _env("BET_EMAIL_TO")
    resend_key = _env("RESEND_API_KEY")
    webhook = _env("NOTIFY_WEBHOOK_URL")

    if resend_key and to_address:
        send_resend(subject, body, resend_key, to_address,
                    _env("NOTIFY_FROM") or None)
        return "resend"
    if webhook:
        send_webhook(subject, body, webhook)
        return "webhook"
    if _env("SMTP_USER") and _env("SMTP_PASSWORD") and to_address:
        send_smtp(subject, body, _env("SMTP_HOST", "smtp.gmail.com"),
                  int(_env("SMTP_PORT", "587")), _env("SMTP_USER"),
                  _env("SMTP_PASSWORD"), to_address)
        return "smtp"
    return None


def main():
    subject = sys.argv[1] if len(sys.argv) > 1 else "claude-fights workflow notice"
    body = sys.argv[2] if len(sys.argv) > 2 else "A claude-fights workflow needs attention."

    try:
        channel = deliver(subject, body)
    except (smtplib.SMTPException, urllib.error.URLError, OSError,
            ValueError) as failure:
        # Never raise. GitHub still emails the owner about the failed run, so
        # the worst case is a duplicate rather than silence.
        print(f"WARNING: could not send notification: {failure}")
        print(f"  undelivered subject: {subject}")
        print(f"  undelivered body: {body}")
        return 0

    if channel is None:
        print("notification skipped; no channel configured "
              "(set RESEND_API_KEY + BET_EMAIL_TO, or NOTIFY_WEBHOOK_URL)")
        print(f"  subject: {subject}")
    else:
        print(f"notification sent via {channel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
