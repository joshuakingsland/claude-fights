"""Send workflow status emails from GitHub Actions."""

import os
import smtplib
import sys
from email.message import EmailMessage


def _env(name, default=""):
    return os.environ.get(name, default).strip()


def main():
    required = ["SMTP_USER", "SMTP_PASSWORD", "BET_EMAIL_TO"]
    missing = [name for name in required if not _env(name)]
    if missing:
        print(f"email skipped; missing secrets: {', '.join(missing)}")
        return 0

    subject = sys.argv[1] if len(sys.argv) > 1 else "claude-fights workflow notice"
    body = sys.argv[2] if len(sys.argv) > 2 else "A claude-fights workflow needs attention."
    host = _env("SMTP_HOST", "smtp.gmail.com")
    port = int(_env("SMTP_PORT", "587"))

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = _env("SMTP_USER")
    message["To"] = _env("BET_EMAIL_TO")
    message.set_content(body)

    with smtplib.SMTP(host, port, timeout=30) as smtp:
        smtp.starttls()
        smtp.login(_env("SMTP_USER"), _env("SMTP_PASSWORD"))
        smtp.send_message(message)
    print("email sent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
