from __future__ import annotations


# Optional webhook/Slack/email push on episode OPEN/escalate.
def notify(event):
    raise NotImplementedError("notifier")
