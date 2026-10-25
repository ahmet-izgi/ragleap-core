"""
Outbound action senders for the autonomy gate (phase 1 of "broad actionable").

send_webhook / send_slack / send_email / send_crm_update / send_calendar_event
are dispatched from core.autonomy._send_via_channel(), so every mode,
allowlist, sensitive-role rule, approval step and log in execute_or_request()
applies to them unchanged.

Security model: a model (or a customer message hijacking it) must never
choose the destination.
  - webhook: `target` is a NAME looked up in WEBHOOK_TARGETS (name=url,...)
    from .env, never a raw URL. HTTPS only; hosts resolving to non-public
    addresses (private, loopback, link-local, cloud metadata) are refused;
    redirects are not followed.
  - slack: posts only to SLACK_WEBHOOK_URL from .env (must be hooks.slack.com).
  - email: recipient must match EMAIL_ALLOWED_RECIPIENTS (addresses or
    @domain entries); the default is empty, so nothing sends until configured.
  - crm: updates properties of ONE existing HubSpot CRM record (Private App
    access token from HUBSPOT_ACCESS_TOKEN). `target` is ignored; the object
    type and every property name must each be on an operator-configured
    allowlist (HUBSPOT_CRM_UPDATE_OBJECTS / HUBSPOT_CRM_UPDATE_PROPERTIES),
    and both default to EMPTY, so nothing is ever written until configured.
  - calendar: creates the event on GOOGLE_CALENDAR_ID (Google Calendar
    events.insert, bring-your-own OAuth client like
    core/integrations/gmail_connector.py); `target` is ignored, and every
    attendee must match EMAIL_ALLOWED_RECIPIENTS - one refusal cancels the
    whole event, the default allowlist is empty, so nothing schedules until
    configured.
Senders never raise and never log secret URLs; they return True/False.
"""
import ipaddress
import logging
import os
import re
import smtplib
import socket
from datetime import datetime
from email.message import EmailMessage
from typing import Dict, Optional
from urllib.parse import quote, urlparse

import requests

logger = logging.getLogger(__name__)

HTTP_TIMEOUT = 15
MAX_TEXT = 3000


def _webhook_targets() -> Dict[str, str]:
    targets = {}
    for item in os.environ.get("WEBHOOK_TARGETS", "").split(","):
        if "=" not in item:
            continue
        name, url = item.split("=", 1)
        name, url = name.strip(), url.strip()
        if name and url:
            targets[name] = url
    return targets


def _is_public_https_url(url: str) -> bool:
    """HTTPS only, and every resolved address must be globally routable."""
    try:
        p = urlparse(url)
        if p.scheme != "https" or not p.hostname:
            return False
        infos = socket.getaddrinfo(p.hostname, p.port or 443, proto=socket.IPPROTO_TCP)
        if not infos:
            return False
        for info in infos:
            ip = ipaddress.ip_address(info[4][0].split("%")[0])
            if not ip.is_global or ip.is_multicast:
                return False
        return True
    except Exception:
        return False


def send_webhook(target: str, content: str) -> bool:
    try:
        name = (target or "").strip()
        url = _webhook_targets().get(name)
        if not url:
            logger.error(f"Webhook target '{name}' is not in WEBHOOK_TARGETS - refused")
            return False
        if not _is_public_https_url(url):
            logger.error(f"Webhook target '{name}' is not a public https URL - refused")
            return False
        resp = requests.post(
            url, json={"text": (content or "")[:MAX_TEXT], "source": "ragleap"},
            timeout=HTTP_TIMEOUT, allow_redirects=False,
        )
        return 200 <= resp.status_code < 300
    except Exception:
        logger.error("Webhook send failed")
        return False


SLACK_PREFIX = "https://hooks.slack.com/"


def send_slack(target: str, content: str) -> bool:
    """`target` is only a label; the destination is always SLACK_WEBHOOK_URL."""
    try:
        url = os.environ.get("SLACK_WEBHOOK_URL", "").strip()
        if not url.startswith(SLACK_PREFIX):
            logger.error("SLACK_WEBHOOK_URL is not set to a hooks.slack.com URL - refused")
            return False
        resp = requests.post(
            url, json={"text": (content or "")[:MAX_TEXT]},
            timeout=HTTP_TIMEOUT, allow_redirects=False,
        )
        return 200 <= resp.status_code < 300
    except Exception:
        logger.error("Slack send failed")
        return False


_FORBIDDEN_ADDR_CHARS = set(" \t\r\n,;<>\"'()[]\\")


def _is_plain_address(addr: str) -> bool:
    """One plain local@domain.tld address. Deliberately regex-free (a regex here
    was flagged by CodeQL as polynomial-ReDoS on model-influenced input) and
    length-capped first, so it runs in linear time."""
    if not addr or len(addr) > 254:
        return False
    for c in addr:
        if c in _FORBIDDEN_ADDR_CHARS or ord(c) < 33 or ord(c) > 126:
            return False
    local, sep, domain = addr.partition("@")
    if not sep or not local or len(local) > 64 or "@" in domain:
        return False
    labels = domain.split(".")
    return len(labels) >= 2 and all(labels)


def _recipient_allowed(addr: str) -> bool:
    """Exactly one plain address, and it must be on EMAIL_ALLOWED_RECIPIENTS
    (full addresses, or '@domain.com' entries matching that exact domain)."""
    addr = (addr or "").strip().lower()
    if not _is_plain_address(addr):
        return False
    for entry in os.environ.get("EMAIL_ALLOWED_RECIPIENTS", "").split(","):
        entry = entry.strip().lower()
        if not entry:
            continue
        if entry.startswith("@"):
            if addr.endswith(entry):
                return True
        elif addr == entry:
            return True
    return False


def send_email(target: str, content: str) -> bool:
    """Optional first line 'Subject: ...' becomes the subject; rest is the body."""
    try:
        host = os.environ.get("SMTP_HOST", "").strip()
        user = os.environ.get("SMTP_USER", "").strip()
        from_addr = os.environ.get("SMTP_FROM", "").strip() or user
        if not host or not from_addr:
            logger.error("SMTP_HOST / SMTP_FROM not configured - email refused")
            return False
        if not _recipient_allowed(target):
            logger.error("Email recipient is not in EMAIL_ALLOWED_RECIPIENTS - refused")
            return False
        subject, body = "Message from RagLeap", content or ""
        first, _, rest = body.partition("\n")
        if first.lower().startswith("subject:"):
            subject = first[8:].strip().replace("\r", " ")[:200] or subject
            body = rest.lstrip("\n")
        msg = EmailMessage()
        msg["From"] = from_addr
        msg["To"] = target.strip()
        msg["Subject"] = subject
        msg.set_content(body[:MAX_TEXT * 4])
        port = int(os.environ.get("SMTP_PORT", "587") or 587)
        password = os.environ.get("SMTP_PASSWORD", "")
        if port == 465:
            with smtplib.SMTP_SSL(host, port, timeout=20) as smtp:
                if user:
                    smtp.login(user, password)
                smtp.send_message(msg)
        else:
            with smtplib.SMTP(host, port, timeout=20) as smtp:
                smtp.starttls()
                if user:
                    smtp.login(user, password)
                smtp.send_message(msg)
        return True
    except Exception:
        logger.error("Email send failed")
        return False


HUBSPOT_API_BASE = "https://api.hubapi.com"
CRM_ENV_KEYS = (
    "HUBSPOT_ACCESS_TOKEN",
    "HUBSPOT_CRM_UPDATE_OBJECTS",
    "HUBSPOT_CRM_UPDATE_PROPERTIES",
)
CRM_MAX_VALUE = 1000
_CRM_ID_RE = re.compile(r"[0-9]{1,19}")


def _crm_config() -> Optional[Dict[str, str]]:
    """The three HUBSPOT_* settings, or None when any is unset/empty - the
    allowlists default to EMPTY, so nothing is ever written until the owner
    configured the token AND both allowlists."""
    cfg = {k: os.environ.get(k, "").strip() for k in CRM_ENV_KEYS}
    return cfg if all(cfg.values()) else None


def _crm_allowlist(key: str) -> list:
    """The operator's comma-separated allowlist entries, stripped; membership
    itself is checked VERBATIM against these entries (see
    _crm_update_allowed)."""
    return [e.strip() for e in os.environ.get(key, "").split(",") if e.strip()]


def _parse_crm_update(content: str) -> Optional[Dict]:
    """Parse the documented update shape out of an action's `content`:
    'Key: value' lines (order-independent), case-insensitive key -
    'Object:', 'Id:', one or more
    'Set: <name> = <value>'. Object and Id must each appear exactly once, with
    exactly one space after the colon and a non-empty value taken VERBATIM
    (no trimming or normalization: ' contacts ' is not 'contacts'). Id is 1-19
    digits (a HubSpot record id); every Set line's property name and value
    must be non-empty with exactly one space around the '=', a value is
    at most CRM_MAX_VALUE characters, and a property name may appear at most
    once - a repeated one refuses the whole update, never last-value-wins.
    Anything malformed is None. No I/O."""
    object_type = record_id = None
    properties: Dict[str, str] = {}
    for line in (content or "").replace("\r\n", "\n").split("\n"):
        if not line:
            continue
        key, _, value = line.partition(":")
        field = key.strip().lower()
        if field not in ("object", "id", "set"):
            return None
        # exactly one space after the colon; the rest is verbatim
        if not value.startswith(" ") or value.startswith("  "):
            return None
        value = value[1:]
        if field == "object":
            if object_type is not None or not value:
                return None
            object_type = value
        elif field == "id":
            if record_id is not None or not _CRM_ID_RE.fullmatch(value):
                return None
            record_id = value
        else:
            name, sep, prop_value = value.partition(" = ")
            if not sep or not name or not prop_value:
                return None
            if len(prop_value) > CRM_MAX_VALUE:
                return None
            if name in properties:
                return None
            properties[name] = prop_value
    if not object_type or record_id is None or not properties:
        return None
    return {"object": object_type, "id": record_id, "properties": properties}


def _crm_update_allowed(update: Dict) -> bool:
    """The parsed object type and EVERY property name must each sit VERBATIM
    on the operator's allowlists - no case-folding, no trimming, no Unicode
    normalization: a near-miss name is refused, never silently mapped onto an
    allowlisted one."""
    if update["object"] not in _crm_allowlist("HUBSPOT_CRM_UPDATE_OBJECTS"):
        return False
    allowed = _crm_allowlist("HUBSPOT_CRM_UPDATE_PROPERTIES")
    return all(name in allowed for name in update["properties"])


def send_crm_update(target: str, content: str) -> bool:
    """Update properties of ONE existing HubSpot CRM record (CRM API v3
    objects PATCH). `target` is ignored on purpose - the account comes from
    HUBSPOT_ACCESS_TOKEN, never from the model; the object type and property
    names must each be on the operator's allowlists. Update only: a not-found
    record is a failure, never an upsert, and no create/merge/archive write
    exists here. Returns True only on a 2xx response; never raises and never
    logs the token, the request body, any response body, or any exception
    detail."""
    try:
        cfg = _crm_config()
        if not cfg:
            logger.error("HUBSPOT_* settings incomplete - CRM update refused")
            return False
        update = _parse_crm_update(content)
        if not update:
            logger.error("CRM update content malformed - refused")
            return False
        if not _crm_update_allowed(update):
            logger.error("CRM object/property outside the operator allowlists - refused")
            return False
        url = (f"{HUBSPOT_API_BASE}/crm/v3/objects/"
               f"{quote(update['object'], safe='')}/{update['id']}")
        resp = requests.patch(
            url, json={"properties": update["properties"]},
            timeout=HTTP_TIMEOUT, allow_redirects=False,
            headers={"Authorization": f"Bearer {cfg['HUBSPOT_ACCESS_TOKEN']}"},
        )
        if not 200 <= resp.status_code < 300:
            logger.error(f"CRM update failed with status {resp.status_code}")
            return False
        return True
    except Exception:
        logger.error("CRM update request error - CRM update refused")
        return False


GOOGLE_TOKEN_URI = "https://oauth2.googleapis.com/token"
CALENDAR_ENV_KEYS = (
    "GOOGLE_CALENDAR_ID",
    "GOOGLE_CALENDAR_CLIENT_ID",
    "GOOGLE_CALENDAR_CLIENT_SECRET",
    "GOOGLE_CALENDAR_REFRESH_TOKEN",
)


def _calendar_config() -> Optional[Dict[str, str]]:
    """The four GOOGLE_CALENDAR_* settings, or None when any is unset -
    nothing is ever scheduled until the owner configured all of them."""
    cfg = {k: os.environ.get(k, "").strip() for k in CALENDAR_ENV_KEYS}
    return cfg if all(cfg.values()) else None


def _parse_rfc3339(value: str) -> Optional[datetime]:
    """A date-time with an explicit UTC offset ('Z' counts). Naive
    strings are refused: there is no owner-chosen default zone to apply."""
    v = (value or "").strip()
    if v[-1:] in ("Z", "z"):
        v = v[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(v)
    except ValueError:
        return None
    return dt if dt.tzinfo is not None else None


def _parse_event(content: str) -> Optional[Dict]:
    """Parse the documented event shape out of an action's `content`: leading
    'Key: value' lines, case-insensitive - 'Title:', 'Start:', 'End:',
    'Attendees:' (comma-separated), 'Description:' (the rest of the content).
    Title/Start/End are required; Start/End must be a date-time with an
    explicit UTC offset (the parser accepts datetime.fromisoformat's forms;
    anything Google's API rejects fails closed at insert), End strictly
    after Start; every field except Description may appear once.
    Anything malformed is None. No I/O."""
    title = start_s = end_s = None
    have_attendees = False
    attendees: list = []
    desc_parts: list = []
    in_desc = False
    for line in (content or "").replace("\r\n", "\n").split("\n"):
        if in_desc:
            desc_parts.append(line)
            continue
        if not line.strip():
            continue
        key, _, value = line.partition(":")
        field = key.strip().lower()
        value = value.strip()
        if field == "title" and title is None:
            # empty value consumes the field, like Start/End below
            title = value.replace("\r", " ")[:200]
        elif field == "start" and start_s is None:
            start_s = value
        elif field == "end" and end_s is None:
            end_s = value
        elif field == "attendees" and not have_attendees:
            have_attendees = True
            attendees = [a.strip() for a in value.split(",") if a.strip()]
        elif field == "description":
            in_desc = True
            desc_parts = [value]
        else:
            return None
    if not title or not start_s or not end_s:
        return None
    start, end = _parse_rfc3339(start_s), _parse_rfc3339(end_s)
    if start is None or end is None or end <= start:
        return None
    return {"title": title, "start": start_s, "end": end_s,
            "attendees": attendees,
            "description": "\n".join(desc_parts).strip()[:MAX_TEXT * 4]}


def _attendees_allowed(attendees: list) -> bool:
    """Every attendee must pass the email allowlist; ONE refusal cancels the
    whole event - a bad member is never silently dropped."""
    return all(_recipient_allowed(a) for a in attendees)


def _google_access_token(cfg: Dict[str, str]) -> Optional[str]:
    """Exchange the owner's refresh token for a short-lived access token
    (bring-your-own OAuth client, core/integrations/gmail_connector.py's
    model). None on any failure it can see; transport errors propagate to
    the sender's catch-all. The token is never logged."""
    resp = requests.post(
        GOOGLE_TOKEN_URI,
        data={"client_id": cfg["GOOGLE_CALENDAR_CLIENT_ID"],
              "client_secret": cfg["GOOGLE_CALENDAR_CLIENT_SECRET"],
              "refresh_token": cfg["GOOGLE_CALENDAR_REFRESH_TOKEN"],
              "grant_type": "refresh_token"},
        timeout=HTTP_TIMEOUT, allow_redirects=False,
    )
    if resp.status_code != 200:
        logger.error(f"Google token exchange failed with status {resp.status_code}")
        return None
    try:
        token = (resp.json() or {}).get("access_token")
    except Exception:
        token = None
    if not token:
        logger.error("Google token exchange returned no access_token")
        return None
    return token


def send_calendar_event(target: str, content: str) -> bool:
    """Create one Google Calendar event (events.insert) on the owner's
    configured calendar. `target` is ignored on purpose - the calendar id
    comes from GOOGLE_CALENDAR_ID, never from the model. Returns True only
    on a 2xx response; never raises and never logs a token, a response body
    or exception detail."""
    try:
        cfg = _calendar_config()
        if not cfg:
            logger.error("GOOGLE_CALENDAR_* settings incomplete - calendar event refused")
            return False
        event = _parse_event(content)
        if not event:
            logger.error("Calendar event content malformed - refused")
            return False
        if not _attendees_allowed(event["attendees"]):
            logger.error("Calendar attendee is not in EMAIL_ALLOWED_RECIPIENTS - event refused")
            return False
        token = _google_access_token(cfg)
        if not token:
            return False
        url = (f"https://www.googleapis.com/calendar/v3/calendars/"
               f"{quote(cfg['GOOGLE_CALENDAR_ID'], safe='')}/events?sendUpdates=all")
        body: Dict = {
            "summary": event["title"],
            "start": {"dateTime": event["start"]},
            "end": {"dateTime": event["end"]},
        }
        if event["attendees"]:
            body["attendees"] = [{"email": a} for a in event["attendees"]]
        if event["description"]:
            body["description"] = event["description"]
        resp = requests.post(url, json=body, timeout=HTTP_TIMEOUT,
                             allow_redirects=False,
                             headers={"Authorization": f"Bearer {token}"})
        if not 200 <= resp.status_code < 300:
            logger.error(f"Calendar insert failed with status {resp.status_code}")
            return False
        return True
    except Exception:
        logger.error("Calendar event failed")
        return False
