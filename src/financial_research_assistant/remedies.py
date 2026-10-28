"""What to do about a failure only the user can fix.

A scheduled job that fails on a credential fails the same way every day until
someone acts, and the raw error ("SendRequest failed: Token has expired.") says
what broke but not where to go. The scheduler adds the matching line here to
the failure it reports. Matched on the error text, since that is all a failed
run leaves behind; the first match wins.
"""

from __future__ import annotations

import re

_FLEX_RENEW = (
    "IBKR Portal → Performance & Reports → Flex Queries → Flex Web Service "
    "Configuration → Generate New Token (longest expiry; IP blank or this "
    "server's). Then set IBKR_FLEX_TOKEN in /etc/fra/env and run "
    "`sudo systemctl restart fra`."
)

RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"Flex sync failed:.*Token has expired", re.I),
     "The IBKR Flex token has expired, so positions stop updating until it is "
     "renewed. " + _FLEX_RENEW),
    (re.compile(r"Flex sync failed:.*Token is invalid", re.I),
     "IBKR does not recognise the Flex token (mistyped, or replaced by a newer "
     "one). " + _FLEX_RENEW),
    (re.compile(r"Flex sync failed:.*IP restriction", re.I),
     "The Flex token is limited to other IP addresses. Regenerate it with the IP "
     "field blank or set to this server's. " + _FLEX_RENEW),
    (re.compile(r"Flex sync failed:.*Query is invalid", re.I),
     "IBKR does not know the Flex query id. Check IBKR_FLEX_QUERY_ID in "
     "/etc/fra/env against the Query ID shown under Flex Queries."),
    (re.compile(r"No Flex token|no Flex token", re.I),
     "IBKR_FLEX_TOKEN is not set on the server. " + _FLEX_RENEW),
    (re.compile(r"AuthenticationError|PermissionDeniedError|invalid x-api-key"
                r"|\b401\b.*(unauthori[sz]ed|invalid)", re.I),
     "The model provider rejected the server's credential, so scheduled reports "
     "and ideas cannot run. Replace the API key (or the signed-in session in "
     "auth.json) on the server and run `sudo systemctl restart fra`."),
]


def advice(error: str) -> str:
    """The fix for ``error`` in one paragraph, or "" when it isn't one the user
    has to make."""
    for pattern, text in RULES:
        if pattern.search(error or ""):
            return text
    return ""
