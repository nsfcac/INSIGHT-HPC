from __future__ import annotations

import re
from typing import Optional

# Matches: rpc-91-5, rpg-93-2, pdu-91-3, etc.  Captures the middle numeric segment.
_RACK_RE = re.compile(r'^[a-zA-Z]+-(\d+)-\d+', re.IGNORECASE)

# Extract the rack number from a hostname.
def rack_id(hostname: str) -> Optional[int]:
    if not hostname:
        return None
    m = _RACK_RE.match(str(hostname).strip())
    return int(m.group(1)) if m else None
