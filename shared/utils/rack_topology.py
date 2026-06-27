from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# Matches: rpc-91-5, rpg-93-2, pdu-91-3, etc.  Captures the middle numeric segment.
RACK_HOSTNAME_PATTERN = re.compile(r"^[a-zA-Z]+-(\d+)-\d+", re.IGNORECASE)


# Extract the rack number from a hostname like rpc-91-5.
def rack_id(hostname: str) -> Optional[int]:
    if not hostname:
        return None
    m = RACK_HOSTNAME_PATTERN.match(str(hostname).strip())
    return int(m.group(1)) if m else None


# Bidirectional mapping between racks, nodes, and PDU units.
@dataclass
class RackMap:
    rack_to_nodes: Dict[int, List[str]] = field(default_factory=dict)
    rack_to_pdus: Dict[int, List[str]] = field(default_factory=dict)
    node_to_rack: Dict[str, int] = field(default_factory=dict)
    pdu_to_rack: Dict[str, int] = field(default_factory=dict)


# Scan master/feature parquet dirs to map racks to their nodes and PDUs.
def build_rack_map_from_paths(master_or_feat_dir: Path) -> RackMap:
    rack_map = RackMap()

    for comp_dir in master_or_feat_dir.iterdir():
        if not comp_dir.is_dir() or comp_dir.name == "infra":
            continue
        for p in comp_dir.glob("*.parquet"):
            r = rack_id(p.stem)
            if r is not None:
                rack_map.node_to_rack[p.stem] = r
                rack_map.rack_to_nodes.setdefault(r, []).append(p.stem)

    pdu_dir = master_or_feat_dir / "infra" / "pdu"
    if pdu_dir.exists():
        for p in pdu_dir.glob("*.parquet"):
            r = rack_id(p.stem)
            if r is not None:
                rack_map.pdu_to_rack[p.stem] = r
                rack_map.rack_to_pdus.setdefault(r, []).append(p.stem)

    return rack_map
