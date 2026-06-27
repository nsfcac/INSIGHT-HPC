from __future__ import annotations


# Per-node OPEN/UPDATE/CLOSE episode state machine over fused votes.
class EpisodeStateMachine:
    def update(self, hostname, ts, fused_score, votes_5min, evidence):
        raise NotImplementedError("episode lifecycle")
