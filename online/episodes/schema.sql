-- insight_hpc DDL. Separate database; never references MonSTer tables.
CREATE TABLE IF NOT EXISTS episodes (
  episode_id       BIGSERIAL PRIMARY KEY,
  hostname         TEXT NOT NULL,
  component        TEXT NOT NULL,
  status           TEXT NOT NULL,
  opened_at        TIMESTAMPTZ NOT NULL,
  last_seen_at     TIMESTAMPTZ NOT NULL,
  closed_at        TIMESTAMPTZ,
  anomaly_type     TEXT,
  max_fused_score  DOUBLE PRECISION,
  peak_votes       INT,
  primary_job_id   BIGINT,
  minutes_into_job INT,
  explanation      JSONB
);

CREATE TABLE IF NOT EXISTS episode_evidence (
  episode_id  BIGINT REFERENCES episodes(episode_id),
  ts          TIMESTAMPTZ NOT NULL,
  fused_score DOUBLE PRECISION,
  votes_5min  INT,
  evidence    JSONB
);
