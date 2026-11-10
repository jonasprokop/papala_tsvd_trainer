"""Shared defaults for the TVSD preprocessing handoff."""

# Full original TVSD V1 inventory: 45 recording sessions (17 F, 28 N).
# Override --expected-sessions for a deliberately partial dataset.
EXPECTED_SESSIONS = 45
DEFAULT_RESPONSE_WINDOW_MS = (40.0, 160.0)
