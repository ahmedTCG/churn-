# src/features/strong_events.py

# List 0 (current)
STRONG_SIGNAL_EVENTS = [
    "emarsys_sessions_content_category",
    "emarsys_sessions_content_url",
    "emarsys_sessions_content_tag",
    "emarsys_sessions_cart_update",
    "emarsys_open",
    "emarsys_sessions_purchase",
    "emarsys_webchannel_click",
    "emarsys_sessions_view",
]

# List 1 (new)
STRONG_SIGNAL_EVENTS_1 = [
    "emarsys_unsub",
    "emarsys_hard_bounce",
    "emarsys_block_bounce",
    "emarsys_cancel",
    "emarsys_open",
    "emarsys_click",
    "emarsys_sessions_view",
    "order",
]

# 🔴 CHANGE ONLY THIS LINE TO SWITCH
ACTIVE_STRONG_SIGNAL_EVENTS = STRONG_SIGNAL_EVENTS_1
# ACTIVE_STRONG_SIGNAL_EVENTS = STRONG_SIGNAL_EVENTS_1
