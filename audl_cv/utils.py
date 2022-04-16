def mmss_to_s(mmss: str) -> int:
    """
    Converts time str in form "mm:ss" to integer of seconds
    """
    l = mmss.split(":")
    return int(l[0]) * 60 + int(l[1])
