def decision(conf):
    if conf >= 0.85:
        return "High Confidence"
    elif conf >= 0.65:
        return "Needs Review"
    else:
        return "Uncertain"