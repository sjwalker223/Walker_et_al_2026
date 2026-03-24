def date_range(start_date, end_date, increment, period):
    from dateutil.relativedelta import relativedelta
    result = []
    nxt = start_date
    delta = relativedelta(**{period:increment})
    while nxt <= end_date:
        result.append(nxt)
        nxt += delta
    return result