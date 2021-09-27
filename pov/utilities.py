def format_time(seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    periods = [("hours", hours), ("minutes", minutes), ("seconds", seconds)]
    time_string = ", ".join(
        "{} {}".format(value, name) for name, value in periods if value
    )

    return time_string
