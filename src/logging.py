import logging
def get_logger():
    log = logging.getLogger('qtree')
    log.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s- %(levelname)s•\t%(message)s',
        datefmt = '%H:%M:%S'
    )
    ch.setFormatter(formatter)
    log.addHandler(ch)

    log.info("foo")
    return log
