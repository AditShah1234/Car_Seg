import logging

format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"


# set up logging to file
def file_logging(f_name, level):
    logging.basicConfig(filename=f_name, level=level, format=format)

    # set up logging to console
    console = logging.StreamHandler()
    console.setLevel(level)
    formatter = logging.Formatter(format)
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)
    logger = logging.getLogger("logging_example")
