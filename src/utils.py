import random
import numpy as np
import torch
import logging
from pathlib import Path
import logging
from pathlib import Path


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logger(
    name: str,
    log_file: str = None,
    level: int = logging.INFO,
    console: bool = True
) -> logging.Logger:
    if log_file:
        log_path = Path(log_file).parent
        log_path.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    handlers = []
    
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)
    
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    logging.basicConfig(
        level=level,
        handlers=handlers,
        force=True
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    return logger
