import threading
from datetime import datetime


def info(msg: str):
    print(f'INFO {datetime.now().isoformat()} - {threading.current_thread().getName()[:20]: <20} - {msg}')


def warning(msg: str):
    print(f'WARN {datetime.now().isoformat()} - {threading.current_thread().getName()[:20]: <20} - {msg}')


def error(msg: str):
    print(f'ERR  {datetime.now().isoformat()} - {threading.current_thread().getName()[:20]: <20} - {msg}')
