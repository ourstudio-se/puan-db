import time
from contextlib import contextmanager
from pldag import PLDAG
from hashlib import sha1
from itertools import starmap

@contextmanager
def timer(name: str = None):
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Computation for {name if name else 'unknown function'} took {elapsed_time:.8f} seconds.")
