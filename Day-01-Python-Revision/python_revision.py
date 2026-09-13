"""
Day 01 — Python Revision
Practice: functions, *args/**kwargs, comprehensions, generators, decorators.
"""

from functools import wraps


def summarize_numbers(*numbers):
    """Return basic statistics for any number of positional arguments."""
    return {
        "count": len(numbers),
        "sum": sum(numbers),
        "min": min(numbers) if numbers else None,
        "max": max(numbers) if numbers else None,
    }


def build_config(**kwargs):
    """Accept flexible keyword configuration."""
    return kwargs


squares = [x * x for x in range(1, 6)]
even_numbers = [x for x in range(10) if x % 2 == 0]
number_map = {x: x * x for x in range(1, 6)}
unique_values = {x % 3 for x in range(10)}


def generate_batches(items, batch_size):
    """Yield batches lazily instead of creating all batches at once."""
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def log_call(func):
    """Simple decorator for observing function calls."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__}...")
        result = func(*args, **kwargs)
        print(f"{func.__name__} finished.")
        return result

    return wrapper


@log_call
def multiply(a, b):
    return a * b


if __name__ == "__main__":
    print(summarize_numbers(10, 20, 30))
    print(build_config(model="demo", temperature=0.2))
    print(squares, even_numbers, number_map, unique_values)
    print(list(generate_batches(list(range(10)), 3)))
    print(multiply(6, 7))
