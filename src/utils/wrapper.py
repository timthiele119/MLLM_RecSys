import time


def tryExcept(func):
    """A decorator to wrap a function with try-except for error handling."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Error in function '{func.__name__}': {str(e)}")
    return wrapper


def timeMeasured(func):
    def wrapper(*args, **kwargs):
        startTime = time.time()
        result = func(*args, **kwargs)
        elapsedTime = time.time() - startTime

        hours = int(elapsedTime // 3600)
        minutes = int((elapsedTime % 3600) // 60)
        seconds = int(elapsedTime % 60)

        print(f"{func.__name__} executed in {hours} hours, {minutes} minutes, {seconds} seconds.\n")
        return result
    return wrapper