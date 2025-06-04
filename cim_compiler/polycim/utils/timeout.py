import functools
import multiprocessing


def timeout(seconds):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            def target(queue, *args, **kwargs):
                try:
                    result = func(*args, **kwargs)
                    queue.put(result)
                except Exception as e:
                    queue.put(e)  # Put the exception in the queue

            queue = multiprocessing.Queue()
            process = multiprocessing.Process(
                target=target, args=(queue, *args), kwargs=kwargs
            )
            process.start()
            process.join(timeout=seconds)

            if process.is_alive():
                process.terminate()
                process.join()
                return None
            else:
                return queue.get() if not queue.empty() else None

        return wrapper

    return decorator
