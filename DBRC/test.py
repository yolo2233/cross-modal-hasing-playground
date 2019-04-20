import time
from functools import wraps

def count_time(result_file=None):
    def wrapper(func):
        def return_wrapper(*args, **kwargs):
            with open(result_file, 'a', encoding='utf-8') as f:
                f.write('excute ' + func.__name__ + '...\n')
            tic = time.clock()
            results = func(*args, **kwargs)
            toc = time.clock()
            with open('results/result_flickr.txt', 'a', encoding='utf-8') as f:
                f.write(func.__name__ + ' cost time: ' + str(toc - tic) + '\n\n')

            return results
        return return_wrapper
    return wrapper



