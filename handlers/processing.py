import multiprocessing as mp
# Only set it once?
start_method = mp.get_start_method(allow_none=True)
if start_method is None:
    mp.set_start_method("spawn")
