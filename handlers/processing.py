import multiprocessing as mp
# Only set it once?
start_method = mp.get_start_method(allow_none=True)
if start_method is None:
    print("Setting start method to spawn")
    mp.set_start_method("spawn")
else:
    print(f"Start method already set to {start_method}")
