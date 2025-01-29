import multiprocessing as mp
import logging

logging.getLogger("audio_separator").setLevel(logging.ERROR)
logging.getLogger("fairseq").setLevel(logging.ERROR)
logging.getLogger("speechbrain").setLevel(logging.ERROR)


# Only set it once?
start_method = mp.get_start_method(allow_none=True)
if start_method is None:
    mp.set_start_method("spawn")
