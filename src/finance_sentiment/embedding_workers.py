"""Persistent GPU workers; only the coordinator reads SQLite or writes vector files."""
from __future__ import annotations

import multiprocessing
from multiprocessing.connection import wait
import os
import signal
import traceback

from .score import PaperBertEncoder
from .tracking import log


def encoder_worker(pipe, config, device, encoder_factory):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        if encoder_factory is PaperBertEncoder:
            import torch
            torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "2"))))
        encoder = encoder_factory(config, device)
        pipe.send(("ready",))
        while True:
            task = pipe.recv()
            if task is None:
                return
            start, texts = task
            pipe.send(("result", start, encoder.encode(texts)))
    except Exception:
        pipe.send(("error", f"{device} worker failed:\n{traceback.format_exc()}"))
    finally:
        pipe.close()


def batch_texts(connection, start, end):
    texts = [row[0] for row in connection.execute(
        "SELECT text FROM vector_rows WHERE row_id>=? AND row_id<? ORDER BY row_id", (start, end))]
    if len(texts) != end-start:
        raise ValueError("Vector row map is not contiguous")
    return texts


def encoded_batches(connection, ranges, config, devices, encoder_factory):
    """Yield (start, end, vectors, device) in completion order, with bounded in-flight work."""
    if len(devices) == 1:
        encoder = encoder_factory(config, devices[0])
        for start, end in ranges:
            yield start, end, encoder.encode(batch_texts(connection, start, end)), devices[0]
        return
    context = multiprocessing.get_context("spawn")  # CUDA must not use fork.
    active = []
    jobs = iter(ranges)

    def dispatch(worker):
        task = next(jobs, None)
        worker["task"] = task
        if task is None:
            worker["pipe"].send(None)
            worker["process"].join()
            worker["process"].close()
            worker["pipe"].close()
            active.remove(worker)
        else:
            start, end = task
            worker["pipe"].send((start, batch_texts(connection, start, end)))

    try:
        for device in devices:
            parent, child = context.Pipe()
            process = context.Process(target=encoder_worker, args=(child, config, device, encoder_factory))
            try:
                process.start()
            except BaseException:
                parent.close()
                if process.pid is not None:
                    process.terminate()
                    process.join()
                raise
            finally:
                child.close()
            active.append(dict(device=device, process=process, pipe=parent, task=None))
        while active:
            ready = wait([w["pipe"] for w in active] + [w["process"].sentinel for w in active], timeout=30)
            if not ready:
                log("2 embedding: waiting for GPU workers (model loading or current batch)")
            for worker in list(active):
                pipe = worker["pipe"]
                if pipe.poll():
                    try:
                        message = pipe.recv()
                    except EOFError as error:
                        raise RuntimeError(f"{worker['device']} worker exited without a result; rerun to resume") from error
                    if message[0] == "error":
                        raise RuntimeError(message[1])
                    if message[0] == "ready":
                        log(f"2 embedding: {worker['device']} model ready")
                    elif message[0] == "result":
                        if worker["task"] is None or worker["task"][0] != message[1]:
                            raise RuntimeError("GPU worker returned an unexpected batch")
                        start, end = worker["task"]
                        # The caller durably writes this result before dispatching another batch here.
                        yield start, end, message[2], worker["device"]
                    else:
                        raise RuntimeError(f"Unknown GPU worker message: {message[0]}")
                    dispatch(worker)
                elif worker["process"].exitcode is not None:
                    raise RuntimeError(f"{worker['device']} worker exited with code {worker['process'].exitcode}; rerun to resume")
    finally:
        # Stop/join children before releasing the coordinator's run lock, including on Ctrl-C.
        for worker in active:
            process = worker["process"]
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()
            process.close()
            worker["pipe"].close()
