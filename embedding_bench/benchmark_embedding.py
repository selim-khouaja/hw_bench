#!/usr/bin/env python3
"""
Async benchmark client for vLLM embedding server.

Measures throughput and latency across combinations of batch size and concurrency.
"""

import argparse
import asyncio
import json
import time
import random
import string
from pathlib import Path

import atexit
import threading
from typing import Optional

import aiohttp

try:
    import pynvml as _pynvml
    _pynvml.nvmlInit()
    atexit.register(_pynvml.nvmlShutdown)
    _NVML_AVAILABLE: bool = True
except Exception:
    _pynvml = None  # type: ignore[assignment]
    _NVML_AVAILABLE = False


class PowerMonitor:
    def __init__(self, poll_interval: float = 0.1) -> None:
        self._poll_interval = poll_interval
        self._stop_event = threading.Event()
        self._samples: list[float] = []
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if not _NVML_AVAILABLE:
            return
        self._stop_event.clear()
        self._samples = []
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def _poll_loop(self) -> None:
        try:
            num_gpus = _pynvml.nvmlDeviceGetCount()
            handles = [_pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(num_gpus)]
        except Exception:
            return
        while not self._stop_event.wait(timeout=self._poll_interval):
            total_mw: float = 0.0
            for handle in handles:
                try:
                    total_mw += _pynvml.nvmlDeviceGetPowerUsage(handle)
                except Exception:
                    pass  # skip GPUs without power sensor
            self._samples.append(total_mw / 1000.0)  # mW -> W

    def stop(self) -> Optional[dict]:
        if not _NVML_AVAILABLE or self._thread is None:
            return None
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        self._thread = None
        if not self._samples:
            return None
        return {"power_avg_w": sum(self._samples) / len(self._samples)}


def generate_text(num_tokens: int) -> str:
    """Generate synthetic text of approximately `num_tokens` tokens.
    Uses simple word-like tokens (~5 chars each) to approximate tokenizer output.
    """
    words_needed = max(1, num_tokens)
    words = []
    for _ in range(words_needed):
        length = random.randint(3, 8)
        word = "".join(random.choices(string.ascii_lowercase, k=length))
        words.append(word)
    return " ".join(words)


async def post_embeddings(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    texts: list[str],
) -> float:
    """Post a single embedding request and return wall-clock latency in ms."""
    url = f"{base_url}/v1/embeddings"
    payload = {"model": model, "input": texts}
    t0 = time.perf_counter()
    async with session.post(url, json=payload) as resp:
        resp.raise_for_status()
        await resp.json()
    return (time.perf_counter() - t0) * 1000.0


async def run_sweep_point(
    base_url: str,
    model: str,
    chunk_size: int,
    batch_size: int,
    concurrency: int,
    num_requests: int,
) -> dict:
    """Run one (batch_size, concurrency) sweep point and return metrics."""
    # Pre-generate all text batches
    batches = [
        [generate_text(chunk_size) for _ in range(batch_size)]
        for _ in range(num_requests)
    ]

    latencies_ms: list[float] = []
    sem = asyncio.Semaphore(concurrency)

    async def bounded_request(texts: list[str], session: aiohttp.ClientSession) -> None:
        async with sem:
            lat = await post_embeddings(session, base_url, model, texts)
            latencies_ms.append(lat)

    monitor = PowerMonitor(poll_interval=0.1)
    connector = aiohttp.TCPConnector(limit=concurrency + 4)
    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        monitor.start()
        t_start = time.perf_counter()
        try:
            await asyncio.gather(*[bounded_request(b, session) for b in batches])
        finally:
            elapsed = time.perf_counter() - t_start
            power_stats = monitor.stop()

    completed = len(latencies_ms)
    latencies_ms.sort()
    p50 = latencies_ms[int(completed * 0.50) - 1] if completed else 0.0
    p99 = latencies_ms[int(completed * 0.99) - 1] if completed else 0.0

    total_embeddings = completed * batch_size
    throughput = total_embeddings / elapsed if elapsed > 0 else 0.0
    throughput_per_user = throughput / concurrency if concurrency > 0 else 0.0

    power_avg_w: Optional[float] = None
    energy_joules: Optional[float] = None
    emb_per_joule: Optional[float] = None

    if power_stats is not None:
        power_avg_w = round(power_stats["power_avg_w"], 2)
        energy_joules = round(power_stats["power_avg_w"] * elapsed, 2)
        if energy_joules and energy_joules > 0:
            emb_per_joule = round(total_embeddings / energy_joules, 4)

    return {
        "model": model,
        "chunk_size": chunk_size,
        "batch_size": batch_size,
        "concurrency": concurrency,
        "num_requests": num_requests,
        "completed_requests": completed,
        "elapsed_sec": round(elapsed, 3),
        "p50_latency_ms": round(p50, 2),
        "p99_latency_ms": round(p99, 2),
        "throughput_emb_per_sec": round(throughput, 2),
        "throughput_per_user": round(throughput_per_user, 2),
        "power_avg_w": power_avg_w,
        "energy_joules": energy_joules,
        "emb_per_joule": emb_per_joule,
    }


def parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",")]


async def main() -> None:
    parser = argparse.ArgumentParser(description="Embedding benchmark client")
    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument("--base-url", default="http://localhost:8000", help="vLLM base URL")
    parser.add_argument("--chunk-size", type=int, required=True, help="Approximate tokens per text")
    parser.add_argument("--batch-sizes", default="1,4,16,64,256", help="Comma-separated batch sizes")
    parser.add_argument("--concurrencies", default="1,4,16,64", help="Comma-separated concurrencies")
    parser.add_argument("--num-requests", type=int, default=200, help="Requests per sweep point")
    parser.add_argument("--result-dir", default="../results", help="Output directory")
    args = parser.parse_args()

    batch_sizes = parse_int_list(args.batch_sizes)
    concurrencies = parse_int_list(args.concurrencies)
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    random.seed(42)

    for batch_size in batch_sizes:
        for concurrency in concurrencies:
            print(
                f"  chunk={args.chunk_size} batch={batch_size} concurrency={concurrency} ...",
                flush=True,
            )
            result = await run_sweep_point(
                base_url=args.base_url,
                model=args.model,
                chunk_size=args.chunk_size,
                batch_size=batch_size,
                concurrency=concurrency,
                num_requests=args.num_requests,
            )
            model_slug = args.model.replace("/", "_")
            fname = f"{model_slug}__chunk{args.chunk_size}__bs{batch_size}__conc{concurrency}.json"
            out_path = result_dir / fname
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            print(
                f"    -> p50={result['p50_latency_ms']}ms  p99={result['p99_latency_ms']}ms  "
                f"tput={result['throughput_emb_per_sec']} emb/s  saved {out_path.name}",
                flush=True,
            )


if __name__ == "__main__":
    asyncio.run(main())
