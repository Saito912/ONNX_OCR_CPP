import numpy as np
import timeit
import ctc_decoder
from ctc_decode_py import ctc_beam_search_decode_py


def run_speed_test():
    print("--- CTC Beam Search Decoder Speed Test ---")

    B, T, C = 4, 12, 63
    beam_size = 10

    print(f"\nTest configuration:")
    print(f"  Batch size (B): {B}")
    print(f"  Time steps (T): {T}")
    print(f"  Num classes (C): {C}")
    print(f"  Beam size: {beam_size}")

    np.random.seed(42)
    logits = np.random.randn(B, T, C).astype(np.float32)

    flat_logits = logits.flatten().tolist()
    shape = list(logits.shape)

    print("\nVerifying results are identical...")

    cpp_results = ctc_decoder.ctc_beam_search(
        output_tensor=flat_logits,
        shape=shape,
        beam_size=beam_size
    )

    py_results = ctc_beam_search_decode_py(
        logits=logits,
        beam_size=beam_size
    )

    if cpp_results == py_results:
        print("SUCCESS: C++ and Python implementations produced identical results.")
    else:
        print("ERROR: Results are different!")
        print(f"  C++: {cpp_results}")
        print(f"  PY:  {py_results}")
        return

    num_runs = 200
    print(f"\nRunning performance test ({num_runs} iterations)...")

    # 测试 C++ 版本
    cpp_timer = timeit.Timer(
        lambda: ctc_decoder.ctc_beam_search(flat_logits, shape, beam_size)
    )
    cpp_time = cpp_timer.timeit(number=num_runs)

    # 测试 Python 版本
    py_timer = timeit.Timer(
        lambda: ctc_beam_search_decode_py(logits, beam_size)
    )
    py_time = py_timer.timeit(number=num_runs)

    print("\n--- Performance Results ---")
    print(f"C++ (pybind11) version total time: {cpp_time:.6f} seconds")
    print(f"Pure Python (numpy) version total time: {py_time:.6f} seconds")

    if cpp_time > 0:
        speedup = py_time / cpp_time
        print(f"\nC++ version is approximately {speedup:.2f}x faster than the Python version.")
    else:
        print("\nC++ version was too fast to measure accurately.")


if __name__ == "__main__":
    run_speed_test()