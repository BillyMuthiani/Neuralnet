# Benchmarks

Performance benchmarks for Kronyx layers.

## Dense Layer

| Input Shape | Output Shape | Time (ms) |
|-------------|--------------|-----------|
| (1000, 100) | (1000, 50) | ~2.5 |
| (1000, 500) | (1000, 200) | ~15 |

## Conv2D Layer

| Input Shape | Filters | Kernel | Time (ms) |
|-------------|---------|--------|-----------|
| (32, 28, 28, 1) | 16 | 3 | ~150 |

## Methodology

- Intel i7-12700H CPU
- NumPy 2.3.4
- Python 3.11
- Warm cache, single run