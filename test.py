import torch
import triton
from distance import chamfer_distance, bundle_distance

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N', ],  # Argument names to use as an x-axis for the plot
        x_vals=[512 * i for i in range(30, 45)],  # Different possible values for `x_name`
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot
        # Possible values for `line_arg`
        line_vals=['triton'],
        # Label name for the lines
        line_names=[ "Triton/(ms)"],
        # Line styles
        styles=[('blue', '-')],
        ylabel="TIME",  # Label name for the y-axis
        plot_name="chamfer-performance",  # Name for the plot, used also as a file name for saving the plot.
        args={},
    ))
def benchmark(N, provider):
    B = 4
    quantiles = [0.5, 0.2, 0.8]
    x = (torch.randn(B, N, 3)).to('cuda:2').requires_grad_(True)
    y = (torch.randn(B, N, 3)).to('cuda:2').requires_grad_(True)
    ms, min_ms, max_ms=triton.testing.do_bench(lambda: chamfer_distance(x, y, squared=True), quantiles=quantiles)
    perf = lambda ms: ms
    return perf(ms), perf(min_ms), perf(max_ms)
# benchmark.run(print_data=True)

def unit_test():
    N = 10000
    M = 10000
    P = 16
    B = 4
    device = torch.device("cuda:2")
    # x = (torch.randn(B, N, 3)).to(device).requires_grad_(True)
    # y = (torch.randn(B, M, 3)).to(device).requires_grad_(True)
    # dist = chamfer_distance(x, y, squared=True)
    x = (torch.randn(B, N, P, 3)).to(device).requires_grad_(True)
    y = (torch.randn(B, M, P, 3)).to(device).requires_grad_(True)
    dist = bundle_distance(x, y, squared=True)
    dist.sum().backward()
    print(x.grad.shape,y.grad.shape)
unit_test()