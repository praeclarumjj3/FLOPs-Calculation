from memory_profiler import memory_usage
from FLOPs import FLOP

def calculate_peak_memory_for_FLOP_calculation():
  flops_calc = FLOP(input_shape=(256, 3 ,300 ,300), filter_shape=(64 ,3 ,3 ,3), stride=1, padding=1, dilation_rate=0)
  flops_calc.calculateFLOPs()

if __name__ == "__main__":
    
    mem = max(memory_usage(proc=calculate_peak_memory_for_FLOP_calculation))
    print('-----------------------------------------------------------------')
    print("\nPeak memory used: {} MiB".format(mem))