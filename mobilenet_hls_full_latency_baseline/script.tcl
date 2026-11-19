# FULL MobileNetV1 (alpha=0.5, 160x160) Original Baseline RTL latency.
# WARNING: full-network C/RTL co-simulation can take very long to complete.
# Keep the device, clock and compile setting equal across ALL variants.

open_project -reset mobilenet_hls_full_latency_baseline
set_top mobilenet_top

add_files mobilenet_kernels.cpp
add_files mobilenet_top.cpp
add_files mobilenet.h
add_files -tb mobilenet_ref.cpp
add_files -tb mobilenet_ref.h
add_files -tb tb_full_latency.cpp

open_solution -reset full_latency_100MHz -flow_target vivado
set_part {xck26-sfvc784-2LV-c}
create_clock -period 10 -name default
config_compile -pipeline_loops 0

csim_design
csynth_design
cosim_design -rtl verilog
exit
