# Independent DWConv loop-pipelining experiment derived from Original Baseline.
# Same device, clock, and global compile setting as standardized baseline.

open_project -reset mobilenet_hls_pipelining
set_top mobilenet_top

add_files mobilenet_kernels.cpp
add_files mobilenet_top.cpp
add_files mobilenet.h

add_files -tb tb_mobilenet.cpp
add_files -tb mobilenet_ref.cpp
add_files -tb mobilenet_ref.h

open_solution -reset pipelining_100MHz -flow_target vivado
set_part {xck26-sfvc784-2LV-c}
create_clock -period 10 -name default

# Disable automatic global pipelining; explicit source pragma remains active.
config_compile -pipeline_loops 0

csim_design
csynth_design
exit
