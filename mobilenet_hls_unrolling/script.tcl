# Independent loop-unrolling experiment based on the original baseline.
# Keep the same part, target clock and automatic-pipelining setting.

open_project -reset mobilenet_hls_unrolling
set_top mobilenet_top

add_files mobilenet_kernels.cpp
add_files mobilenet_top.cpp
add_files mobilenet.h

add_files -tb tb_mobilenet.cpp
add_files -tb mobilenet_ref.cpp
add_files -tb mobilenet_ref.h

open_solution -reset unrolling_100MHz -flow_target vivado
set_part {xck26-sfvc784-2LV-c}
create_clock -period 10 -name default

config_compile -pipeline_loops 0

csim_design
csynth_design
exit
