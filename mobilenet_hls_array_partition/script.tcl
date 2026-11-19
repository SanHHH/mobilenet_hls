# Array Partitioning experiment: original baseline with partitioned DW weights
# Device, clock, and automatic pipelining policy match the standardized baseline.

open_project -reset mobilenet_hls_array_partition
set_top mobilenet_top

add_files mobilenet_kernels.cpp
add_files mobilenet_top.cpp
add_files mobilenet.h

add_files -tb tb_mobilenet.cpp
add_files -tb mobilenet_ref.cpp
add_files -tb mobilenet_ref.h

open_solution -reset array_partition_100MHz -flow_target vivado
set_part {xck26-sfvc784-2LV-c}
create_clock -period 10 -name default

# Keep the same setting as the original standardized baseline.
config_compile -pipeline_loops 0

csim_design
csynth_design
exit
