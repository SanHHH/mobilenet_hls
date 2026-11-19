# Combined WITH BUFFER: 8x8 DWConv tiling + on-chip input/weight buffers
# + banked local input + KJ unroll factor 3 + KI pipeline II=1 request.

open_project -reset mobilenet_hls_combined_with_buffer
set_top mobilenet_top

add_files mobilenet_kernels.cpp
add_files mobilenet_top.cpp
add_files mobilenet.h

add_files -tb tb_mobilenet.cpp
add_files -tb mobilenet_ref.cpp
add_files -tb mobilenet_ref.h

open_solution -reset combined_with_buffer_100MHz -flow_target vivado
set_part {xck26-sfvc784-2LV-c}
create_clock -period 10 -name default

# Same global automatic-pipeline policy used for all comparison variants.
config_compile -pipeline_loops 0

csim_design
csynth_design
exit
