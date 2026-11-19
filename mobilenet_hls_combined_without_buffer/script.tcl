# Combined WITHOUT BUFFER: DWConv 8x8 tiling + KJ unroll + KI pipeline.
# Independent from all buffered variants; standardized Kria K26 at 100 MHz.

open_project -reset mobilenet_hls_combined_without_buffer
set_top mobilenet_top

add_files mobilenet_kernels.cpp
add_files mobilenet_top.cpp
add_files mobilenet.h

add_files -tb tb_mobilenet.cpp
add_files -tb mobilenet_ref.cpp
add_files -tb mobilenet_ref.h

open_solution -reset combined_without_buffer_100MHz -flow_target vivado
set_part {xck26-sfvc784-2LV-c}
create_clock -period 10 -name default

# Only explicit source pragmas can request pipeline or unrolling.
config_compile -pipeline_loops 0

csim_design
csynth_design
exit
