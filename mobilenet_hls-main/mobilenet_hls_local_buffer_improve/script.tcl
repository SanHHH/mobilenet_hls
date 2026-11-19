# ============================================================
# MobileNetV1 HLS - Standardized Synthesis Configuration
# ============================================================
#
# Project:
#   MobileNetV1 Hardware Accelerator
#
# Optimization Stage:
#   Local Buffer Improvement
#
# Description:
#   This script establishes a standardized HLS synthesis
#   environment for comparing different optimization methods.
#
#   All experimental versions should use identical:
#     1. Vitis HLS version
#     2. Target FPGA device
#     3. Target clock period
#     4. Synthesis flow
#     5. Global HLS compilation settings
#
#   Design-specific optimizations, including local buffering,
#   loop unrolling, pipelining, and array partitioning, are
#   implemented separately in the C++ source code.
#
# Tool:
#   AMD Vitis HLS 2023.1
#
# Target Device:
#   AMD Kria K26
#   xck26-sfvc784-2LV-c
#
# Target Frequency:
#   100 MHz (10 ns clock period)
#
# ============================================================


# ============================================================
# 1. Project Configuration
# ============================================================
#
# Description:
#   Create or reset the HLS project.
#
#   The project name is unique to this optimization stage.
#   Resetting the project ensures that previous synthesis
#   results do not affect the current experiment.
#
# ============================================================

open_project -reset mobilenet_hls_local_buffer_improve


# ============================================================
# 2. Top-Level Function
# ============================================================
#
# Description:
#   Define mobilenet_top as the top-level synthesis function.
#
#   All hardware modules are synthesized through this
#   function.
#
# ============================================================

set_top mobilenet_top


# ============================================================
# 3. Design Source Files
# ============================================================
#
# Description:
#   Add the C++ implementation files required for synthesis.
#
#   mobilenet_top.cpp:
#     Top-level MobileNet accelerator architecture.
#
#   mobilenet_kernels.cpp:
#     Convolution, depthwise convolution, pointwise convolution,
#     pooling, and fully connected kernel implementations.
#
#   mobilenet.h:
#     Data types, function declarations, and model definitions.
#
# ============================================================

add_files mobilenet_top.cpp
add_files mobilenet_kernels.cpp
add_files mobilenet.h


# ============================================================
# 4. Testbench Files
# ============================================================
#
# Description:
#   Add testbench and reference implementation files.
#
#   These files are used for functional verification and
#   are not synthesized into hardware.
#
#   tb_mobilenet.cpp:
#     Kernel-level and full-network verification.
#
#   mobilenet_ref.cpp:
#     Software reference implementation.
#
#   mobilenet_ref.h:
#     Reference function declarations.
#
# ============================================================

add_files -tb tb_mobilenet.cpp
add_files -tb mobilenet_ref.cpp
add_files -tb mobilenet_ref.h


# ============================================================
# 5. Solution Configuration
# ============================================================
#
# Description:
#   Create a synthesis solution using the Vivado IP flow.
#
#   The same solution name should be used across all
#   experimental versions.
#
# ============================================================

open_solution -reset "solution_100MHz" -flow_target vivado


# ============================================================
# 6. Target FPGA
# ============================================================
#
# Description:
#   Select the AMD Kria K26 FPGA device.
#
#   All experiments must use the same target part to ensure
#   consistent hardware resource comparisons.
#
# ============================================================

set_part {xck26-sfvc784-2LV-c}


# ============================================================
# 7. Clock Configuration
# ============================================================
#
# Description:
#   Set the target clock period to 10 ns (100 MHz).
#
#   The clock constraint is identical for all experiments.
#
# ============================================================

create_clock -period 10 -name default


# ============================================================
# 8. Standardized HLS Compilation Settings
# ============================================================
#
# Description:
#   Disable automatic loop pipelining.
#
#   This allows the effects of explicitly implemented
#   optimization directives to be evaluated more clearly.
#
#   Explicit PIPELINE directives in the source code
#   remain effective.
#
#   This setting must be identical across all experiments.
#
# ============================================================

config_compile -pipeline_loops 0


# ============================================================
# 9. C Simulation
# ============================================================
#
# Description:
#   Perform functional verification before synthesis.
#
#   The optimized C++ implementation is compared against
#   the reference model.
#
#   Expected result:
#     ALL TESTS PASSED
#     0 mismatching values
#
# ============================================================

csim_design


# ============================================================
# 10. C Synthesis
# ============================================================
#
# Description:
#   Convert the C++ design into RTL hardware.
#
#   Collect the following performance metrics:
#
#     1. Estimated clock period
#     2. Estimated latency
#     3. DSP utilization
#     4. LUT utilization
#     5. Flip-flop utilization
#     6. BRAM utilization
#     7. Pipeline initiation interval
#
#   The synthesis report is used to compare the baseline
#   and optimized implementations.
#
# ============================================================

csynth_design


# ============================================================
# 11. End of Script
# ============================================================
#
# Description:
#   Exit Vitis HLS after completing simulation and synthesis.
#
# ============================================================

exit