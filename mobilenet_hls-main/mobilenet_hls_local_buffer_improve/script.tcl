
# ============================================================
# EECS 298 - Deep Learning Accelerator Design
# Final Project: MobileNetV1 HLS Optimization
#
# Version: Local Buffer Optimization
# Target: Kria KV260 (K26 SOM)
# Tool: Vitis HLS 2023.1
# ============================================================

# ------------------------------------------------------------
# 1. Create HLS Project
# ------------------------------------------------------------

open_project -reset mobilenet_hls_local_buffer

# Top-level function
set_top mobilenet_top

# ------------------------------------------------------------
# 2. Add Design Source Files
# ------------------------------------------------------------

add_files mobilenet_kernels.cpp
add_files mobilenet_top.cpp

# ------------------------------------------------------------
# 3. Add Testbench Files
# ------------------------------------------------------------

add_files -tb tb_mobilenet.cpp
add_files -tb mobilenet_ref.cpp
add_files -tb mobilenet_ref.h

# ------------------------------------------------------------
# 4. Create Solution
# ------------------------------------------------------------

open_solution -reset "local_buffer_100MHz" -flow_target vivado

# Target FPGA: Kria K26
set_part {xck26-sfvc784-2LV-c}

# Target Clock: 100 MHz (10 ns)
create_clock -period 10 -name default

# Diagnostic setting for Vitis HLS 2023.1
config_compile -pipeline_loops 1000000

# ------------------------------------------------------------
# 5. Run C Simulation
# ------------------------------------------------------------

csim_design

# ------------------------------------------------------------
# 6. Run C Synthesis
# ------------------------------------------------------------

csynth_design

# ------------------------------------------------------------
# 7. Optional RTL Co-simulation
# ------------------------------------------------------------

# Uncomment to run RTL Co-simulation.
# Note: Full MobileNet simulation may take a long time.

# cosim_design

# ------------------------------------------------------------
# 8. Optional IP Export
# ------------------------------------------------------------

# Uncomment when preparing for Vivado integration.

# export_design -format ip_catalog

# ------------------------------------------------------------
# 9. Exit Vitis HLS
# ------------------------------------------------------------

exit
