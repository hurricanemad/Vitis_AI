
####################################################################################
# Generated by Vivado 2021.2 built on 'Tue Oct 19 02:47:39 MDT 2021' by 'xbuild'
# Command Used: write_xdc -force -exclude_physical ./xilinx_zcu102_base_wrapper_timing_golden.xdc
####################################################################################


####################################################################################
# Constraints from file : 'xilinx_zcu102_base_ps_e_0.xdc'
####################################################################################

current_instance xilinx_zcu102_base_i/ps_e/inst
create_clock -period 10.000 -name clk_pl_0 [get_pins {PS8_i/PLCLK[0]}]

####################################################################################
# Constraints from file : 'xilinx_zcu102_base_axi_intc_0_0.xdc'
####################################################################################

current_instance -quiet
current_instance xilinx_zcu102_base_i/axi_intc_0/U0
set_false_path -to [get_cells -filter IS_SEQUENTIAL {INTC_CORE_I/*ASYNC_GEN.intr_ff*[0]}]

####################################################################################
# Constraints from file : 'xilinx_zcu102_base_proc_sys_reset_2_0.xdc'
####################################################################################

current_instance -quiet
current_instance xilinx_zcu102_base_i/proc_sys_reset_2/U0
set_false_path -to [get_pins -hier *cdc_to*/D]

####################################################################################
# Constraints from file : 'xilinx_zcu102_base_proc_sys_reset_1_0.xdc'
####################################################################################

current_instance -quiet
current_instance xilinx_zcu102_base_i/proc_sys_reset_1/U0
set_false_path -to [get_pins -hier *cdc_to*/D]

####################################################################################
# Constraints from file : 'xilinx_zcu102_base_clk_wiz_0_0.xdc'
####################################################################################

current_instance -quiet
current_instance xilinx_zcu102_base_i/clk_wiz_0/inst
create_clock -period 10.000 [get_ports -scoped_to_current_instance clk_in1]
set_input_jitter [get_clocks -of_objects [get_ports -scoped_to_current_instance clk_in1]] 0.100

####################################################################################
# Constraints from file : 'timing_clocks.xdc'
####################################################################################

current_instance -quiet
current_instance xilinx_zcu102_base_i/DPUCZDX8G_1/inst
set_multicycle_path -setup -start -from [get_clocks -of_objects [get_ports -scoped_to_current_instance ap_clk_2]] -to [get_clocks -of_objects [get_ports -scoped_to_current_instance aclk]] 2
set_multicycle_path -hold -start -from [get_clocks -of_objects [get_ports -scoped_to_current_instance ap_clk_2]] -to [get_clocks -of_objects [get_ports -scoped_to_current_instance aclk]] 1
set_multicycle_path -setup -end -from [get_clocks -of_objects [get_ports -scoped_to_current_instance aclk]] -to [get_clocks -of_objects [get_ports -scoped_to_current_instance ap_clk_2]] 2
set_multicycle_path -hold -end -from [get_clocks -of_objects [get_ports -scoped_to_current_instance aclk]] -to [get_clocks -of_objects [get_ports -scoped_to_current_instance ap_clk_2]] 1

####################################################################################
# Constraints from file : 'timing_clocks.xdc'
####################################################################################

current_instance -quiet
current_instance xilinx_zcu102_base_i/DPUCZDX8G_2/inst
set_multicycle_path -setup -start -from [get_clocks -of_objects [get_ports -scoped_to_current_instance ap_clk_2]] -to [get_clocks -of_objects [get_ports -scoped_to_current_instance aclk]] 2
set_multicycle_path -hold -start -from [get_clocks -of_objects [get_ports -scoped_to_current_instance ap_clk_2]] -to [get_clocks -of_objects [get_ports -scoped_to_current_instance aclk]] 1
set_multicycle_path -setup -end -from [get_clocks -of_objects [get_ports -scoped_to_current_instance aclk]] -to [get_clocks -of_objects [get_ports -scoped_to_current_instance ap_clk_2]] 2
set_multicycle_path -hold -end -from [get_clocks -of_objects [get_ports -scoped_to_current_instance aclk]] -to [get_clocks -of_objects [get_ports -scoped_to_current_instance ap_clk_2]] 1

####################################################################################
# Constraints from file : 'xpm_cdc_async_rst.tcl'
####################################################################################

current_instance -quiet
current_instance xilinx_zcu102_base_i/axi_ic_ps_e_S_AXI_HPC0_FPD/s00_couplers/auto_us_df/inst/gen_upsizer.gen_full_upsizer.axi_upsizer_inst/USE_READ.gen_pktfifo_r_upsizer.pktfifo_read_data_inst/dw_fifogen_ar/inst_fifo_gen/gaxi_full_lite.gread_ch.grach2.axi_rach/grf.rf/rstblk/ngwrdrst.grst.g7serrst.gnsckt_wrst.rst_wr_reg2_inst
set_false_path -through [get_ports -scoped_to_current_instance -no_traverse src_arst]
current_instance -quiet
current_instance xilinx_zcu102_base_i/axi_ic_ps_e_S_AXI_HPC0_FPD/s00_couplers/auto_us_df/inst/gen_upsizer.gen_full_upsizer.axi_upsizer_inst/USE_WRITE.gen_pktfifo_w_upsizer.pktfifo_write_data_inst/dw_fifogen_aw/inst_fifo_gen/gaxi_full_lite.gwrite_ch.gwach2.axi_wach/grf.rf/rstblk/ngwrdrst.grst.g7serrst.gnsckt_wrst.rst_wr_reg2_inst
set_false_path -through [get_ports -scoped_to_current_instance -no_traverse src_arst]
current_instance -quiet
current_instance xilinx_zcu102_base_i/axi_ic_ps_e_S_AXI_HPC0_FPD/s01_couplers/auto_us_df/inst/gen_upsizer.gen_full_upsizer.axi_upsizer_inst/USE_READ.gen_pktfifo_r_upsizer.pktfifo_read_data_inst/dw_fifogen_ar/inst_fifo_gen/gaxi_full_lite.gread_ch.grach2.axi_rach/grf.rf/rstblk/ngwrdrst.grst.g7serrst.gnsckt_wrst.rst_wr_reg2_inst
set_false_path -through [get_ports -scoped_to_current_instance -no_traverse src_arst]
current_instance -quiet
current_instance xilinx_zcu102_base_i/axi_ic_ps_e_S_AXI_HPC0_FPD/s01_couplers/auto_us_df/inst/gen_upsizer.gen_full_upsizer.axi_upsizer_inst/USE_WRITE.gen_pktfifo_w_upsizer.pktfifo_write_data_inst/dw_fifogen_aw/inst_fifo_gen/gaxi_full_lite.gwrite_ch.gwach2.axi_wach/grf.rf/rstblk/ngwrdrst.grst.g7serrst.gnsckt_wrst.rst_wr_reg2_inst
set_false_path -through [get_ports -scoped_to_current_instance -no_traverse src_arst]
current_instance -quiet
current_instance xilinx_zcu102_base_i/axi_ic_ps_e_S_AXI_HP2_FPD/s01_couplers/auto_us_df/inst/gen_upsizer.gen_full_upsizer.axi_upsizer_inst/USE_READ.gen_pktfifo_r_upsizer.pktfifo_read_data_inst/dw_fifogen_ar/inst_fifo_gen/gaxi_full_lite.gread_ch.grach2.axi_rach/grf.rf/rstblk/ngwrdrst.grst.g7serrst.gnsckt_wrst.rst_wr_reg2_inst
set_false_path -through [get_ports -scoped_to_current_instance -no_traverse src_arst]
current_instance -quiet
current_instance xilinx_zcu102_base_i/axi_ic_ps_e_S_AXI_HP2_FPD/s01_couplers/auto_us_df/inst/gen_upsizer.gen_full_upsizer.axi_upsizer_inst/USE_WRITE.gen_pktfifo_w_upsizer.pktfifo_write_data_inst/dw_fifogen_aw/inst_fifo_gen/gaxi_full_lite.gwrite_ch.gwach2.axi_wach/grf.rf/rstblk/ngwrdrst.grst.g7serrst.gnsckt_wrst.rst_wr_reg2_inst
set_false_path -through [get_ports -scoped_to_current_instance -no_traverse src_arst]
current_instance -quiet
current_instance xilinx_zcu102_base_i/axi_ic_ps_e_S_AXI_HP2_FPD/s00_couplers/s00_data_fifo/inst/gen_fifo.fifo_gen_inst/inst_fifo_gen/gaxi_full_lite.gread_ch.grach2.gaxi_arvld.rach_pkt_reg_slice/rstblk/ngwrdrst.grst.g7serrst.gnsckt_wrst.rst_wr_reg2_inst
set_false_path -through [get_ports -scoped_to_current_instance -no_traverse src_arst]
current_instance -quiet
current_instance xilinx_zcu102_base_i/axi_ic_ps_e_S_AXI_HP2_FPD/s00_couplers/s00_data_fifo/inst/gen_fifo.fifo_gen_inst/inst_fifo_gen/reset_gen_cc.rstblk_cc/ngwrdrst.grst.g7serrst.gnsckt_wrst.rst_wr_reg2_inst
set_false_path -through [get_ports -scoped_to_current_instance -no_traverse src_arst]
current_instance -quiet
current_instance xilinx_zcu102_base_i/axi_ic_ps_e_S_AXI_HP2_FPD/s00_couplers/s00_data_fifo/inst/gen_fifo.fifo_gen_inst/inst_fifo_gen/gaxi_full_lite.gwrite_ch.gwach2.axi_wach/grf.rf/rstblk/ngwrdrst.grst.g7serrst.gnsckt_wrst.rst_wr_reg2_inst
set_false_path -through [get_ports -scoped_to_current_instance -no_traverse src_arst]
current_instance -quiet
current_instance xilinx_zcu102_base_i/axi_ic_ps_e_S_AXI_HP2_FPD/s00_couplers/s00_data_fifo/inst/gen_fifo.fifo_gen_inst/inst_fifo_gen/gaxi_full_lite.gwrite_ch.gwach2.gawvld_pkt_fifo.wach_pkt_reg_slice/rstblk/ngwrdrst.grst.g7serrst.gnsckt_wrst.rst_wr_reg2_inst
set_false_path -through [get_ports -scoped_to_current_instance -no_traverse src_arst]
current_instance -quiet
current_instance xilinx_zcu102_base_i/axi_ic_ps_e_S_AXI_HP2_FPD/s00_couplers/s00_data_fifo/inst/gen_fifo.fifo_gen_inst/inst_fifo_gen/gaxi_full_lite.gread_ch.grach2.axi_rach/grf.rf/rstblk/ngwrdrst.grst.g7serrst.gnsckt_wrst.rst_wr_reg2_inst
set_false_path -through [get_ports -scoped_to_current_instance -no_traverse src_arst]
current_instance -quiet
current_instance xilinx_zcu102_base_i/axi_ic_ps_e_S_AXI_HP0_FPD/s00_couplers/s00_data_fifo/inst/gen_fifo.fifo_gen_inst/inst_fifo_gen/gaxi_full_lite.gread_ch.grach2.gaxi_arvld.rach_pkt_reg_slice/rstblk/ngwrdrst.grst.g7serrst.gnsckt_wrst.rst_wr_reg2_inst
set_false_path -through [get_ports -scoped_to_current_instance -no_traverse src_arst]
current_instance -quiet
current_instance xilinx_zcu102_base_i/axi_ic_ps_e_S_AXI_HP0_FPD/s00_couplers/s00_data_fifo/inst/gen_fifo.fifo_gen_inst/inst_fifo_gen/reset_gen_cc.rstblk_cc/ngwrdrst.grst.g7serrst.gnsckt_wrst.rst_wr_reg2_inst
set_false_path -through [get_ports -scoped_to_current_instance -no_traverse src_arst]
current_instance -quiet
current_instance xilinx_zcu102_base_i/axi_ic_ps_e_S_AXI_HP0_FPD/s00_couplers/s00_data_fifo/inst/gen_fifo.fifo_gen_inst/inst_fifo_gen/gaxi_full_lite.gwrite_ch.gwach2.axi_wach/grf.rf/rstblk/ngwrdrst.grst.g7serrst.gnsckt_wrst.rst_wr_reg2_inst
set_false_path -through [get_ports -scoped_to_current_instance -no_traverse src_arst]
current_instance -quiet
current_instance xilinx_zcu102_base_i/axi_ic_ps_e_S_AXI_HP0_FPD/s00_couplers/s00_data_fifo/inst/gen_fifo.fifo_gen_inst/inst_fifo_gen/gaxi_full_lite.gwrite_ch.gwach2.gawvld_pkt_fifo.wach_pkt_reg_slice/rstblk/ngwrdrst.grst.g7serrst.gnsckt_wrst.rst_wr_reg2_inst
set_false_path -through [get_ports -scoped_to_current_instance -no_traverse src_arst]
current_instance -quiet
current_instance xilinx_zcu102_base_i/axi_ic_ps_e_S_AXI_HP0_FPD/s00_couplers/s00_data_fifo/inst/gen_fifo.fifo_gen_inst/inst_fifo_gen/gaxi_full_lite.gread_ch.grach2.axi_rach/grf.rf/rstblk/ngwrdrst.grst.g7serrst.gnsckt_wrst.rst_wr_reg2_inst
set_false_path -through [get_ports -scoped_to_current_instance -no_traverse src_arst]
current_instance -quiet
current_instance xilinx_zcu102_base_i/axi_ic_ps_e_S_AXI_HP1_FPD/s00_couplers/s00_data_fifo/inst/gen_fifo.fifo_gen_inst/inst_fifo_gen/gaxi_full_lite.gread_ch.grach2.gaxi_arvld.rach_pkt_reg_slice/rstblk/ngwrdrst.grst.g7serrst.gnsckt_wrst.rst_wr_reg2_inst
set_false_path -through [get_ports -scoped_to_current_instance -no_traverse src_arst]
current_instance -quiet
current_instance xilinx_zcu102_base_i/axi_ic_ps_e_S_AXI_HP1_FPD/s00_couplers/s00_data_fifo/inst/gen_fifo.fifo_gen_inst/inst_fifo_gen/reset_gen_cc.rstblk_cc/ngwrdrst.grst.g7serrst.gnsckt_wrst.rst_wr_reg2_inst
set_false_path -through [get_ports -scoped_to_current_instance -no_traverse src_arst]
current_instance -quiet
current_instance xilinx_zcu102_base_i/axi_ic_ps_e_S_AXI_HP1_FPD/s00_couplers/s00_data_fifo/inst/gen_fifo.fifo_gen_inst/inst_fifo_gen/gaxi_full_lite.gwrite_ch.gwach2.axi_wach/grf.rf/rstblk/ngwrdrst.grst.g7serrst.gnsckt_wrst.rst_wr_reg2_inst
set_false_path -through [get_ports -scoped_to_current_instance -no_traverse src_arst]
current_instance -quiet
current_instance xilinx_zcu102_base_i/axi_ic_ps_e_S_AXI_HP1_FPD/s00_couplers/s00_data_fifo/inst/gen_fifo.fifo_gen_inst/inst_fifo_gen/gaxi_full_lite.gwrite_ch.gwach2.gawvld_pkt_fifo.wach_pkt_reg_slice/rstblk/ngwrdrst.grst.g7serrst.gnsckt_wrst.rst_wr_reg2_inst
set_false_path -through [get_ports -scoped_to_current_instance -no_traverse src_arst]
current_instance -quiet
current_instance xilinx_zcu102_base_i/axi_ic_ps_e_S_AXI_HP1_FPD/s00_couplers/s00_data_fifo/inst/gen_fifo.fifo_gen_inst/inst_fifo_gen/gaxi_full_lite.gread_ch.grach2.axi_rach/grf.rf/rstblk/ngwrdrst.grst.g7serrst.gnsckt_wrst.rst_wr_reg2_inst
set_false_path -through [get_ports -scoped_to_current_instance -no_traverse src_arst]
current_instance -quiet
current_instance xilinx_zcu102_base_i/axi_ic_ps_e_S_AXI_HP1_FPD/s01_couplers/s01_data_fifo/inst/gen_fifo.fifo_gen_inst/inst_fifo_gen/gaxi_full_lite.gread_ch.grach2.gaxi_arvld.rach_pkt_reg_slice/rstblk/ngwrdrst.grst.g7serrst.gnsckt_wrst.rst_wr_reg2_inst
set_false_path -through [get_ports -scoped_to_current_instance -no_traverse src_arst]
current_instance -quiet
current_instance xilinx_zcu102_base_i/axi_ic_ps_e_S_AXI_HP1_FPD/s01_couplers/s01_data_fifo/inst/gen_fifo.fifo_gen_inst/inst_fifo_gen/reset_gen_cc.rstblk_cc/ngwrdrst.grst.g7serrst.gnsckt_wrst.rst_wr_reg2_inst
set_false_path -through [get_ports -scoped_to_current_instance -no_traverse src_arst]
current_instance -quiet
current_instance xilinx_zcu102_base_i/axi_ic_ps_e_S_AXI_HP1_FPD/s01_couplers/s01_data_fifo/inst/gen_fifo.fifo_gen_inst/inst_fifo_gen/gaxi_full_lite.gwrite_ch.gwach2.axi_wach/grf.rf/rstblk/ngwrdrst.grst.g7serrst.gnsckt_wrst.rst_wr_reg2_inst
set_false_path -through [get_ports -scoped_to_current_instance -no_traverse src_arst]
current_instance -quiet
current_instance xilinx_zcu102_base_i/axi_ic_ps_e_S_AXI_HP1_FPD/s01_couplers/s01_data_fifo/inst/gen_fifo.fifo_gen_inst/inst_fifo_gen/gaxi_full_lite.gwrite_ch.gwach2.gawvld_pkt_fifo.wach_pkt_reg_slice/rstblk/ngwrdrst.grst.g7serrst.gnsckt_wrst.rst_wr_reg2_inst
set_false_path -through [get_ports -scoped_to_current_instance -no_traverse src_arst]
current_instance -quiet
current_instance xilinx_zcu102_base_i/axi_ic_ps_e_S_AXI_HP1_FPD/s01_couplers/s01_data_fifo/inst/gen_fifo.fifo_gen_inst/inst_fifo_gen/gaxi_full_lite.gread_ch.grach2.axi_rach/grf.rf/rstblk/ngwrdrst.grst.g7serrst.gnsckt_wrst.rst_wr_reg2_inst
set_false_path -through [get_ports -scoped_to_current_instance -no_traverse src_arst]
current_instance -quiet
current_instance xilinx_zcu102_base_i/axi_ic_ps_e_S_AXI_HP0_FPD/s01_couplers/s01_data_fifo/inst/gen_fifo.fifo_gen_inst/inst_fifo_gen/gaxi_full_lite.gread_ch.grach2.gaxi_arvld.rach_pkt_reg_slice/rstblk/ngwrdrst.grst.g7serrst.gnsckt_wrst.rst_wr_reg2_inst
set_false_path -through [get_ports -scoped_to_current_instance -no_traverse src_arst]
current_instance -quiet
current_instance xilinx_zcu102_base_i/axi_ic_ps_e_S_AXI_HP0_FPD/s01_couplers/s01_data_fifo/inst/gen_fifo.fifo_gen_inst/inst_fifo_gen/reset_gen_cc.rstblk_cc/ngwrdrst.grst.g7serrst.gnsckt_wrst.rst_wr_reg2_inst
set_false_path -through [get_ports -scoped_to_current_instance -no_traverse src_arst]
current_instance -quiet
current_instance xilinx_zcu102_base_i/axi_ic_ps_e_S_AXI_HP0_FPD/s01_couplers/s01_data_fifo/inst/gen_fifo.fifo_gen_inst/inst_fifo_gen/gaxi_full_lite.gwrite_ch.gwach2.axi_wach/grf.rf/rstblk/ngwrdrst.grst.g7serrst.gnsckt_wrst.rst_wr_reg2_inst
set_false_path -through [get_ports -scoped_to_current_instance -no_traverse src_arst]
current_instance -quiet
current_instance xilinx_zcu102_base_i/axi_ic_ps_e_S_AXI_HP0_FPD/s01_couplers/s01_data_fifo/inst/gen_fifo.fifo_gen_inst/inst_fifo_gen/gaxi_full_lite.gwrite_ch.gwach2.gawvld_pkt_fifo.wach_pkt_reg_slice/rstblk/ngwrdrst.grst.g7serrst.gnsckt_wrst.rst_wr_reg2_inst
set_false_path -through [get_ports -scoped_to_current_instance -no_traverse src_arst]
current_instance -quiet
current_instance xilinx_zcu102_base_i/axi_ic_ps_e_S_AXI_HP0_FPD/s01_couplers/s01_data_fifo/inst/gen_fifo.fifo_gen_inst/inst_fifo_gen/gaxi_full_lite.gread_ch.grach2.axi_rach/grf.rf/rstblk/ngwrdrst.grst.g7serrst.gnsckt_wrst.rst_wr_reg2_inst
set_false_path -through [get_ports -scoped_to_current_instance -no_traverse src_arst]
current_instance -quiet
current_instance xilinx_zcu102_base_i/axi_ic_ps_e_S_AXI_HP0_FPD/s02_couplers/s02_data_fifo/inst/gen_fifo.fifo_gen_inst/inst_fifo_gen/gaxi_full_lite.gread_ch.grach2.gaxi_arvld.rach_pkt_reg_slice/rstblk/ngwrdrst.grst.g7serrst.gnsckt_wrst.rst_wr_reg2_inst
set_false_path -through [get_ports -scoped_to_current_instance -no_traverse src_arst]
current_instance -quiet
current_instance xilinx_zcu102_base_i/axi_ic_ps_e_S_AXI_HP0_FPD/s02_couplers/s02_data_fifo/inst/gen_fifo.fifo_gen_inst/inst_fifo_gen/reset_gen_cc.rstblk_cc/ngwrdrst.grst.g7serrst.gnsckt_wrst.rst_wr_reg2_inst
set_false_path -through [get_ports -scoped_to_current_instance -no_traverse src_arst]
current_instance -quiet
current_instance xilinx_zcu102_base_i/axi_ic_ps_e_S_AXI_HP0_FPD/s02_couplers/s02_data_fifo/inst/gen_fifo.fifo_gen_inst/inst_fifo_gen/gaxi_full_lite.gwrite_ch.gwach2.axi_wach/grf.rf/rstblk/ngwrdrst.grst.g7serrst.gnsckt_wrst.rst_wr_reg2_inst
set_false_path -through [get_ports -scoped_to_current_instance -no_traverse src_arst]
current_instance -quiet
current_instance xilinx_zcu102_base_i/axi_ic_ps_e_S_AXI_HP0_FPD/s02_couplers/s02_data_fifo/inst/gen_fifo.fifo_gen_inst/inst_fifo_gen/gaxi_full_lite.gwrite_ch.gwach2.gawvld_pkt_fifo.wach_pkt_reg_slice/rstblk/ngwrdrst.grst.g7serrst.gnsckt_wrst.rst_wr_reg2_inst
set_false_path -through [get_ports -scoped_to_current_instance -no_traverse src_arst]
current_instance -quiet
current_instance xilinx_zcu102_base_i/axi_ic_ps_e_S_AXI_HP0_FPD/s02_couplers/s02_data_fifo/inst/gen_fifo.fifo_gen_inst/inst_fifo_gen/gaxi_full_lite.gread_ch.grach2.axi_rach/grf.rf/rstblk/ngwrdrst.grst.g7serrst.gnsckt_wrst.rst_wr_reg2_inst
set_false_path -through [get_ports -scoped_to_current_instance -no_traverse src_arst]
current_instance -quiet
current_instance xilinx_zcu102_base_i/axi_ic_ps_e_S_AXI_HP0_FPD/s03_couplers/s03_data_fifo/inst/gen_fifo.fifo_gen_inst/inst_fifo_gen/gaxi_full_lite.gread_ch.grach2.gaxi_arvld.rach_pkt_reg_slice/rstblk/ngwrdrst.grst.g7serrst.gnsckt_wrst.rst_wr_reg2_inst
set_false_path -through [get_ports -scoped_to_current_instance -no_traverse src_arst]
current_instance -quiet
current_instance xilinx_zcu102_base_i/axi_ic_ps_e_S_AXI_HP0_FPD/s03_couplers/s03_data_fifo/inst/gen_fifo.fifo_gen_inst/inst_fifo_gen/reset_gen_cc.rstblk_cc/ngwrdrst.grst.g7serrst.gnsckt_wrst.rst_wr_reg2_inst
set_false_path -through [get_ports -scoped_to_current_instance -no_traverse src_arst]
current_instance -quiet
current_instance xilinx_zcu102_base_i/axi_ic_ps_e_S_AXI_HP0_FPD/s03_couplers/s03_data_fifo/inst/gen_fifo.fifo_gen_inst/inst_fifo_gen/gaxi_full_lite.gwrite_ch.gwach2.axi_wach/grf.rf/rstblk/ngwrdrst.grst.g7serrst.gnsckt_wrst.rst_wr_reg2_inst
set_false_path -through [get_ports -scoped_to_current_instance -no_traverse src_arst]
current_instance -quiet
current_instance xilinx_zcu102_base_i/axi_ic_ps_e_S_AXI_HP0_FPD/s03_couplers/s03_data_fifo/inst/gen_fifo.fifo_gen_inst/inst_fifo_gen/gaxi_full_lite.gwrite_ch.gwach2.gawvld_pkt_fifo.wach_pkt_reg_slice/rstblk/ngwrdrst.grst.g7serrst.gnsckt_wrst.rst_wr_reg2_inst
set_false_path -through [get_ports -scoped_to_current_instance -no_traverse src_arst]
current_instance -quiet
current_instance xilinx_zcu102_base_i/axi_ic_ps_e_S_AXI_HP0_FPD/s03_couplers/s03_data_fifo/inst/gen_fifo.fifo_gen_inst/inst_fifo_gen/gaxi_full_lite.gread_ch.grach2.axi_rach/grf.rf/rstblk/ngwrdrst.grst.g7serrst.gnsckt_wrst.rst_wr_reg2_inst
set_false_path -through [get_ports -scoped_to_current_instance -no_traverse src_arst]

####################################################################################
# Constraints from file : 'xpm_cdc_sync_rst.tcl'
####################################################################################

current_instance -quiet
current_instance xilinx_zcu102_base_i/axi_ic_ps_e_S_AXI_HP2_FPD/s00_couplers/s00_data_fifo/inst/gen_fifo.fifo_gen_inst/inst_fifo_gen/gaxi_full_lite.gread_ch.grdch2.axi_rdch/grf.rf/rstblk/ngwrdrst.grst.g7serrst.gsckt_wrst.xpm_cdc_sync_rst_inst_wrst
set_false_path -to [get_cells {syncstages_ff_reg[0]}]
current_instance -quiet
current_instance xilinx_zcu102_base_i/axi_ic_ps_e_S_AXI_HP2_FPD/s00_couplers/s00_data_fifo/inst/gen_fifo.fifo_gen_inst/inst_fifo_gen/gaxi_full_lite.gwrite_ch.gwdch2.axi_wdch/grf.rf/rstblk/ngwrdrst.grst.g7serrst.gsckt_wrst.xpm_cdc_sync_rst_inst_wrst
set_false_path -to [get_cells {syncstages_ff_reg[0]}]
current_instance -quiet
current_instance xilinx_zcu102_base_i/axi_ic_ps_e_S_AXI_HP0_FPD/s00_couplers/s00_data_fifo/inst/gen_fifo.fifo_gen_inst/inst_fifo_gen/gaxi_full_lite.gread_ch.grdch2.axi_rdch/grf.rf/rstblk/ngwrdrst.grst.g7serrst.gsckt_wrst.xpm_cdc_sync_rst_inst_wrst
set_false_path -to [get_cells {syncstages_ff_reg[0]}]
current_instance -quiet
current_instance xilinx_zcu102_base_i/axi_ic_ps_e_S_AXI_HP0_FPD/s00_couplers/s00_data_fifo/inst/gen_fifo.fifo_gen_inst/inst_fifo_gen/gaxi_full_lite.gwrite_ch.gwdch2.axi_wdch/grf.rf/rstblk/ngwrdrst.grst.g7serrst.gsckt_wrst.xpm_cdc_sync_rst_inst_wrst
set_false_path -to [get_cells {syncstages_ff_reg[0]}]
current_instance -quiet
current_instance xilinx_zcu102_base_i/axi_ic_ps_e_S_AXI_HP1_FPD/s00_couplers/s00_data_fifo/inst/gen_fifo.fifo_gen_inst/inst_fifo_gen/gaxi_full_lite.gread_ch.grdch2.axi_rdch/grf.rf/rstblk/ngwrdrst.grst.g7serrst.gsckt_wrst.xpm_cdc_sync_rst_inst_wrst
set_false_path -to [get_cells {syncstages_ff_reg[0]}]
current_instance -quiet
current_instance xilinx_zcu102_base_i/axi_ic_ps_e_S_AXI_HP1_FPD/s00_couplers/s00_data_fifo/inst/gen_fifo.fifo_gen_inst/inst_fifo_gen/gaxi_full_lite.gwrite_ch.gwdch2.axi_wdch/grf.rf/rstblk/ngwrdrst.grst.g7serrst.gsckt_wrst.xpm_cdc_sync_rst_inst_wrst
set_false_path -to [get_cells {syncstages_ff_reg[0]}]
current_instance -quiet
current_instance xilinx_zcu102_base_i/axi_ic_ps_e_S_AXI_HP1_FPD/s01_couplers/s01_data_fifo/inst/gen_fifo.fifo_gen_inst/inst_fifo_gen/gaxi_full_lite.gread_ch.grdch2.axi_rdch/grf.rf/rstblk/ngwrdrst.grst.g7serrst.gsckt_wrst.xpm_cdc_sync_rst_inst_wrst
set_false_path -to [get_cells {syncstages_ff_reg[0]}]
current_instance -quiet
current_instance xilinx_zcu102_base_i/axi_ic_ps_e_S_AXI_HP1_FPD/s01_couplers/s01_data_fifo/inst/gen_fifo.fifo_gen_inst/inst_fifo_gen/gaxi_full_lite.gwrite_ch.gwdch2.axi_wdch/grf.rf/rstblk/ngwrdrst.grst.g7serrst.gsckt_wrst.xpm_cdc_sync_rst_inst_wrst
set_false_path -to [get_cells {syncstages_ff_reg[0]}]
current_instance -quiet
current_instance xilinx_zcu102_base_i/axi_ic_ps_e_S_AXI_HP0_FPD/s01_couplers/s01_data_fifo/inst/gen_fifo.fifo_gen_inst/inst_fifo_gen/gaxi_full_lite.gread_ch.grdch2.axi_rdch/grf.rf/rstblk/ngwrdrst.grst.g7serrst.gsckt_wrst.xpm_cdc_sync_rst_inst_wrst
set_false_path -to [get_cells {syncstages_ff_reg[0]}]
current_instance -quiet
current_instance xilinx_zcu102_base_i/axi_ic_ps_e_S_AXI_HP0_FPD/s01_couplers/s01_data_fifo/inst/gen_fifo.fifo_gen_inst/inst_fifo_gen/gaxi_full_lite.gwrite_ch.gwdch2.axi_wdch/grf.rf/rstblk/ngwrdrst.grst.g7serrst.gsckt_wrst.xpm_cdc_sync_rst_inst_wrst
set_false_path -to [get_cells {syncstages_ff_reg[0]}]
current_instance -quiet
current_instance xilinx_zcu102_base_i/axi_ic_ps_e_S_AXI_HP0_FPD/s02_couplers/s02_data_fifo/inst/gen_fifo.fifo_gen_inst/inst_fifo_gen/gaxi_full_lite.gread_ch.grdch2.axi_rdch/grf.rf/rstblk/ngwrdrst.grst.g7serrst.gsckt_wrst.xpm_cdc_sync_rst_inst_wrst
set_false_path -to [get_cells {syncstages_ff_reg[0]}]
current_instance -quiet
current_instance xilinx_zcu102_base_i/axi_ic_ps_e_S_AXI_HP0_FPD/s02_couplers/s02_data_fifo/inst/gen_fifo.fifo_gen_inst/inst_fifo_gen/gaxi_full_lite.gwrite_ch.gwdch2.axi_wdch/grf.rf/rstblk/ngwrdrst.grst.g7serrst.gsckt_wrst.xpm_cdc_sync_rst_inst_wrst
set_false_path -to [get_cells {syncstages_ff_reg[0]}]
current_instance -quiet
current_instance xilinx_zcu102_base_i/axi_ic_ps_e_S_AXI_HP0_FPD/s03_couplers/s03_data_fifo/inst/gen_fifo.fifo_gen_inst/inst_fifo_gen/gaxi_full_lite.gread_ch.grdch2.axi_rdch/grf.rf/rstblk/ngwrdrst.grst.g7serrst.gsckt_wrst.xpm_cdc_sync_rst_inst_wrst
set_false_path -to [get_cells {syncstages_ff_reg[0]}]
current_instance -quiet
current_instance xilinx_zcu102_base_i/axi_ic_ps_e_S_AXI_HP0_FPD/s03_couplers/s03_data_fifo/inst/gen_fifo.fifo_gen_inst/inst_fifo_gen/gaxi_full_lite.gwrite_ch.gwdch2.axi_wdch/grf.rf/rstblk/ngwrdrst.grst.g7serrst.gsckt_wrst.xpm_cdc_sync_rst_inst_wrst
set_false_path -to [get_cells {syncstages_ff_reg[0]}]

# Vivado Generated miscellaneous constraints 

#revert back to original instance
current_instance -quiet
