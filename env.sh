# Dump data Options
###### XPUSIM_DUMP_PATH= ["", "dump", "AnyNameYouLike", ...]
###### XPUSIM_DUMP_MODE= [MPW, SSE_IP, SSE_SOC]
###### XPUSIM_DUMP_INSTR= ["", "instr", "AnyNameYouLike", ...]
export XPUSIM_DUMP_PATH=""
export XPUSIM_DUMP_MODE=SSE_IP
export XPUSIM_DUMP_INSTR=""

# Memory Allocator
###### (HBM_HADDR, HBM_BASE)=(2, 0x00001000)
###### L3_RANGE=(0, 0xC0000000) ~ (0, 0xC0FFFFFF), add following code at python to enforce GC: gc.collect();check_call(_LIB.polaris_reset())
###### XPUSIM_MALLOC_HADDR= [0, 1, 2, 3, ..., 7]
###### XPUSIM_MALLOC_BASE= [0x00001000, 0xC0000000]
export XPUSIM_MALLOC_HADDR=2
export XPUSIM_MALLOC_BASE=0x00001000

# Performance Simulator
###### XPUSIM_SIMULATOR_MODE= [FUNCTION, SYSTEMC]
###### XPUSIM_SSE_LOG_LEVEL= [DISABLE, INFO]
###### XPUSIM_HBM_READ_LATENCY= [250, ...]
###### XPUSIM_HBM_WRITE_LATENCY= [90, ...]
unset XPUSIM_SIMULATOR_MODE
unset XPUSIM_SSE_LOG_LEVEL
#export XPUSIM_SIMULATOR_MODE=SYSTEMC
#export XPUSIM_SIMULATOR_MODE=FUNCTION
#export XPUSIM_SSE_LOG_LEVEL=INFO
export XPUSIM_HBM_READ_LATENCY=250
export XPUSIM_HBM_WRITE_LATENCY=90

# Trace
###### XPUSIM_DUMP_TRACE_EVENT= ["", "trace", "AnyNameYouLike", ...]
###### XPUSIM_TRACE_EVENT_DISABLE="sse,issue_cdnn"
export XPUSIM_DUMP_TRACE_EVENT=""
export XPUSIM_TRACE_EVENT_DISABLE="sse,issue_cdnn"

# for more detail, ref to
# http://icode.baidu.com/repos/baidu/xpu/simulator/blob/master:common/config_list.inc
