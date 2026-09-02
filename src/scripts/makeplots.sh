#!/bin/sh
Rscript rate2.R perf_avg
Rscript scale_m.R sm_580
Rscript scale_N.R sN_580
Rscript scale_s.R ss_580
Rscript resampling.R valias
#Rscript plot.R topo-alltoall
#Rscript plot.R topo-ring
#Rscript plot.R topo-2dtorus
#Rscript plot_t.R exchange-t 0
#Rscript plot_t.R exchange-t 1
#Rscript plot_t.R exchange-t 2
Rscript trace_path.R trace0
Rscript est.R acc_stat
