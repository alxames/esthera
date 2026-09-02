library(ggplot2)
library(Cairo)

args <- commandArgs(trailingOnly = TRUE)
input_file <- args[1]
rm(args)

d <- read.table(input_file, header = T)

data_ocl <- droplevels(subset(d,
	dev == 'ocl_gtx680_novose' |
	dev == 'ocl_gtx580_novose' |
	dev == 'ocl_hd7970_novose' |
	dev == 'ocl_hd6970_novose' |
	dev == 'ocl_i7-2820QM_novose' |
	dev == 'ocl_e5-2680_novose'))
data_seq_i7 <- droplevels(subset(d, N*m >= 256 & dev == 'c_i7-2820QM'))
data_seq_e5 <- droplevels(subset(d, N*m >= 256 & dev == 'c_e5-2680'))

data <- rbind(data_ocl, data_seq_e5, data_seq_i7)

q <- ggplot() +
geom_point(data=data, aes(x=N*m, y=1000000.0/total, color=dev, shape=dev)) +
geom_line(data=data, aes(x=N*m, y=1000000.0/total, color=dev)) +
scale_color_manual(
	name = 'Distributed\n(OpenCL)',
	labels = c(
		'GTX 680',
		'GTX 580',
		'HD 7970',
		'HD 6970',
		'2x E5-2650',
		'i7-2820QM',
		'2x E5-2680*',
		'i7-2820QM*'
	),
	breaks = c(
		'ocl_gtx680_novose',
		'ocl_gtx580_novose',
		'ocl_hd7970_novose',
		'ocl_hd6970_novose',
		'ocl_e5-2680_novose',
		'ocl_i7-2820QM_novose',
		'c_e5-2680',
		'c_i7-2820QM'
	),
	values = c(
		'ocl_gtx680_novose'    = '#F8766D',
		'ocl_gtx580_novose'    = '#B79F00',
		'ocl_hd7970_novose'    = '#00BA38',
		'ocl_hd6970_novose'    = '#00BFC4',
		'ocl_e5-2680_novose'   = '#619CFF',
		'ocl_i7-2820QM_novose' = '#F564E3',
		'c_e5-2680'            = '#606060',
		'c_i7-2820QM'          = '#000000'
	)
) +
scale_shape_manual(
	#name = 'Centralized (C)',
	name = 'Distributed\n(OpenCL)',
	labels = c(
		'GTX 680',
		'GTX 580',
		'HD 7970',
		'HD 6970',
		'2x E5-2650',
		'i7-2820QM',
		'2x E5-2680*',
		'i7-2820QM*'
	),
	breaks = c(
		'ocl_gtx680_novose',
		'ocl_gtx580_novose',
		'ocl_hd7970_novose',
		'ocl_hd6970_novose',
		'ocl_e5-2680_novose',
		'ocl_i7-2820QM_novose',
		'c_e5-2680',
		'c_i7-2820QM'
	),
	values = c(
		'ocl_gtx680_novose'    = 17,
		'ocl_gtx580_novose'    = 3,
		'ocl_hd7970_novose'    = 23,
		'ocl_hd6970_novose'    = 8,
		'ocl_e5-2680_novose'   = 15,
		'ocl_i7-2820QM_novose' = 16,
		'c_e5-2680'            = 0,
		'c_i7-2820QM'          = 1
	)
) +
scale_x_continuous(
	"Number of particles",
	labels = c('256', '1K','4K','16K','64K','256K','1M','4M'),
	breaks = c(2^8, 2^10, 2^12, 2^14, 2^16, 2^18, 2^20, 2^22),
	minor_breaks = waiver(),
	trans = "log2"
) +
scale_y_continuous(
	"Update rate [Hz]",
	labels = c(1,4,16,64,256,1024,4096),
	breaks = c(2^0,2^2,2^4,2^6,2^8,2^10,2^12),
	minor_breaks = waiver(),
	trans = "log2"
) +
coord_cartesian(ylim = c(2^-2, 2^14), xlim = c(2^6, 2^24)) +
#guides(colour = guide_legend(order = 1), shape = guide_legend(order = 2)) +
theme_bw() +
theme(
	aspect.ratio = 8/9,
	panel.grid.major = element_line(linewidth = 0.2, colour = "grey80", linetype = "solid"),
	panel.grid.minor = element_line(linewidth = 0.2, colour = "grey95", linetype = "solid"),
	legend.key = element_blank()
)
CairoPDF(paste(basename(input_file),'2.pdf', sep=""), title=input_file, width=6, height=4.25)
#CairoSVG(paste(basename(input_file),'2.svg', sep=""), title=input_file, width=6, height=4.25)
print(q)
dev.off()

