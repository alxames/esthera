library(ggplot2)
library(Cairo)

args <- commandArgs(trailingOnly = TRUE)
input_file <- args[1]
rm(args)

data = read.table(input_file, header = T)

data_ocl <- droplevels(subset(data,
	dev == 'ocl_gtx680' |
	dev == 'ocl_hd7970' |
	dev == 'ocl_i7-2820QM'))
data_seq <- droplevels(subset(data,
	dev == 'c_i7-2820QM'))

q <- ggplot() +
geom_point(data=data_ocl, aes(x=N*m, y=t7/1000, color=dev, shape=resampling)) +
geom_line (data=data_ocl, aes(x=N*m, y=t7/1000, color=dev, shape=resampling)) +
geom_point(data=data_seq, aes(x=N*m, y=t7/1000, color=dev, shape=resampling)) +
geom_line (data=data_seq, aes(x=N*m, y=t7/1000, color=dev, shape=resampling)) +
scale_color_manual(
	name = 'Platform',
	breaks = c(
		'ocl_gtx680',
		'ocl_hd7970',
		'ocl_i7-2820QM',
		'c_i7-2820QM'),
	labels = c(
		'OpenCL - GTX 680',
		'OpenCL - HD 7970',
		'OpenCL - i7-2820QM',
		'C (centr.) - i7-2820QM'),
	# colors from colorbrewer.org
	values = c("#08519C", "#BDD7E7", "#6BAED6", "#3182BD")
) +
scale_shape_discrete(
	name = 'Resampling Method',
	breaks = c(
		'vose',
		'novose'),
	labels = c(
		'Alias (Vose)',
		'Roulette Wheel')
) +
scale_x_continuous(
	"Number of particles",
	labels = c('256', '1K','4K','16K','64K','256K','1M','4M'),
	breaks = c(2^8, 2^10, 2^12, 2^14, 2^16, 2^18, 2^20, 2^22),
	minor_breaks = c(0),
	trans = "log2"
) +
scale_y_continuous(
	"Resampling [ms]",
	labels = c(.015, .25, 4, 64, 1024, 16384),
	breaks = c(2^-6, 2^-2, 2^2, 2^6, 2^10, 2^14),
	minor_breaks = c(0),
	trans = "log2"
) +
coord_cartesian(ylim = c(2^-10, 2^18), xlim = c(2^6, 2^24)) +
theme_bw() +
theme(
	aspect.ratio = 8/9,
	panel.grid.major = element_line(linewidth = 0.2, colour = "grey80", linetype = "solid"),
	panel.grid.minor = element_line(linewidth = 0.2, colour = "grey95", linetype = "solid"),
	legend.key = element_blank()
)

CairoPDF(paste(basename(input_file),'.pdf', sep=""), title=input_file, width=6, height=3.5)
print(q)
dev.off()

