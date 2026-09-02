library(ggplot2)
library(Cairo)

args <- commandArgs(trailingOnly = TRUE)
input_file <- args[1]
rm(args)

d = read.table(input_file, header = T)

data_seq  = subset(d, m*N >= 128 & m == 1)
data = subset(d, m*N >= 128 & m < 1024)

q <- ggplot() +
geom_line (data=data, aes(x=m*N, y=mean_error, color=factor(m), size=factor(m))) + 
geom_point(data=data, aes(x=m*N, y=mean_error, color=factor(m), fill=factor(m), shape=factor(m))) +
geom_line (data=data_seq, aes(x=m*N, y=mean_error), color='black') + 
scale_colour_manual(
	name   = "Centralized/\nDistributed\n(particles p/filter)",
	breaks = c(2,4,8,16,32,64,128,256,512,1024,1),
	labels = c(
		'distr. (2)',
		'distr. (4)',
		'distr. (8)',
		'distr. (16)',
		'distr. (32)',
		'distr. (64)',
		'distr. (128)',
		'distr. (256)',
		'distr. (512)',
		'distr. (1024)',
		'centralized'
	),
	values = c(
		"#000000",
		"#F8766D",
		"#D39200",
		"#93AA00",
		"#00BA38",
		"#00C19F",
		"#00B9E3",
		"#619CFF",
		"#DB72FB",
		"#FF61C3"
	)
) +
scale_fill_manual(
	name = "Centralized/\nDistributed\n(particles p/filter)",
	breaks = c(2,4,8,16,32,64,128,256,512,1024,1),
	labels = c(
		'distr. (2)',
		'distr. (4)',
		'distr. (8)',
		'distr. (16)',
		'distr. (32)',
		'distr. (64)',
		'distr. (128)',
		'distr. (256)',
		'distr. (512)',
		'distr. (1024)',
		'centralized'
	),
	values = c(
		"#000000",
		"#F8766D",
		"#D39200",
		"#93AA00",
		"#00BA38",
		"#00C19F",
		"#00B9E3",
		"#619CFF",
		"#DB72FB",
		"#FF61C3"
	)
) +
scale_shape_manual(
	name   = "Centralized/\nDistributed\n(particles p/filter)",
	breaks = c(2,4,8,16,32,64,128,256,512,1024,1),
	labels = c(
		'distr. (2)',
		'distr. (4)',
		'distr. (8)',
		'distr. (16)',
		'distr. (32)',
		'distr. (64)',
		'distr. (128)',
		'distr. (256)',
		'distr. (512)',
		'distr. (1024)',
		'centralized'
	),
	values = c(16,15,1,23,13,0,8,5,4,17)
) +
scale_size_manual(guide='none', values = c(1,.5,.5,.5,.5,.5,.5,.5,.5,.5)) +
scale_x_continuous(
	"Number of particles",
	labels = c('256', '1024', '4K', '16K', '64K', '256K', '1M', '4M'),
	breaks = c(2^8, 2^10, 2^12, 2^14, 2^16, 2^18, 2^20, 2^22),
	minor_breaks = waiver(),
	trans = "log2"
) +
scale_y_continuous(
	"Estimation error [-]",
	labels = c(0.25, 0.5, 1, 2, 4, 8, 16),
	breaks = c(2^-2, 2^-1, 2^0, 2^1, 2^2, 2^3, 2^4),
	minor_breaks = waiver(),
	trans = "log2"
) +
coord_cartesian(ylim = c(2^-2, 2^4), xlim = c(2^6, 2^24)) + 
theme_bw() +
theme(
	aspect.ratio = 8/9,
	panel.grid.major = element_line(linewidth = 0.2, colour = "grey80", linetype = "solid"),
	panel.grid.minor = element_line(linewidth = 0.2, colour = "grey95", linetype = "solid"),
	legend.key = element_blank()
)

CairoPDF(paste(basename(input_file),'.pdf', sep=""), title=input_file, width=6, height=4)
print(q)
dev.off()

