library(ggplot2)
library(Cairo)
library(grid)

args <- commandArgs(trailingOnly = TRUE)
input_file <- args[1]
rm(args)

d <- read.table(input_file, header = T)
data <- subset(d, key != "total")

q <- ggplot(data, aes(x=s, y=val, fill=key)) +
geom_bar(position="fill", stat="identity") +
scale_x_continuous(
	"State dimensions",
	breaks = c(8,16,24,32,40,48),
	minor_breaks = NULL
) +
scale_y_continuous(
	"",
	breaks = c(0, 0.2, 0.4, 0.6, 0.8, 1.0),
	minor_breaks = NULL
) +
scale_fill_brewer(
	name   = "",
	breaks = c("t6","t5","t4","t3","t2","t1"),
	labels = c("resampling","exchange","global estimate","local sort","sampling","rand")
) +
theme_bw() +
theme(
	panel.grid.major = element_line(linewidth = 0.2, colour = "grey80", linetype = "solid"),
	panel.grid.minor = element_blank(),
	text = element_text(size = 22),
	plot.margin = unit(c(0,0,0,0), 'lines'),
	panel.margin = unit(c(0,0,0,0), 'lines'),
	legend.key.size = unit(c(8), 'mm')
)

CairoPDF(paste(basename(input_file),'_relative.pdf', sep=""), title=input_file, width=7, height=5)
print(q)
dev.off()

