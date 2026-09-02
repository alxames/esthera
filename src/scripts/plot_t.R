library(ggplot2)
library(Cairo)
library(grid)
library(proto)

# there is a bug in stat_summary() in ggplot 0.9.3
# https://github.com/hadley/ggplot2/issues/732
# revert to that of 0.9.2.1
StatSummary <- proto(ggplot2:::Stat, {
	objname <- "summary"
	default_geom <- function(.) GeomPointrange
	required_aes <- c("x", "y")
	calculate_groups <- function(., data, scales, fun.data = NULL, fun.y = NULL, fun.ymax = NULL, fun.ymin = NULL, na.rm = FALSE, ...) {
		data <- remove_missing(data, na.rm, c("x", "y"), name = "stat_summary")
		if (!missing(fun.data)) {
			fun.data <- match.fun(fun.data)
			fun <- function(df, ...) {
				fun.data(df$y, ...)
			}
		} else {
			fs <- compact(list(ymin = fun.ymin, y = fun.y, ymax = fun.ymax))
			fun <- function(df, ...) {
				res <- llply(fs, function(f) do.call(f, list(df$y, ...)))
				names(res) <- names(fs)
				as.data.frame(res)
			}
		}
		summarise_by_x(data, fun, ...)
	}
})
assignInNamespace("StatSummary", StatSummary, pos = "package:ggplot2")

args <- commandArgs(trailingOnly = TRUE)
input_file <- args[1]
trans <- args[2]
rm(args)

d = read.table(input_file, header = T)
data = subset(d, t == trans)

q <- ggplot(data, aes(N, e, color=factor(m))) + 
stat_summary(aes(group = m), fun.y = mean, geom = "line") +
stat_summary(aes(group = m, shape=factor(m)), size=3, fun.y = mean, geom = "point") +
scale_x_continuous(
	"Number of filters",
	labels = c(4,8,16,32,64,128,256,512,'1K','2K'),
	breaks = c(4,8,16,32,64,128,256,512,1024,2048),
	minor_breaks = waiver(),
	trans = "log2"
) +
scale_y_continuous(
	"Estimation error [-]",
	labels = c(0.25,0.5,1,2,4,8,16,32,64),
	breaks = c(0.25,0.5,1,2,4,8,16,32,64),
	minor_breaks = waiver(),
	trans = "log2"
) +
coord_cartesian(ylim = c(2^-2.5, 2^6.5), xlim = c(2^1.5, 2^11.5)) + 
scale_colour_manual(
	name = "Particles\np/filter",
	breaks = c(4, 8, 16, 32, 64, 128, 256, 512),
	values = c(
	          "4" = "#F8766D",
                  "8" = "#CD9600",
                 "16" = "#7CAE00",
                 "32" = "#00BE67",
                 "64" = "#00BFC4",
                "128" = "#00A9FF",
                "256" = "#C77CFF",
                "512" = "#FF61CC")
) +
scale_shape_manual(
	name = "Particles\np/filter",
	breaks = c(4, 8, 16, 32, 64, 128, 256, 512),
	values = c("4"=15, "8"=16, "16"=17, "32"=18, "64"=3, "128"=4, "256"=7, "512"=8)
) +
theme_bw() +
theme(
	aspect.ratio = 9/10,
	plot.margin=unit(c(0.5,0.5,0.5,0.5), "lines"),
	panel.grid.major = element_line(linewidth = 0.2, colour = "grey80", linetype = "solid"),
	panel.grid.minor = element_line(linewidth = 0.2, colour = "grey95", linetype = "solid"),
	legend.key = element_blank(),
	text = element_text(size=22),
	plot.margin = unit(c(0,0,0,0), 'lines'),
	panel.margin = unit(c(0,0,0,0), 'lines'),
	legend.key.size = unit(c(8), 'mm')
)

CairoPDF(paste(paste(basename(input_file), trans, sep=""), '.pdf', sep=""), width=7, height=5, title=input_file)
print(q)
dev.off()

