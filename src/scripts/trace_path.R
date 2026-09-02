library(ggplot2)
library(Cairo)

args <- commandArgs(trailingOnly = TRUE)

input_file <- args[1]
rm(args)

data=read.table(input_file, header=T)

q <- ggplot(data) +
geom_point(aes(x=ax,y=ay, colour='actual'), size=.5) +
geom_point(aes(x=gx,y=gy, colour='estimate1'), size=.5) +
geom_point(aes(x=bx,y=by, colour='estimate2'), size=.5) +
coord_cartesian(ylim = c(-7.5, 7.5), xlim = c(-17.5, 17.5)) + 
scale_x_continuous(name='', breaks = c(-15,-10,-5,0,5,10,15)) +
scale_y_continuous(name='', breaks = c(-5,0,5)) +
scale_colour_manual(
	name='Trajectory',
	values=c('actual'='#2171B5','estimate1'='#6BAED6','estimate2'='#BDD7E7'),
	labels = c('Ground truth', 'High particle\nestimation', 'Low particle\nestimation')
) +
guides(colour=guide_legend(keyheight=2, override.aes = list(size = 2))) +
theme_bw() +
theme(
	panel.grid.major = element_line(linewidth = 0.2, colour = 'grey80', linetype = 'solid'),
	panel.grid.minor = element_line(linewidth = 0.2, colour = 'grey95', linetype = 'solid'),
	legend.key = element_blank()
)

CairoPDF(paste(basename(input_file),'.pdf', sep=''), title=input_file, width=7, height=3)
print(q)
dev.off()

