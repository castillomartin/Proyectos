gnuplot << EOP
set yrange[0:60]
set terminal jpeg size 640,480
set output "total.jpg"
set title "total"
plot 'plot' u 1:2 w l
EOP
	