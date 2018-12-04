gnuplot << EOP
set yrange[0.08:0.1]
set terminal jpeg size 640,480
set output "total.jpg"
set title "total"
plot 'result4096x' u 1:2 w l
EOP
	