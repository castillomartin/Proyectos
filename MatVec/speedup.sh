gnuplot << EOP
set yrange[30:500]
set terminal jpeg size 640,480
set output "Speedup.jpg"
set title "Speedup"
plot 'result4096x' u 1:4 w l
EOP
	