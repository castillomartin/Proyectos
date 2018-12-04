gnuplot << EOP
set yrange[0.96:1]
set terminal jpeg size 640,480
set output "efficiency.jpg"
set title "Efficiency"
plot 'result4096x' u 1:5 w l
EOP
	