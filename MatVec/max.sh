gnuplot << EOP
set yrange[0.001:0.0003]
set terminal jpeg size 640,480
set output "Max.jpg"
set title "Max"
plot 'result4096x' u 1:3 w l
EOP
