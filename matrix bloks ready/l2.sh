#!/bin/bash       
    # gnuplot <<- EOF
        # set xlabel "Type"
        # set ylabel "Time"
        # set title "Time in Sec Matrix 1000x1000"   
        # set term png
        # set output "adsa.png"
        # plot "plot.bin" using 1:2 with linespoints
	# EOF
	
echo | gnuplot -p -e "plot 'l2.bin' using 1:2 with linespoints"
