#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int Min(int a,int b);

int Low(int rank, int sizee, int n, int m);

int High(int rank, int sizee, int n, int m);

int Lenght(int rank, int sizee, int n, int m);

int main(int argc, char** argv){

    int     count,count2;
    double  elapsed_time;
    int     first,global_count,global_count2,high_value,i,my_rank ,index,low_value,n,m,my_size,prime;
    int     size,first_value_index,prime_doubled,koren,p;
    int     num_per_block,block_low_value,block_high_value;
    char*   marked,*primes;

    MPI_Init(&argc, &argv);

    /* start the timer */
    MPI_Barrier(MPI_COMM_WORLD);
    elapsed_time = -MPI_Wtime();

    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &my_size);

    if (argc != 3)    {
        if (my_rank == 0)
            printf("Range need: %s <n> <m>\n", argv[0]);
        MPI_Finalize();
        exit(1);
    }

    m = atoi(argv[1]);
    n = atoi(argv[2]);

    low_value  = 3 + Low(my_rank, my_size, n - 1,m)  * 2;
    high_value = 3 + High(my_rank, my_size, n - 1,m) * 2;
    size       = Lenght(my_rank, my_size, n - 1,m);

    koren = sqrt(n);
    primes = (char*)calloc(koren + 1, 1);
    for (p = 2; p <= koren; p += 2)    {
        primes[p] = 1;

    }

    for (prime = 3; prime <= koren; prime += 2)    {
        if (primes[prime] == 1)
            continue;

        for (p = prime * 2; p <= koren; p += prime)    {
            primes[p] = 1;
        }
    }

    marked = (char*)calloc(size * sizeof(char), 1);
    marked[1] = 0;
    num_per_block    = 1048576;
    block_low_value  = low_value;
    block_high_value = Min(high_value,low_value + num_per_block * 2);

//    printf("rank : %d\n",my_rank);
//    printf("lv : %d\n",low_value);
//    printf("lh : %d\n",high_value);
//    printf("s : %d\n",size);


    for (i = 0;i < size; i += num_per_block){
        for (prime = 3; prime <= koren; prime++){

            if (primes[prime] == 1)
                continue;

            if (prime * prime > block_low_value){
                first = prime * prime;
            }
           else{
                if (!(block_low_value % prime))    {
                    first = block_low_value;
                }
                else    {
                    first = prime - (block_low_value % prime) +
                            block_low_value;
                }
           }

           if ((first + prime) & 1) // is odd
              first += prime;


           first_value_index = (first - 3) / 2 - Low(my_rank, my_size, n - 1,m);
           prime_doubled     = prime * 2 ;

           for (i = first; i <= high_value; i += prime_doubled)   {
			   marked[first_value_index] = 1;
               first_value_index += prime;
           }
        }

        block_low_value += num_per_block * 2;
        block_high_value = Min(high_value,block_high_value + num_per_block * 2);

    }




	char file[] = "rankX.bin";
	file[4] = my_rank+'0';
    FILE * f = fopen(file,"a+b");
    count = 0;
    count2 = 0;

    if(my_rank == 0){
        if(2 >=m && 2 <=n)
                fprintf(f,"%d\n",2);

    }
    int pos;
    for (i = 0; i < size; i++){
            pos = (i+Low(my_rank, my_size, n - 1,m))*2+3;
        if (!marked[i]){
            count++;
            if(pos < m)
                count2++;
             if(pos >=m && pos <=n)
                fprintf(f,"%d\n",(i+Low(my_rank, my_size, n - 1,m))*2+3);
            }
    }
    fclose(f);
    MPI_Reduce(&count, &global_count, 1, MPI_INT,MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&count2, &global_count2, 1, MPI_INT,MPI_SUM, 0, MPI_COMM_WORLD);

    elapsed_time += MPI_Wtime();

    if (my_rank == 0)   {
        global_count += 1;
        global_count2 += 1;
        //printf("%d primes are less than or equal to %d\n",global_count2, m);
        //printf("%d primes are less than or equal to %d\n",global_count, n);
        printf("there are %d number between %d and %d\n",global_count-global_count2, m,n);
        printf("Total elapsed time: %10.6fs\n",elapsed_time);
    }

    MPI_Finalize();

    return 0;
}



int Min(int a,int b){
    return (a<b)?a:b;
}
int Low(int rank, int sizee, int n, int m){
    return (m+ rank*(n-m)/sizee +1)/2;
}
int High(int rank, int sizee, int n, int m){
    return Low(rank+1,sizee,n,m)-1;
}
int Lenght(int rank, int sizee, int n, int m){
    return Low(rank+1,sizee,n,m)-Low((rank), sizee, n,m);
}
