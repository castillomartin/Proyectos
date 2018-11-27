#include "stdio.h"
#include "stdlib.h"
#include "time.h"
#include "math.h"
#include "mpi.h"
#include "string.h"


int AROW,ACOL,i,j;
/* Process mapping function */
int proc_map(int i, int size);

int main(int argc, char** argv)
{
    int size, rank;
    MPI_Status Stat;
    double  elapsed_time,tm,maxtime,s,maxi;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
    {

        MPI_File cFile,cFile2;
        MPI_Status  status,status1, status2;


        //char *a1,*b1;
        //strcpy(a1,argv[1]);
        //strcpy(b1,argv[2]);
        int r1 = MPI_File_open( MPI_COMM_SELF, argv[1], MPI_MODE_RDONLY, MPI_INFO_NULL, &cFile);
        float dd;

        MPI_File_read(cFile,&AROW,1,MPI_INT,&status2);
        MPI_File_read(cFile,&ACOL,1,MPI_INT,&status1);

        double a[AROW][ACOL];
        double b[ACOL];

        /* Generating Random Values for A & B Array*/
        //srand(time(NULL));
        for (int i=0;i<AROW;i++)
        {
            for (int j=0;j<ACOL;j++)
            {
                MPI_File_read(cFile,&dd,1,MPI_DOUBLE,&status);
                a[i][j] = dd;
                //printf("%d  ",a[i][j]);
            }
        }
        MPI_File_close( &cFile );

        int r2 = MPI_File_open( MPI_COMM_SELF, argv[2], MPI_MODE_RDONLY, MPI_INFO_NULL, &cFile2);

        for (int j=0;j<ACOL;j++)
        {
            MPI_File_read(cFile2,&dd,1,MPI_DOUBLE,&status);
            b[j] = dd;
                //printf("%d  ",a[i][j]);
        }
        MPI_File_close( &cFile2 );


        /* Printing the Matrix*/

//        printf("Matrix A :\n");
//        for (int i=0;i<AROW;i++)
//        {
//            for (int j=0;j<ACOL;j++)
//            {
//                printf("%.3lf ", a[i][j]);
//            }
//            printf("\n");
//        }
//        printf("nMatrix B :\n");
//        for (int i=0;i<ACOL;i++)
//        {
//            printf("%.3lf ", b[i]);
//        }
//        printf("\n");

        elapsed_time = -MPI_Wtime();

        /* (1) Sending B Values to other processes */
        for (int j=1;j<size;j++)
        {
            MPI_Send(b, ACOL, MPI_DOUBLE, j, 99, MPI_COMM_WORLD);
        }

        /* (2) Sending Required A Values to specific process */
        for (int i=0;i<AROW;i++)
        {
            int processor = proc_map(i, size);
            MPI_Send(a[i], ACOL, MPI_DOUBLE, processor, (100*(i+1)), MPI_COMM_WORLD);
        }



    }

    MPI_Bcast(&AROW, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ACOL, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    if(rank != 0)
    {
        double b[ACOL];
//
//        /* (1) Each process get B Values from Master */
        MPI_Recv(b, ACOL, MPI_DOUBLE, 0, 99, MPI_COMM_WORLD, &Stat);
//
//        /* (2) Get Required A Values from Master then Compute the result */
        for (int i=0;i<AROW;i++)
        {
            int processor = proc_map(i, size);
            if (rank == processor)
            {
                double buffer[ACOL];
                MPI_Recv(buffer, ACOL, MPI_DOUBLE, 0, (100*(i+1)), MPI_COMM_WORLD, &Stat);
                double sum = 0.0;
                for (int j=0;j<ACOL;j++)
                {
                    sum = sum + (buffer[j] * b[j] );
                }
                MPI_Send(&sum, 1, MPI_DOUBLE, 0, i, MPI_COMM_WORLD);
                //printf("rank %d i %d\n",rank,i);
            }
        }
    }

    if(rank == 0){
         /* (3) Gathering the result from other processes*/

        double c[AROW];
        for (int i=0;i<AROW;i++)
        {
            int source_process = proc_map(i, size);
            MPI_Recv(&c[i], 1, MPI_DOUBLE, source_process, i, MPI_COMM_WORLD, &Stat);
            //printf("%.2f ", c[i]);
            //printf("rank %d i %d\n",source_process,i);
        }

    }


    tm = MPI_Wtime();
    elapsed_time += tm;
	if(maxtime > tm)
		maxtime = tm;

    MPI_Reduce(&elapsed_time, &s, 1, MPI_DOUBLE,MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&maxtime, &maxi, 1, MPI_DOUBLE,MPI_MAX, 0, MPI_COMM_WORLD);

    if(rank == 0){
        printf("Total elapsed time: %10.6fs\n",s/100000);
        printf("Max time: %10.6fs\n",maxi/100000);
    }

    MPI_Finalize();

      FILE * f = fopen(argv[3],"a+b");
            fprintf(f,"%d ", AROW);
            fprintf(f,"%d ", ACOL);
        double c[AROW];
        for (int i=0;i<AROW;i++)
        {
            fprintf(f,"%.2f ", c[i]);
        }
        fclose(f);


    return 0;
}


int proc_map(int i, int size)
{
    size = size - 1;
    int r = (int) ceil( (double)AROW / (double)size);
    int proc = i / r;
    return proc + 1;
}
