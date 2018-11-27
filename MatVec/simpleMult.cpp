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
        double c[AROW];
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

		
		double sum = 0.0;
		for(int i=0;i<AROW;i++){
			sum = 0.0;
			for(int j=0;j<ACOL;j++){
				sum+= a[i][j] * b[j];
			}
			c[i] = sum;
		}
   
		FILE * f = fopen(argv[3],"a+b");
            fprintf(f,"%d ", AROW);
            fprintf(f,"%d ", ACOL);
        double c[AROW];
        for (int i=0;i<AROW;i++)
        {
            fprintf(f,"%.2f ", c[i]);
        }
        fclose(f);
		
	}
    return 0;
}


int proc_map(int i, int size)
{
    size = size - 1;
    int r = (int) ceil( (double)AROW / (double)size);
    int proc = i / r;
    return proc + 1;
}
