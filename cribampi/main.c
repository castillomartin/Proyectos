#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <assert.h>
//
//int main(int args, char* argvs){
//    int rank = 0, numOfProcess = 0,a,tag=0,b,i;
//    MPI_Status status;
//    MPI_Init(&args,&argvs);
//    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//    MPI_Comm_size(MPI_COMM_WORLD, &numOfProcess);
//
//
//    //printf("Hello World from process rank(number): %d from %d \n", rank, numOfProcess);
//
//            if(rank == 0){
//            b=10;
//            a=5;
//            for(i=1;i<numOfProcess;i++){
//			MPI_Send(&a,1,MPI_INT,i,0,MPI_COMM_WORLD);
//			MPI_Send(&b,1,MPI_INT,i,1,MPI_COMM_WORLD);}
//            printf("I'm 0 and send to 1 : %d and %d: \n",b,a);
//			}
//
//			if(rank != 0){
//			MPI_Recv(&b,1,MPI_INT,0,0,MPI_COMM_WORLD,&status);
//			MPI_Recv(&a,1,MPI_INT,0,1,MPI_COMM_WORLD,&status);
//            printf("I'm 1 and recive from 0 : %d",b);
//            printf("and : %d\n",a);
//            }
//
//    MPI_Finalize();
//    return 0;
//}
//


int main(int args, char* argvs){

//
    double start_time; //hold start time
    double end_time; // hold end time
    int a[150000],i,sum=0,psum=0,xsum=0;


    int rank = 0, numOfProcess = 0, tag=0, r;

    MPI_Status status;
    MPI_Init(&args,&argvs);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numOfProcess);

    if(rank == 0)
    for(i=0;i<150000;i++){
        a[i]=i+1;
    }


    start_time = MPI_Wtime();
    int ini2 = rank*150000/numOfProcess;
    for(i=ini2 ;i< ini2+150000/numOfProcess;i++)
                psum+=a[i];

    MPI_Reduce(&psum,&sum,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);

    end_time = MPI_Wtime();
        if(rank == 0)
        printf("\nRunning Time = %f\n\n", end_time - start_time);
    MPI_Finalize();



    //2da variante
//    int a[150000],i,sum=0,psum,xsum=0;
//    for(i=0;i<150000;i++){
//        a[i]=i+1;
//    }
//
//    int rank = 0, numOfProcess = 0, tag=0, r;
//
//    MPI_Status status;
//    MPI_Init(&args,&argvs);
//
//    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//    MPI_Comm_size(MPI_COMM_WORLD, &numOfProcess);
//
//
//     start_time; //hold start time
//
//    if(rank == numOfProcess-1){
//            int ini = 150000-(150000/numOfProcess);
//        for(i=ini;i<150000;i++)
//            sum+=a[i];
//
//        for(i=0;i<numOfProcess-1;i++){
//            MPI_Recv(&psum,1,MPI_INT,i,tag,MPI_COMM_WORLD,&status);
//            sum+=psum;
//        }
//
//    }
//
//
//    if(rank!=numOfProcess-1){
//            xsum =0;
//            int ini2 = rank*150000/numOfProcess;
//            for(i=ini2 ;i< ini2+150000/numOfProcess;i++)
//                xsum+=a[i];
//
//            MPI_Send(&xsum,1,MPI_INT,numOfProcess-1,tag,MPI_COMM_WORLD);
//
//    }
//
//
//    end_time = MPI_Wtime();
//    if(rank == 0)
//        printf("\nRunning Time = %f\n\n", end_time - start_time);
//
//    MPI_Finalize();
//
    return 0;
}



//
//
//int main(int argc, char* argv[]) {
//    int         mi_rango;      /* rango del proceso    */
//    int         p;             /* numero de procesos   */
//    int         fuente;        /* rango del que envia  */
//    int         dest;          /* rango del que recibe */
//    int         tag = 0;       /* etiqueta del mensaje */
//    char        mensaje[200000];  /* mensaje  */
//    double start_time; //hold start time
//    double end_time; // hold end time
//    MPI_Status  estado;   /* devuelve estado al recibir*/
//
//    /* Comienza las llamadas a MPI */
//    MPI_Init(&argc, &argv);
//
//    /* Averiguamos el rango de nuestro proceso */
//    MPI_Comm_rank(MPI_COMM_WORLD, &mi_rango);
//
//    /* Averiguamos el número de procesos que estan
//     * ejecutando nuestro porgrama
//     */
//    MPI_Comm_size(MPI_COMM_WORLD, &p);
//
//        start_time = MPI_Wtime();
//    if (mi_rango == 0){
//        sprintf(mensaje, "SaludoSaludos del proceso Saludludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso s del proceso Saludos del proceso  Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso aludoSaludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso s del proceso Saludos del proceso  Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso Saludos del proceso  %d!",
//            mi_rango);
//        MPI_Send(mensaje, strlen(mensaje)+1, MPI_CHAR,1, tag, MPI_COMM_WORLD);
//    } else { /* mi_rango == 1 */
//            MPI_Recv(mensaje, 50000, MPI_CHAR, 0,tag, MPI_COMM_WORLD, &estado);
//            MPI_Send(mensaje, strlen(mensaje)+1, MPI_CHAR,0, tag, MPI_COMM_WORLD);
//    }
//    if(mi_rango == 0){
//        MPI_Recv(mensaje, 50000, MPI_CHAR, 1,tag, MPI_COMM_WORLD, &estado);
//    }
//
//        end_time = MPI_Wtime();
//        if(mi_rango == 0)
//        printf("\nRunning Time = %f\n\n", (end_time - start_time) /2);
//
//     MPI_Finalize();
//     return 0;
//} /* main */





		//
//#define N 3
//#include <stdio.h>
//#include <math.h>
//#include <sys/time.h>
//#include <stdlib.h>
//#include <stddef.h>
//#include "mpi.h"
//
//
//void print_results(char *prompt, double a[N][N]);
//
//int main(int argc, char *argv[])
//{
//    int i, j, k, rank, size, tag = 99;
//    double sum = 0.0;
//    double a[N][N];
//    double b[N][N];
//    double c[N][N];
//    double aa[N],cc[N];
//
//    MPI_Init(&argc, &argv);
//    MPI_Comm_size(MPI_COMM_WORLD, &size);
//    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//
//    double start = MPI_Wtime();
//
//    if(rank == 0){
//    for(i=0;i<N;i++)
//        for(j=0;j<N;j++)
//        {
//            a[i][j] = (double)((double)((double)N-2)/2);
//            b[i][j] = (double)((double)((double)N-2)/2);
//            if(i!=j){
//            a[i][j] = (double)(-2/(double)N);
//            b[i][j] = (double)(-2/(double)N);
//            }
//        }
//    }
//    //scatter rows of first matrix to different processes
//    MPI_Scatter(a, N*N/size, MPI_DOUBLE, aa, N*N/size, MPI_DOUBLE,0,MPI_COMM_WORLD);
//
//    //broadcast second matrix to all processes
//    MPI_Bcast(b, N*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//
//    MPI_Barrier(MPI_COMM_WORLD);
//
//          //perform vector multiplication by all processes
//          for (i = 0; i < N; i++)
//            {
//                    for (j = 0; j < N; j++)
//                    {
//                            sum = sum + aa[j] * b[j][i];
//                    }
//                    printf(" (%d) %.2lf",rank, sum);
//                    cc[i] = sum;
//                    sum = 0;
//            }
//            printf("\n");
//
//    MPI_Gather(cc, N*N/size, MPI_DOUBLE, c, N*N/size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//
//    MPI_Barrier(MPI_COMM_WORLD);
//
//    double finish = MPI_Wtime();
//    if(rank==0)
//    printf("Done in %f seconds.\n", finish - start);
//
//    MPI_Finalize();
//
//    if(rank==0)
//    print_results("C = ", c);
//}
//
//void print_results(char *prompt, double a[N][N])
//{
//    int i, j;
//
//    printf ("\n\n%s\n", prompt);
//    for (i = 0; i < N; i++) {
//            for (j = 0; j < N; j++) {
//                    printf(" %.2lf", a[i][j]);
//            }
//            printf ("\n");
//    }
//    printf ("\n\n");
//}
//
//
//
//#define NUM_ROWS_A 100 //rows of input [A]
//#define N 100 //rows of input [A]
//#define NUM_COLUMNS_A 100 //columns of input [A]
//#define NUM_ROWS_B 100 //rows of input [B]
//#define NUM_COLUMNS_B 100 //columns of input [B]
//#define MASTER_TO_SLAVE_TAG 1 //tag for messages sent from master to slaves
//#define SLAVE_TO_MASTER_TAG 4 //tag for messages sent from slaves to master
//
//void makeAB(); //makes the [A] and [B] matrixes
//void printArray(); //print the content of output matrix [C];
//int rank; //process rank
//int size; //number of processes
//int i, j, k; //helper variables
//double mat_a[NUM_ROWS_A][NUM_COLUMNS_A]; //declare input [A]
//double mat_b[NUM_ROWS_B][NUM_COLUMNS_B]; //declare input [B]
//double mat_result[NUM_ROWS_A][NUM_COLUMNS_B]; //declare output [C]
//double start_time; //hold start time
//double end_time; // hold end time
//int low_bound; //low bound of the number of rows of [A] allocated to a slave
//int upper_bound; //upper bound of the number of rows of [A] allocated to a slave
//int portion; //portion of the number of rows of [A] allocated to a slave
//MPI_Status status; // store status of a MPI_Recv
//MPI_Request request; //capture request of a MPI_Isend
//int main(int argc, char *argv[])
//{
//    MPI_Init(&argc, &argv); //initialize MPI operations
//    MPI_Comm_rank(MPI_COMM_WORLD, &rank); //get the rank
//    MPI_Comm_size(MPI_COMM_WORLD, &size); //get number of processes
//    /* master initializes work*/
//    if (rank == 0) {
//        makeAB();
//        start_time = MPI_Wtime();
//        for (i = 1; i < size; i++) {//for each slave other than the master
//            portion = (NUM_ROWS_A / (size - 1)); // calculate portion without master
//            low_bound = (i - 1) * portion;
//            if (((i + 1) == size) && ((NUM_ROWS_A % (size - 1)) != 0)) {//if rows of [A] cannot be equally divided among slaves
//                upper_bound = NUM_ROWS_A; //last slave gets all the remaining rows
//            } else {
//                upper_bound = low_bound + portion; //rows of [A] are equally divisable among slaves
//            }
//            //send the low bound first without blocking, to the intended slave
//            MPI_Isend(&low_bound, 1, MPI_INT, i, MASTER_TO_SLAVE_TAG, MPI_COMM_WORLD, &request);
//            //next send the upper bound without blocking, to the intended slave
//            MPI_Isend(&upper_bound, 1, MPI_INT, i, MASTER_TO_SLAVE_TAG + 1, MPI_COMM_WORLD, &request);
//            //finally send the allocated row portion of [A] without blocking, to the intended slave
//            MPI_Isend(&mat_a[low_bound][0], (upper_bound - low_bound) * NUM_COLUMNS_A, MPI_DOUBLE, i, MASTER_TO_SLAVE_TAG + 2, MPI_COMM_WORLD, &request);
//        }
//    }
//    //broadcast [B] to all the slaves
//    MPI_Bcast(&mat_b, NUM_ROWS_B*NUM_COLUMNS_B, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//    /* work done by slaves*/
//    if (rank > 0) {
//        //receive low bound from the master
//        MPI_Recv(&low_bound, 1, MPI_INT, 0, MASTER_TO_SLAVE_TAG, MPI_COMM_WORLD, &status);
//        //next receive upper bound from the master
//        MPI_Recv(&upper_bound, 1, MPI_INT, 0, MASTER_TO_SLAVE_TAG + 1, MPI_COMM_WORLD, &status);
//        //finally receive row portion of [A] to be processed from the master
//        MPI_Recv(&mat_a[low_bound][0], (upper_bound - low_bound) * NUM_COLUMNS_A, MPI_DOUBLE, 0, MASTER_TO_SLAVE_TAG + 2, MPI_COMM_WORLD, &status);
//        for (i = low_bound; i < upper_bound; i++) {//iterate through a given set of rows of [A]
//            for (j = 0; j < NUM_COLUMNS_B; j++) {//iterate through columns of [B]
//                for (k = 0; k < NUM_ROWS_B; k++) {//iterate through rows of [B]
//                    mat_result[i][j] += (mat_a[i][k] * mat_b[k][j]);
//                }
//            }
//        }
//        //send back the low bound first without blocking, to the master
//        MPI_Isend(&low_bound, 1, MPI_INT, 0, SLAVE_TO_MASTER_TAG, MPI_COMM_WORLD, &request);
//        //send the upper bound next without blocking, to the master
//        MPI_Isend(&upper_bound, 1, MPI_INT, 0, SLAVE_TO_MASTER_TAG + 1, MPI_COMM_WORLD, &request);
//        //finally send the processed portion of data without blocking, to the master
//        MPI_Isend(&mat_result[low_bound][0], (upper_bound - low_bound) * NUM_COLUMNS_B, MPI_DOUBLE, 0, SLAVE_TO_MASTER_TAG + 2, MPI_COMM_WORLD, &request);
//    }
//    /* master gathers processed work*/
//    if (rank == 0) {
//        for (i = 1; i < size; i++) {// untill all slaves have handed back the processed data
//            //receive low bound from a slave
//            MPI_Recv(&low_bound, 1, MPI_INT, i, SLAVE_TO_MASTER_TAG, MPI_COMM_WORLD, &status);
//            //receive upper bound from a slave
//            MPI_Recv(&upper_bound, 1, MPI_INT, i, SLAVE_TO_MASTER_TAG + 1, MPI_COMM_WORLD, &status);
//            //receive processed data from a slave
//            MPI_Recv(&mat_result[low_bound][0], (upper_bound - low_bound) * NUM_COLUMNS_B, MPI_DOUBLE, i, SLAVE_TO_MASTER_TAG + 2, MPI_COMM_WORLD, &status);
//        }
//        end_time = MPI_Wtime();
//        printf("\nRunning Time = %f\n\n", end_time - start_time);
//        printArray();
//    }
//    MPI_Finalize(); //finalize MPI operations
//    return 0;
//}
//void makeAB()
//{
//    for (i = 0; i < NUM_ROWS_A; i++) {
//        for (j = 0; j < NUM_COLUMNS_A; j++) {
//            mat_a[i][j] = (double)((double)N - 2) / (double)N;
//            if(i!=j)
//                mat_a[i][j] = (double)((-2) / (double)N);
//        }
//    }
//    for (i = 0; i < NUM_ROWS_B; i++) {
//        for (j = 0; j < NUM_COLUMNS_B; j++) {
//            mat_b[i][j] = (double)((double)(N) - 2) / (double)(N);
//                if(i!=j)
//                mat_b[i][j] = (double)((-2) / (double)N);
//
//        }
//    }
//}
//void printArray()
//{
//    for (i = 0; i < NUM_ROWS_A; i++) {
//        printf("\n");
//        for (j = 0; j < NUM_COLUMNS_A; j++)
//            printf("%8.2f  ", mat_a[i][j]);
//    }
//    printf("\n\n\n");
//    for (i = 0; i < NUM_ROWS_B; i++) {
//        printf("\n");
//        for (j = 0; j < NUM_COLUMNS_B; j++)
//            printf("%8.2f  ", mat_b[i][j]);
//    }
//    printf("\n\n\n");
//    for (i = 0; i < NUM_ROWS_A; i++) {
//        printf("\n");
//        for (j = 0; j < NUM_COLUMNS_B; j++)
//            printf("%8.2f  ", mat_result[i][j]);
//    }
//    printf("\n\n");
//}
//
//

//
//
//
//int main(int args, char* argvs){
//
//    char host[20] ,rec[20];
//    strcpy(host,"Computadora1");
//    MPI_Status status;
//    int rank = 0, numOfProcess = 0, tag=0, i;
//    MPI_Init(&args,&argvs);
//    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//    MPI_Comm_size(MPI_COMM_WORLD, &numOfProcess);
//    char c[20],d[20];
//    if(rank == 0){
//        for(i=1;i<numOfProcess;i++){
//            strcpy(c,"Hola");
//            sprintf(d,"%d",i);
//            strcat(c,d);
//            MPI_Send(c, 20, MPI_CHAR,i,tag,MPI_COMM_WORLD);
//            printf("Hi, nodo %d, de %d totales. Maquina %s y envio %s a Computadora %i\n",rank,numOfProcess,host,c,i);
//        }
//    }
//    else{
//        MPI_Recv(rec,20,MPI_CHAR,0,tag,MPI_COMM_WORLD,&status);
//        printf("Hi, nodo %d de %d totales. Maquina %s, y recibi %s\n", rank, numOfProcess,host,rec);
//    }
//
//    MPI_Finalize();
//    return 0;
//}


//
//
//
//int main(int args, char* argvs){
//
//
//    char host[20] ,rec[20];
//    double tiempo_inicial, tiempo_final;
//
//    strcpy(host,"Computadora1");
//    MPI_Status status;
//    int rank = 0, numOfProcess = 0, tag=0, i, r;
//    MPI_Init(&args,&argvs);
//
//    tiempo_inicial = MPI_Wtime();
//
//
//    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//    MPI_Comm_size(MPI_COMM_WORLD, &numOfProcess);
//    char c[20],d[20],e[20];
//        int n;
//    if(rank == 0){
//        n=200;
//    }
//    MPI_Bcast(&n,1,MPI_INT,0,MPI_COMM_WORLD);
//    if(rank!=0){
//    printf("proceso # %d, numero %d\n",rank,n);
//    }
//
//    tiempo_final = MPI_Wtime();
//
//
//    MPI_Finalize();
//
//    printf("Inicio: %lf, fin: %lf\n",tiempo_inicial,tiempo_final);
//
//    return 0;
//}


//
//
//
//int main(int args, char* argvs){
//
//    int a[100],i,sum=0,psum,xsum=0;
//
//
//    int rank = 0, numOfProcess = 0, tag=0, r;
//
//    MPI_Status status;
//    MPI_Init(&args,&argvs);
//
//    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//    MPI_Comm_size(MPI_COMM_WORLD, &numOfProcess);
//
//    if(rank == numOfProcess-1){
//
//        for(i=0;i<100;i++){
//        a[i]=i+1;
//        }
//
//        int ini = 100-(100/numOfProcess);
//        for(i=ini;i<100;i++)
//            sum+=a[i];
//
//        for(i=0;i<numOfProcess-1;i++){
//            MPI_Send(&a[i*100/numOfProcess],100/numOfProcess,MPI_INT,i,tag,MPI_COMM_WORLD);
//        }
//        for(i=0;i<numOfProcess-1;i++){
//            MPI_Recv(&psum,1,MPI_INT,i,tag,MPI_COMM_WORLD,&status);
//            sum+=psum;
//        }
//
//    }
//
//
//    if(rank!=numOfProcess-1){
//            MPI_Recv(&a,100/numOfProcess,MPI_INT,numOfProcess-1,tag,MPI_COMM_WORLD,&status);
//            xsum =0;
//            for(i=0;i<100/numOfProcess;i++)
//                xsum+=a[i];
//            MPI_Send(&xsum,1,MPI_INT,numOfProcess-1,tag,MPI_COMM_WORLD);
//
//    }
//    if(rank==numOfProcess-1)
//        printf("suma %d",sum);
//
//
//    MPI_Finalize();
//
//
//
//    return 0;
//}
//
//
//
//
//
//
//int main(int args, char* argvs){
//
//    int a[100],i,sum=0,psum,xsum=0;
//    for(i=0;i<100;i++){
//        a[i]=i+1;
//    }
//
//    int rank = 0, numOfProcess = 0, tag=0, r;
//
//    MPI_Status status;
//    MPI_Init(&args,&argvs);
//
//    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//    MPI_Comm_size(MPI_COMM_WORLD, &numOfProcess);
//
//    if(rank == numOfProcess-1){
//            int ini = 100-(100/numOfProcess);
//        for(i=ini;i<100;i++)
//            sum+=a[i];
//
//        for(i=0;i<numOfProcess-1;i++){
//            MPI_Recv(&psum,1,MPI_INT,i,tag,MPI_COMM_WORLD,&status);
//            sum+=psum;
//        }
//
//    }
//
//
//    if(rank!=numOfProcess-1){
//            xsum =0;
//            int ini2 = rank*100/numOfProcess;
//            for(i=ini2 ;i< ini2+100/numOfProcess;i++)
//                xsum+=a[i];
//
//            MPI_Send(&xsum,1,MPI_INT,numOfProcess-1,tag,MPI_COMM_WORLD);
//
//    }
//
//    printf("suma %d",sum);
//
//
//    MPI_Finalize();
//
//
//    return 0;
//}


//
//
//int main(int args, char* argvs){
//
//    int a[100],i,sum=0,psum=0,xsum=0;
//    for(i=0;i<100;i++){
//        a[i]=i+1;
//    }
//
//    int rank = 0, numOfProcess = 0, tag=0, r;
//
//    MPI_Status status;
//    MPI_Init(&args,&argvs);
//
//    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//    MPI_Comm_size(MPI_COMM_WORLD, &numOfProcess);
//
//
//        int ini2 = rank*100/numOfProcess;
//        for(i=ini2 ;i< ini2+100/numOfProcess;i++)
//                psum+=a[i];
//
//    MPI_Reduce(&psum,&sum,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
//
//
//    printf("suma %d",sum);
//
//    MPI_Finalize();
//
//
//    return 0;
//}
//
//

//
//int main(int args, char* argvs){
//
//
//    int ** matriz1, **matriz2, *cmatriz1 , *cmatriz2,fila1=4,columna1=4,fila2 =4, columna2 = 4;
//    int * filaproceso;
//    int rank = 0, numOfProcess = 0, tag=0, r,i,j,k,xx;
//    int  matriz3[fila1][columna2];
//    int  matriz4[fila1][columna2];
//
//    matriz1 = (int**)malloc(fila1*sizeof(int*));
//    matriz2 = (int**)malloc(fila2*sizeof(int*));
//    cmatriz1 = (int*)malloc(fila1*columna1*sizeof(int));
//    cmatriz2 = (int*)malloc(fila2*columna2*sizeof(int));
//
//    filaproceso = (int*)malloc(columna2*sizeof(int));
//
//
//    MPI_Status status;
//    MPI_Init(&args,&argvs);
//
//    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//    MPI_Comm_size(MPI_COMM_WORLD, &numOfProcess);
//
//    if(rank == 0){
//
//        for(i=0;i<fila1;i++){
//            matriz1[i] = cmatriz1+ i*columna1;
//        }
//        for(i=0;i<fila2;i++){
//            matriz2[i] = cmatriz2 + i*columna2;
//        }
//
//        for(i=0;i<fila1;i++)
//            for(j=0;j<columna1;j++){
//                matriz1[i][j] =  rand() % 5;
//
//                printf("%d ",matriz1[i][j] );
//        }
//        for(i=0;i<fila2;i++)
//            for(j=0;j<columna2;j++){
//                matriz2[i][j] =  rand() % 4;
//        }
//
//        for(i=0;i<fila1;i++)
//            for(j=0;j<columna2;j++){
//                matriz3[i][j] =  0;
//                matriz4[i][j] =  0;
//        }
//
//
//        for(i=0;i<fila1;i++)
//            for(j=0;j<columna2;j++)
//                for(k=0;k<columna2;k++){
//                    matriz4[i][j] += matriz1[i][k] * matriz2[k][j];
//                }
//
//    }
//
//
//        MPI_Bcast(matriz1,columna1*fila1,MPI_INT,0,MPI_COMM_WORLD);
//
//
//   if(rank!=0){
//
//        for(i=0;i<columna2;i++){
//            filaproceso[i] = 0;
//            for(j=0;j<fila1;j++){
//                //filaproceso[i] += matriz1[rank][j];
//            }
//        //        printf("%d - ", matriz1[0][0] );
//        }
//
//        // MPI_Send(&filaproceso,columna2,MPI_INT,0,tag,MPI_COMM_WORLD);
//    }
//
////
////    if(rank ==0){
////        for(i = 1; i<numOfProcess; i++){
////             MPI_Recv(&filaproceso,columna2,MPI_INT,i,tag,MPI_COMM_WORLD,&status);
////             for(j=0;j<columna2;j++){
////                    printf("%d  ",filaproceso[j]);
////                }
////        }
////    }
//
//
//    MPI_Finalize();
//
////
////    for(i=0;i<fila1;i++){
////        for(j=0;j<columna2;j++){
////            printf("%d  ",matriz4[i][j]);
////        }
////        printf("\n");
////    }
//    return 0;
//}



//
//int main(int args, char* argvs){
//
//
//    clock_t start_t, end_t, total_t;
//
//    int ** matriz1, **matriz2, *cmatriz1 , *cmatriz2,fila1=5,columna1=5,fila2 =5, columna2 = 5;
//    int i,j,k;
//    int matriz3[fila1][columna2];
//
//    matriz1 = (int**)malloc(fila1*sizeof(int*));
//    matriz2 = (int**)malloc(fila2*sizeof(int*));
//    cmatriz1 = (int*)malloc(fila1*columna1*sizeof(int));
//    cmatriz2 = (int*)malloc(fila2*columna2*sizeof(int));
//
//        for(i=0;i<fila1;i++){
//            matriz1[i] = cmatriz1+ i*columna1;
//        }
//        for(i=0;i<fila2;i++){
//            matriz2[i] = cmatriz2 + i*columna2;
//        }
//
//        for(i=0;i<fila1;i++)
//            for(j=0;j<columna1;j++){
//                matriz1[i][j] =  rand() % 5;
//
//        }
//        for(i=0;i<fila2;i++)
//            for(j=0;j<columna2;j++){
//                matriz2[i][j] =  rand() % 4;
//        }
//
//
//    start_t = clock();
//    printf("Starting of the program, start_t = %ld\n", start_t);
//
//    printf("Going to scan a big loop, start_t = %ld\n", start_t);
//
//        for(i=0;i<fila1;i++){
//            for(j=0;j<columna2;j++){
//                    matriz3[i][j]=0;
//                for(k=0;k<columna2;k++){
//                    matriz3[i][j] += matriz1[i][k] * matriz2[k][j];
//                }
//            printf("%d ",matriz3[i][j]);
//            }
//            printf("\n");
//        }
//
//    end_t = clock();
//
//
//   printf("End of the big loop, end_t = %ld\n", end_t);
//
//   total_t = (double)(end_t - start_t) / CLOCKS_PER_SEC;
//   printf("Total time taken by CPU: %f\n", total_t );
//   printf("Exiting of the program...\n");
//
//
//return 0;
//}
//int main(int args, char* argvs){
//
//    int a[100],i,sum=0,psum=0,xsum=0;
//    for(i=0;i<100;i++){
//        a[i]=i+1;
//    }
//
//    int rank = 0, numOfProcess = 0, tag=0, r;
//
//    MPI_Status status;
//    MPI_Init(&args,&argvs);
//
//    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//    MPI_Comm_size(MPI_COMM_WORLD, &numOfProcess);
//
//
//        int ini2 = rank*100/numOfProcess;
//
//        printf("Nodo # %d: \n",rank );
//        for(i=ini2 ;i< ini2+100/numOfProcess;i++){
//                psum+=a[i];
//                printf("%d - ", a[i]);
//        }
//        printf("\n");
//
//    MPI_Reduce(&psum,&sum,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
//
//
//    printf("suma %d",sum);
//
//    MPI_Finalize();
//
//
//    return 0;
//}

////
//int main(int args, char* argvs){
//
//    int i;
//    float * a,ssum=0.0,psum,xsum=0.0;
//    float *b;
//    float * c;
//    int rank = 0, numOfProcess = 0, tag=0;
//    float ksum = 0.0;
//
//    MPI_Status status;
//    MPI_Init(&args,&argvs);
//
//    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//    MPI_Comm_size(MPI_COMM_WORLD, &numOfProcess);
//
//    if(rank == 0){
//    a = (float*)malloc(sizeof(float)*100);
//    for(i=0;i<100;i++)
//        a[i]= (float)(i+1);
//    }
//
//    b = (float*)malloc(sizeof(float)*100/numOfProcess);
//    MPI_Scatter(a,100/numOfProcess,MPI_FLOAT,b,100/numOfProcess,MPI_FLOAT,0,MPI_COMM_WORLD);
//
//
//    printf("Nodo # %d: \n",rank );
//    for(i=0;i<100/numOfProcess;i++){
//        ksum+=b[i];
//        printf("%.1lf - ", b[i]);
//    }
//
//    //MPI_Reduce(&ksum,&ssum,1,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);
//
//    if(rank == 0){
//    c = (float*)malloc(sizeof(float)*numOfProcess);
//    }
//
//
//    MPI_Gather(&ksum,1,MPI_FLOAT,c,1,MPI_FLOAT,0,MPI_COMM_WORLD);
//
//    if(rank == 0){
//
//    for(i=0;i<numOfProcess;i++){
//        ssum+= c[i];
//    printf("  ",c[i]);
//    }
//
//    }
//
//
//    //printf("%.1lf - ", ssum);
//
//    MPI_Finalize();
//
//
//    return 0;
//}


// Creates an array of random numbers. Each number has a value from 0 - 1


