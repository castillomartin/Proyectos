#include <pthread.h>
#include <semaphore.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <ctime>
#include <time.h>
#include <unistd.h>

#define BILLION  1000000000L;

int NUMTHRDS = 4;
pthread_t * callThd;
pthread_mutex_t mutexsum;
pthread_mutex_t mutetime;
int sum;
float sum1 = 0.0,maxi=0.0, maxi2 = 0.0;
long * Comp;
int VECLEN = 50;
int IECLEN = 10;
long long int start1,end1;
struct timespec starttime, stoptime;
double accum;

void *setTotalPrimes(void *arg) {
    /* Define and use local variables for convenience */

    int i, start, end, len ;
    long offset;
    int mysum = 0;

    offset = (long)arg;

    char file[] = "rankX.bin";
	file[4] = offset+'0';
    FILE * f = fopen(file,"a+b");

    //printf ("I am thread number  %ld \n", offset);
    len = (VECLEN-IECLEN)/NUMTHRDS;
    if(double(VECLEN-IECLEN)/NUMTHRDS != (VECLEN-IECLEN)/NUMTHRDS)
        len++;
    start = offset*len + IECLEN;


    /**
     VECLEN is not divisible by 3.
     */
    if (NUMTHRDS == 3 && offset == 2) {
        end   = start + len + 2;
    }else {
        end   = start + len;
    }

    if(end > VECLEN)
        end = VECLEN;
     if(offset == NUMTHRDS-1)
        end++;


    /**
     Initialize the portion of Comp
     */
    for (i = start; i < end; i++) {
        if (i < 3){
            if (i < 2) {
                Comp[i] = 1;
            }else{
                Comp[i] = 0;
            }
        }else{
            if (i % 2 == 0 || i % 3 == 0 || i % 5 == 0 || i % 7 == 0|| i % 11 == 0){
                Comp[i] = 1;
            }
        }
    }
	
	
    if( clock_gettime( CLOCK_REALTIME, &starttime) == -1 ) {
      perror( "clock gettime" );
      exit( EXIT_FAILURE );
    }
	
    // start1 = clock();

    /**
     Sieve of Eratoshenes.
     */
    for (i = start; i < end; i++) {
        if (i > 2) {
            if (Comp[i] == 0) {
                for (int x = 3 ; x < i; x ++){
                    if ( x*x > i)
                        break;
                    if ( x % 2 != 0 && i % x == 0 ){
                        Comp[i] = 1;
                        break;
                    }
                }
            }
        }
    }

    for (i = start; i < end; i++) {
        if (i > 1 && Comp[i] == 0) {
            mysum++;
            fprintf(f,"%d\n",i);
        }
    }

	//end1 = clock();
	
    if( clock_gettime( CLOCK_REALTIME, &stoptime) == -1 ) {
      perror( "clock gettime" );
      exit( EXIT_FAILURE );
    }


    pthread_mutex_lock (&mutetime);
	
	accum = (double)( stoptime.tv_sec - starttime.tv_sec )
          + (double)( stoptime.tv_nsec - starttime.tv_nsec )
            / (double)BILLION;
			
     sum1 = sum1 + accum;
	 
	if(maxi2 < accum)
        maxi2 = accum;
    pthread_mutex_unlock (&mutetime);

    fclose(f);

    pthread_mutex_lock (&mutexsum);
	
    sum += mysum;
    pthread_mutex_unlock (&mutexsum);


    pthread_exit((void*) 0);
}


int main (int argc, char *argv[])
{

     if (argc != 4)    {
        printf("Range need: %s <n> <m> <thread>\n", argv[0]);
        exit(1);
    }

    IECLEN = atoi(argv[1]);
    VECLEN = atoi(argv[2]);
    NUMTHRDS = atoi(argv[3]);

    long i;
    void *status;
    pthread_attr_t attr;
    int input = 0;
    int isLoop = 1;
    Comp = new long[VECLEN];
	callThd = new pthread_t[NUMTHRDS];
    sum = 0;

    pthread_mutex_init(&mutexsum, NULL);
    pthread_mutex_init(&mutetime, NULL);

    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);


	for(i=0; i<NUMTHRDS; i++)
    {
           pthread_create(&callThd[i], &attr, setTotalPrimes, (void *)i);
	}

 	pthread_attr_destroy(&attr);

    /* Wait on the other threads */
	for(i=0; i<NUMTHRDS; i++)
    {
        pthread_join(callThd[i], &status);
	}

    char file[] = "plot";
    FILE * g = fopen(file,"a+b");
    fprintf(g,"%d %f %f\n",NUMTHRDS,sum1,maxi2);
    fclose(g);
    /* After joining, print out the results and cleanup */
    printf ("Total number of Prime =  %d \nTime: %f\nMax: %f\n", sum, sum1,maxi2);


//
//    for (int i = 0;  i <= VECLEN; i++) {
//        std::cout << i << ":" << Comp[i] << std::endl ;
//    }

    pthread_mutex_destroy(&mutexsum);
    pthread_exit(NULL);

}
