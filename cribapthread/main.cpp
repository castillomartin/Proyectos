#include <pthread.h>
#include <semaphore.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <ctime>


int NUMTHRDS = 4;
pthread_t * callThd;
pthread_mutex_t mutexsum;
pthread_mutex_t mutetime;
int sum;
float sum1 = 0.0,maxi=0.0;
long * Comp;
int VECLEN = 50;
int IECLEN = 10;
long long int start1,end1;

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
     Sieve of Eratoshenes.
     */
	 
	start1 = clock();
	
    for (i = start; i < end	; i++) {
        if (i > 2) {
            if (Comp[i] == 0) {
                for (int x = 3 ; x < i; x +=2){
                    if ( x*x > i)
                        break;
                    if ( i % x == 0 ){
                        Comp[i] = 1;
						if(offset == 2)
							//printf("%d %d \n",i,x);
                        break;
                    }
                }
            }
        }
    }  
	
	end1 = clock();
	
	for (i = start; i < end; i++) {
        if (i > 1 && Comp[i] == 0) {
            mysum++;
            fprintf(f,"%d\n",i);
        }
    }

  

    pthread_mutex_lock (&mutetime);
    sum1 = sum1 + (double)(end1 - start1) / 1000;
    //printf("%d-%f: \n",offset,(double)(end1 - start1) / 1000);
    if(maxi < (double)(end1 - start1) / 1000)
        maxi = (double)(end1 - start1) / 1000;
    pthread_mutex_unlock (&mutetime);

    fclose(f);

    pthread_mutex_lock (&mutexsum);
    std::cout << "Start : " << start << " End : " << end << " len: " << len << std::endl;
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
	
	/**
     Initialize the portion of Comp
     */
    for (i = 0; i < VECLEN; i++) {
        if (i < 3){
            if (i < 2) {
                Comp[i] = 1;
            }else{
                Comp[i] = 0;
            }
        }else{
            if (i % 2 == 0 || i%3 == 0 || i%5 ==0 || i%7 == 0 || i%11 == 0 ) {
				if(i!=3 && i!=5 && i!=7 && i!=11)
					Comp[i] = 1;
            }
        }
    }
	
	
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
    fprintf(g,"%d %f %f\n",NUMTHRDS,sum1,maxi);
    fclose(g);
    /* After joining, print out the results and cleanup */
    printf ("Total number of Prime =  %d \nTime: %f\nMax: %f\n", sum, sum1,maxi);


//
//    for (int i = 0;  i <= VECLEN; i++) {
//        std::cout << i << ":" << Comp[i] << std::endl ;
//    }

    pthread_mutex_destroy(&mutexsum);
    pthread_exit(NULL);

}
