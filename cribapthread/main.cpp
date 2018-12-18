#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <ctime>
#include <semaphore.h>
#define BILLION  1000000000L;

struct thrd_data{
  long id;
  long start;
  long end;
};
typedef struct {
  pthread_mutex_t     count_lock;
  pthread_cond_t      ok_to_proceed;
  long                count;
} mylib_barrier_t;

double sum1 = 0.0,maxi=0.0,aux = 0.0;
pthread_mutex_t mutetime;
long long int start1,end1;
bool *GlobalList;
long Num_Threads;
mylib_barrier_t barrier;
struct timespec starttime, stoptime;


void mylib_barrier_init(mylib_barrier_t *b)
{
  b -> count = 0;
  pthread_mutex_init(&(b -> count_lock), NULL);
  pthread_cond_init(&(b -> ok_to_proceed), NULL);
}

void mylib_barrier(mylib_barrier_t *b, long id)
{
   pthread_mutex_lock(&(b -> count_lock));
   b -> count ++;
   if (b -> count == Num_Threads)
   {
     b -> count = 0;
     pthread_cond_broadcast(&(b -> ok_to_proceed));
   }
   else
   {
    while (pthread_cond_wait(&(b -> ok_to_proceed), &(b -> count_lock)) !=    0);

    }
    pthread_mutex_unlock(&(b -> count_lock));
}

void mylib_barrier_destroy(mylib_barrier_t *b)
{
  pthread_mutex_destroy(&(b -> count_lock));
  pthread_cond_destroy(&(b -> ok_to_proceed));
}

void *Sieve(void *thrd_arg)
{

  struct thrd_data *t_data;
  long i,start, end;
  long k=2;
  long myid;

  /* Initialize my part of the global array */
  t_data = (struct thrd_data *) thrd_arg;
  

  myid = t_data->id;
  start = t_data->start;
  end = t_data->end;

  printf ("Thread %ld: %ld - %ld\n", myid,start,end);
  
  if( clock_gettime( CLOCK_REALTIME, &starttime) == -1 ) {
      perror( "clock gettime" );
      exit( EXIT_FAILURE );
  }
	
	
  //First loop: find all prime numbers that's less than sqrt(n)
  while (k*k<=end)
  {
      int flag;
      if(k*k>=start)
        flag=0;
      else
        flag=1;
      //Second loop: mark all multiples of current prime number
      for (i = !flag? k*k-1:start+k-start%k-1; i <= end; i += k)
        GlobalList[i] = 1;
      i=k;
      mylib_barrier(&barrier,myid);
      
      while (GlobalList[i] == 1)
            i++;
         k = i+1;

   }

	//end1 = clock();

    if( clock_gettime( CLOCK_REALTIME, &stoptime) == -1 ) {
      perror( "clock gettime" );
      exit( EXIT_FAILURE );
    }
	
	
    pthread_mutex_lock (&mutetime);
  	
	aux = ( stoptime.tv_sec - starttime.tv_sec )
          + ( stoptime.tv_nsec - starttime.tv_nsec )
            / (double)BILLION;
	sum1 += aux;
	if(maxi < aux)
        maxi = aux;
	
	
    pthread_mutex_unlock (&mutetime);
	
	
  pthread_mutex_lock (&barrier.count_lock);
  Num_Threads--;
  if (barrier.count == Num_Threads)
  {
    barrier.count = 0;  
    pthread_cond_broadcast(&(barrier.ok_to_proceed));
  }
  pthread_mutex_unlock (&barrier.count_lock);
  pthread_exit(NULL);
}


int main(int argc, char *argv[])
{
	char* end;
  long i, n,n_threads,m;
  long k, nq, nr;
  FILE *results;
  struct thrd_data *t_arg;
  pthread_t *thread_id;
  pthread_attr_t attr;
	
	if(argc>=3){
		n_threads=strtol(argv[3],&end,10);
		m = strtol(argv[1],&end,10);
		n = strtol(argv[2],&end,10);
	}
	
  mylib_barrier_init(&barrier);
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

  
 
  //Initialize global list
  GlobalList=(bool *)malloc(sizeof(bool)*n);
  for(i=0;i<n;i++)
    GlobalList[i]=0;

	GlobalList[0] = GlobalList[1] = 1;
	GlobalList[2] = GlobalList[3] = 0;
  thread_id = (pthread_t *)malloc(sizeof(pthread_t)*n_threads);
  t_arg = (struct thrd_data *)malloc(sizeof(struct thrd_data)*n_threads);

  nq = (n-m) / n_threads;
  nr = (n-m) % n_threads;

  //if(m<1)m++
  k = m;
  Num_Threads=n_threads;
  for (i=0; i<n_threads; i++){
    t_arg[i].id = i;
    t_arg[i].start = k;
    if (i < nr)
        k = k + nq + 1;
    else
        k = k + nq;
    t_arg[i].end = k-1;
    pthread_create(&thread_id[i], &attr, Sieve, (void *) &t_arg[i]);
  }

  /* Wait for all threads to complete then print all prime numbers */
  for (i=0; i<n_threads; i++) {
    pthread_join(thread_id[i], NULL);
  }
  int j=0;
  
  char file[] = "result.bin";
  FILE * f = fopen(file,"a+b");
  
  
  
  GlobalList[1] = 0;
  GlobalList[2] = 0;
  for (i = m; i < n; i++)
  {
	  
    if (GlobalList[i] == 0)
    {
		 fprintf(f,"%ld ", i + 1);
        //printf("%ld ", i + 1);
        j++;
    }
  }
   printf ("Total number of Prime =  %d \nTime: %f\nMax: %f\n", j, sum1,maxi);
  printf("\n");
  // Clean up and exit
  free(GlobalList);
  pthread_attr_destroy(&attr);
  mylib_barrier_destroy(&barrier); // destroy barrier object
  pthread_exit (NULL);
}
