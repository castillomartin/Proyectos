#include "Matriz.h"
#include "papi.h"

static void test_fail(char *file, int line, char * call, int retval);

int main(int argc, char **argv) {

   int events[1];
   long long counts[1];
   
   int retval,quiet;

   char test_string[]="Testing PAPI_TLB_TL predefined event...";
   
  
    char salida[100];
	strcpy(salida,"Papi_flops");
   retval = PAPI_library_init(PAPI_VER_CURRENT);
   if (retval != PAPI_VER_CURRENT) {	
      //test_fail(test_string);
   }

   retval = PAPI_query_event(PAPI_TLB_TL);
   if (retval != PAPI_OK) {
      if (!quiet){ printf("PAPI_TLB_TL not available\n");return 0;}
		
   }

   events[0]=PAPI_TLB_TL;

   PAPI_start_counters(events,1);
     
   PAPI_stop_counters(counts,1);

   if (counts[0]<1) {
      }

   PAPI_shutdown();

   
   return 0;
}


static void test_fail(char *file, int line, char * call, int retval){
    printf("%s\tFAILED\nLine # %d\n", file, line);
    if ( retval == PAPI_ESYS ) {
        char buf[128];
        memset( buf, '\0', sizeof(buf) );
        sprintf(buf, "System error in %s:", call );
        perror(buf);
    }
    else if ( retval > 0 ) {
        printf("Error calculating: %s\n", call );
    }
    else {
        printf("Error in %s: %s\n", call, PAPI_strerror(retval) );
    }
    printf("\n");
    exit(1);
}
