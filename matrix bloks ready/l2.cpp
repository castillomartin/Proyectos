#include "Matriz.h"
#include "papi.h"

#define numEvents 1
using namespace std;

//Multiplication Matrix
void MultiplicationMatriz(MatrizP * m, char tipo, char * C);//tets filebuf
static void test_fail(char *file, int line, char *call, int retval);


int main(int argc, char* argv[])
{
	extern void dummy(void *);
	float real_time, proc_time, mflops;
	long long flpins;
	int retval;
  
    MatrizP * m;
    m = new Matriz<float>();

    m->ReadM1(argv[1]);
    m->ReadM2(argv[2]);
    char  C[100];
	strcpy(C,argv[3]);
    char t = argv[4][0];
    char salida[100];
	strcpy(salida,"Papi_flops");
    long long values[numEvents];
    values[0] = 0;
    int events[numEvents] = {PAPI_L2_DCM};

	
	if(argc == 6){
		m->setblock(atoi(argv[5]));
	}
	
	else{
		m->setblock(32);
	}
	
    if ((retval = PAPI_start_counters(events, numEvents)) != PAPI_OK)
        test_fail(__FILE__, __LINE__, salida, retval);

	//Matrix Multiplication
    MultiplicationMatriz(m,t,C);

	 if ((retval = PAPI_stop_counters(values, numEvents)) != PAPI_OK)
        test_fail(__FILE__, __LINE__, salida, retval);
	

	
    printf("L2 misses: %ld\n",values[0]);
    
	FILE * pFile;
	char name [100],x1;
	strcpy(name,"l2.bin");
	pFile = fopen (name,"a+b");
	    
	fseek(pFile,0,SEEK_END);
	int tipo;
	if(argc < 6)tipo = 1;
	else if(t=='7')tipo = 2;
	else if(t=='8')tipo = 3;
	if(argc == 6 && m->getblock()!=32) tipo = 4;
	
	if(tipo!=1)
    fprintf (pFile, "%d %d\n",tipo-1,values[0]);
	
	fclose (pFile);


	PAPI_shutdown();


  
	return 0;
}


void MultiplicationMatriz(MatrizP * m, char tipo, char * C){

    float sum=0.0;
    int i = 0;
	char  y[100];
	
	m->MultMatriz(tipo);
    
	if(i==0){
		m->SaveM3(C);
		
		if(tipo=='1'){
			strcpy(y,"C0");
			m->SaveM3(y);
		}
	}
   


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
