#include "Matriz.h"
#include "papi.h"

#define numEvents 6
using namespace std;

//Multiplication Matrix
void MultiplicationMatriz(MatrizP * m, char tipo, char * C);
//tets filebuf
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
    
	if(argc == 6){
		m->setblock(atoi(argv[5]));
	}
	
	else{
		m->setblock(32);
	}
	
	
	int tipo;
	if(argc < 6)tipo = 1;
	else if(t=='7')tipo = 2;
	else if(t=='8')tipo = 3;
	if(argc == 6 && m->getblock()==1) tipo = 4;
	
	
	if((retval=PAPI_flops( &real_time, &proc_time, &flpins, &mflops))<PAPI_OK)
		test_fail(__FILE__, __LINE__, salida, retval);

	//Matrix Multiplication
    MultiplicationMatriz(m,t,C);

	if((retval=PAPI_flops( &real_time, &proc_time, &flpins, &mflops))<PAPI_OK)
		test_fail(__FILE__, __LINE__, salida, retval);


	printf("Real_time:\t%f\nProc_time:\t%f\nTotal flpins:\t%lld\nMFLOPS:\t\t%f\n",
	real_time, proc_time, flpins, mflops);
	printf("%s\tPASSED\n", __FILE__);

	
	PAPI_shutdown();
	

	FILE * pFile;
	char name [100],x1;
	strcpy(name,"flops.bin");
	pFile = fopen (name,"a+b");
    fseek(pFile,0,SEEK_END);

	if(tipo!=1)
	fprintf (pFile, "%d %f\n",tipo-1,mflops);
	
	strcpy(name,"time.bin");
	pFile = fopen (name,"a+b");    
	fseek(pFile,0,SEEK_END);
	
	if(tipo!=1)
	fprintf (pFile, "%d %f\n",tipo-1,proc_time);

	fclose (pFile);
	
	
  
	return 0;
}


void MultiplicationMatriz(MatrizP * m, char tipo, char * C){

    float sum=0.0;
    int i;
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

static void test_fail(char *file, int line, char *call, int retval){
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

void handle_error(int err){
    std::cerr << "PAPI error: " << err << std::endl;
}