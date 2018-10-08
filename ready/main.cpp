#include "Matriz.h"
using namespace std;

//n number of repetitions to calculate the average of the times
void ShowTimeAvg(MatrizP * m, int n, char tipo);

int main(int argc, char* argv[])
{
    MatrizP * m;
    m = new Matriz<float>();

    m->ReadM1(argv[1]);
    m->ReadM2(argv[2]);
    char t = argv[4][0];
	
    ShowTimeAvg(m,1,t);

	return 0;
}


void ShowTimeAvg(MatrizP * m, int n, char tipo){

    float sum=0.0;
    int i;
    long long int start1,end1;
	char  y[100];
    for(i=0;i<n;i++){

    start1 = clock();
    m->MultMatriz(tipo);
    end1 = clock();
    
	if(i==0){
	strcpy(y,"C");
    m->SaveM3(y);
	
	
	if(tipo=='1'){
		strcpy(y,"C0");
        m->SaveM3(y);
	}
	}
	
    sum = sum + (double)(end1 - start1) / 1000;

   }
   
	
	FILE * pFile;
	char name [100],x1;
	strcpy(name,"plot.bin");
	pFile = fopen (name,"rb + wb");
	
	//if(tipo != '1'){
	// while (!feof(pFile))
		 // fscanf (pFile, "%c", &x1);
	// }
    
	fseek(pFile,0,SEEK_END);
    fprintf (pFile, "%c %f\n",tipo,sum/n);
	
	fclose (pFile);


}
