#include "Matriz.h"
using namespace std;

//n number of repetitions to calculate the average of the times
void ShowTimeAvg(MatrizP * m, int n);

int main(int argc, char* argv[])
{
    MatrizP * m;
    m = new Matriz<float>();

    m->ReadM1(argv[1]);
    m->ReadM2(argv[2]);

    ShowTimeAvg(m,1);

	return 0;
}


void ShowTimeAvg(MatrizP * m, int n){

    float sum1=0.0,sum2=0.0,sum3=0.0,sum4=0.0,sum5=0.0,sum6=0.0;
    int i;
    long long int start1,end1;
	char  y[100];
    for(i=0;i<n;i++){

    start1 = clock();
    m->MultMatriz(1);
    //m->PrintMatriz(3);
    end1 = clock();
    if(i==0){
		strcpy(y,"C1.bin");
        m->SaveM3(y);
		 strcpy(y,"C0.bin");
         m->SaveM3(y);
	}
    sum1 = sum1 + (double)(end1 - start1) / 1000;

    start1 = clock();
    m->MultMatriz(2);
    //m->PrintMatriz(3);
    end1 = clock();
     if(i==0){
		 strcpy(y,"C2.bin");
         m->SaveM3(y);
	 }
    sum2 = sum2 + (double)(end1 - start1) / 1000;

    start1 = clock();
    m->MultMatriz(3);
    //m->PrintMatriz(3);
    end1 = clock();
     if(i==0){
		 strcpy(y,"C3.bin");
         m->SaveM3(y);
	 }
    sum3 = sum3 + (double)(end1 - start1) / 1000;

    start1 = clock();
    m->MultMatriz(4);
    //m->PrintMatriz(3);
    end1 = clock();
     if(i==0){
		 strcpy(y,"C4.bin");
         m->SaveM3(y);
	 }
    sum4 = sum4 + (double)(end1 - start1) / 1000;

    start1 = clock();
    m->MultMatriz(5);
    //m->PrintMatriz(3);
    end1 = clock();
     if(i==0){
		 strcpy(y,"C5.bin");
         m->SaveM3(y);
	 }
    sum5 = sum5 + (double)(end1 - start1) / 1000;

    start1 = clock();
    m->MultMatriz(6);
    //m->PrintMatriz(3);
    end1 = clock();
     if(i==0){
		 strcpy(y,"C6.bin");
         m->SaveM3(y);
	 }
    sum6 = sum6 + (double)(end1 - start1) / 1000;

    }

	FILE * pFile;
	char name [100];
	strcpy(name,"plot.bin");
	pFile = fopen (name,"w");
	
	puts("\n");
    printf("   i-j-k %5.8f seconds\n\n", sum1/n);
    fprintf (pFile, "%d %f\n",1,sum1/n);
    printf("   i-k-j %5.8f seconds\n\n", sum2/n);
    fprintf (pFile, "%d %f\n",2,sum2/n);
    printf("   j-k-i %5.8f seconds\n\n", sum3/n);
    fprintf (pFile, "%d %f\n",3,sum3/n);
    printf("   j-i-k %5.8f seconds\n\n", sum4/n);
    fprintf (pFile, "%d %f\n",4,sum4/n);
    printf("   k-i-j %5.8f seconds\n\n", sum5/n);
    fprintf (pFile, "%d %f\n",5,sum5/n);
    printf("   k-j-i %5.8f seconds\n\n", sum6/n);
    fprintf (pFile, "%d %f\n",6,sum6/n);
	
	fclose (pFile);
}
