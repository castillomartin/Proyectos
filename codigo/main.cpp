#include "Matriz.h"
using namespace std;

int ToCheck();

//n number of repetitions to calculate the average of the times
void ShowTimeAvg(MatrizP * m, int n);

int main()
{
    int flag = ToCheck();
    MatrizP * m;

    if(flag == 0){
        printf("The file does not exists");
        return 0;
    }
    else if(flag == 1)
         m = new Matriz<float>();
    else if(flag == 2)
         m = new Matriz<double>();
    else{
        printf("error while reading data type");
        return 0;
    }

    m->ReadFile();
    ShowTimeAvg(m,5);

	return 0;
}




int ToCheck(){
    FILE * archivo;
    char tipa;
    archivo = fopen("lectura500x100x300.bin","r+");
        if (archivo == NULL)
            return 0;
        else{
            fscanf (archivo, "%c", &tipa);
            if(tipa == 'f')
                return 1;
            else if(tipa == 'd')return 2;
            else return 3;

        }
}

void ShowTimeAvg(MatrizP * m, int n){

    float sum1=0.0,sum2=0.0,sum3=0.0,sum4=0.0,sum5=0.0,sum6=0.0;
    int i;
    long long int start1,end1;

    for(i=0;i<n;i++){

    start1 = clock();
    m->MultMatriz(1);
    //m->PrintMatriz(3);
    end1 = clock();
    sum1 = sum1 + (double)(end1 - start1) / 1000000;

    start1 = clock();
    m->MultMatriz(2);
    //m->PrintMatriz(3);
    end1 = clock();
    sum2 = sum2 + (double)(end1 - start1) / 1000000;

    start1 = clock();
    m->MultMatriz(3);
    //m->PrintMatriz(3);
    end1 = clock();
    sum3 = sum3 + (double)(end1 - start1) / 1000000;

    start1 = clock();
    m->MultMatriz(4);
    //m->PrintMatriz(3);
    end1 = clock();
    sum4 = sum4 + (double)(end1 - start1) / 1000000;

    start1 = clock();
    m->MultMatriz(5);
    //m->PrintMatriz(3);
    end1 = clock();
    sum5 = sum5 + (double)(end1 - start1) / 1000000;

    start1 = clock();
    m->MultMatriz(6);
    //m->PrintMatriz(3);
    end1 = clock();
    sum6 = sum6 + (double)(end1 - start1) / 1000000;

    }


    printf("   i-j-k %5.8f seconds\n\n", sum1/n);
    printf("   i-k-j %5.8f seconds\n\n", sum2/n);
    printf("   j-k-i %5.8f seconds\n\n", sum3/n);
    printf("   j-i-k %5.8f seconds\n\n", sum4/n);
    printf("   k-i-j %5.8f seconds\n\n", sum5/n);
    printf("   k-j-i %5.8f seconds\n\n", sum6/n);
}
