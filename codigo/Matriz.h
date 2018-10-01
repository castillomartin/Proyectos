#ifndef MATRIZ_H
#define MATRIZ_H
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctime>

class MatrizP{
public:
MatrizP(){}
 virtual void ReadFile()=0;
 virtual void PrintMatriz(int x)=0;
 virtual void MultMatriz(int x)=0;
};

template <class T>
class Matriz : public MatrizP
{
   private:
    int f1,c1,f2,c2,f3,c3;
    T **m1, **m2, **m3;

    public:
        Matriz(){}
        Matriz(int pf1,int pc1,int pf2,int pc2){

        f1 = pf1;
        f2 = pf2;
        f3 = pf1;
        c1 = pc1;
        c2 = pc2;
        c3 = pc2;

        m1 = (T**)malloc(f1*sizeof(T*));
        m2 = (T**)malloc(f2*sizeof(T*));

        int i;
        for(i=0;i<f1;i++){
            m1[i] = (T*)malloc(c1*sizeof(T));
        }
        for(i=0;i<f2;i++){
            m2[i] = (T*)malloc(c2*sizeof(T));
        }

        }

        int getf1(){return f1;}
        int getf2(){return f2;}
        int getf3(){return f3;}
        int getc1(){return c1;}
        int getc2(){return c2;}
        int getc3(){return c3;}
        T ** getm1(){return m1;}
        T ** getm2(){return m2;}
        T ** getm3(){return m3;}

        void setf1(int x){f1 = x;}
        void setf2(int x){f2 = x;}
        void setf3(int x){f3 = x;}
        void setc1(int x){c1 = x;}
        void setc2(int x){c2 = x;}
        void setc3(int x){c3 = x;}

        void setM1(int i, int j,T  x){m1[i][j] = x;}
        void setM2(int i, int j,T x){m2[i][j] = x;}

        void ReadFile(){

         FILE * archivo;


    char tipa;
    int N1,M1,N2,M2;
    int cont = 0,i=0,j=0,k=0;
    T d = 0.0;
	archivo = fopen("lectura500x100x300.bin","r+");

	if (archivo == NULL)
        {
            printf("\nError de apertura del archivo. \n\n");
        }
        else
        {
            printf("\n Content \n\n");
            while (!feof(archivo))
            {
                if(cont == 0)
                fscanf (archivo, "%c", &tipa);
                else if(cont == 1)
                fscanf (archivo, "%d", &N1);
                else if(cont == 2)
                fscanf (archivo, "%d", &M1);
                else if(cont == 3)
                fscanf (archivo, "%d", &N2);
                else if(cont == 4)
                fscanf (archivo, "%d", &M2);

                else{
//                    printf("%d %d %d %d", N1,M1,N2,M2);

                    if( cont == 5){
                        setf1(N1);
                        setf2(N2);
                        setf3(N1);
                        setc1(M1);
                        setc2(M2);
                        setc3(M2);

                        m1 = (T**)malloc(N1*sizeof(T*));
                        m2 = (T**)malloc(N2*sizeof(T*));
                        m3 = (T**)malloc(N1*sizeof(T*));

                        int i;
                        for(i=0;i<N1;i++){
                            m1[i] = (T*)malloc(M1*sizeof(T));
                            m3[i] = (T*)malloc(M2*sizeof(T));
                        }
                        for(i=0;i<N2;i++){
                            m2[i] = (T*)malloc(M2*sizeof(T));
                        }

                    }

                    fscanf (archivo, "%f", &d);


                    if(i<N1){
                    setM1(i,j++,d);

                    //printf("%d", d);
                    if(j == M1 )
                    {
                        i++;
                        j=0;
                    //printf("\n");
                    }
                    }
                    else{
                    setM2(k,j++,d);

                    //printf("%d", d);
                    if(j == M2 )
                    {
                        k++;
                        j=0;
                    //printf("\n");
                    }
                    }


                }

            cont++;
            }
        }
        fclose(archivo);

        }

        void PrintMatriz(int v){
            int i,j;
            if(v == 1){

                for(i = 0;i <getf1();i++){
                    for(j = 0;j <getc1();j++)
                        printf("%6.1f ",getm1()[i][j]);
                    printf("\n");
                }
            }
            else if(v == 2){

                for(i = 0;i <getf2();i++){
                    for(j = 0;j <getc2();j++)
                        printf("%6.1f ",getm2()[i][j]);
                    printf("\n");
                }
            }
            else if(v == 3){

                for(i = 0;i <getf3();i++){
                    for(j = 0;j <getc3();j++)
                        printf("%6.1f ",getm3()[i][j]);
                    printf("\n");
                }
            }

            printf("\n");
        }

        void MultMatriz(int tipo){

            double sum = 0.0;
            int i,j,k;

            for(i=0;i<getf1();i++){
                 m3[i] = (T*)malloc(getc2()*sizeof(T));
            }
            //i-j-k
            if(tipo == 1)
            for(i = 0;i<getf1();i++)
                for(j = 0;j<getc2();j++){
                        sum = 0.0;
                    for(k = 0;k<getf2();k++){
                        sum+=m1[i][k]*m2[k][j];
                        m3[i][j] = sum;
                    }
                }

            //i-k-j
            if(tipo == 2)
            for(i = 0;i<getf1();i++)
                for(k = 0;k<getc1();k++){
                    sum=m1[i][k];
                    for(j = 0;j<getc2();j++){
                        m3[i][j] += sum*m2[k][j];
                    }
                }

            //j-i-k
            if(tipo == 3)
            for( j = 0; j < getc2(); j ++){
              for( i = 0; i <  getf1(); i++){
                 for( k = 0; k < getc1(); k ++){
                    m3[i][j] += m1[i][k] * m2[k][j];
                 }
               }
            }


            //j-k-i
            if(tipo == 4)
            for(j = 0;j<getc2();j++)
                for(k = 0;k<getf2();k++){
                    sum=m2[k][j];
                    for(i = 0;i<getf1();i++){
                        m3[i][j] += sum*m1[i][k];
                    }

                }
            //k-i-j
            if(tipo == 5)
            for( k = 0; k < getc1(); k ++){
                for( i = 0; i < getf1(); i ++){
                    for( j = 0; j < getc2(); j ++){
                        m3[i][j] += m1[i][k] * m2[k][j];
                    }
                }
            }

            //k-j-i
            if(tipo == 6)
            for(int k = 0; k < getc1(); k ++){
                for(int j = 0; j < getc2(); j ++){
                    for(int i = 0; i < getf1(); i ++){
                        m3[i][j] += m1[i][k] * m2[k][j];
                    }
                }
            }

        }

};

#endif // MATRIZ_H
