#ifndef MATRIZ_H
#define MATRIZ_H
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctime>

class MatrizP{
public:
MatrizP(){}
 virtual void PrintMatriz(int x)=0;
 virtual void MultMatriz(char x)=0;
 virtual void ReadM1(char * x)=0;
 virtual void ReadM2(char * x)=0;
 virtual void SaveM3(char * x)=0;
 virtual int getc1()=0;
 virtual int getf1()=0;
 virtual int getc2()=0;
 virtual int getf2()=0;
 virtual float getm1pos(int i, int j)=0;
 virtual float getm2pos(int i, int j)=0;
 virtual void setblock(int x)=0;
 virtual int getblock()=0;
};

template <class T>
class Matriz : public MatrizP
{
   private:
    int f1,c1,f2,c2,f3,c3,block;
    T **m1, **m2, **m3;
    public:
        Matriz(){}

        //GET
        int getf1(){return f1;}
        int getf2(){return f2;}
        int getf3(){return f3;}
        int getc1(){return c1;}
        int getc2(){return c2;}
        int getc3(){return c3;}
        T ** getm1(){return m1;}
        T ** getm2(){return m2;}
        T ** getm3(){return m3;}
		int getblock(){return block;}
		float getm1pos(int i, int j){return m1[i][j];}
		float getm2pos(int i, int j){return m2[i][j];}

        //SET
        void setf1(int x){f1 = x;}
        void setf2(int x){f2 = x;}
        void setf3(int x){f3 = x;}
        void setc1(int x){c1 = x;}
        void setc2(int x){c2 = x;}
        void setc3(int x){c3 = x;}
        void setblock(int x){block = x;}
        void setM1(int i, int j,T  x){m1[i][j] = x;}
        void setM2(int i, int j,T x){m2[i][j] = x;}

        //PRINT MATRIX
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

        //MUTL MATRIX
        void MultMatriz(char tipo){

            double sum = 0.0;
            int i,j,k;

            setf3(getf1());
            setc3(getc2());

            m3 = (T**)malloc(getf1()*sizeof(T*));
            for(i=0;i<getf1();i++){
                 m3[i] = (T*)malloc(getc2()*sizeof(T));
            }
            //i-j-k
            if(tipo == '1')
            for(i = 0;i<getf1();i++)
                for(j = 0;j<getc2();j++){
                        sum = 0.0;
                    for(k = 0;k<getf2();k++){
                        sum+=m1[i][k]*m2[k][j];
                        m3[i][j] = sum;
                    }
                }

            //i-k-j
            if(tipo == '2')
            for(i = 0;i<getf1();i++)
                for(k = 0;k<getc1();k++){
                    sum=m1[i][k];
                    for(j = 0;j<getc2();j++){
                        m3[i][j] += sum*m2[k][j];
                    }
                }

            //j-i-k
            if(tipo == '3')
            for( j = 0; j < getc2(); j ++){
              for( i = 0; i <  getf1(); i++){
                 for( k = 0; k < getc1(); k ++){
                    m3[i][j] += m1[i][k] * m2[k][j];
                 }
               }
            }


            //j-k-i
            if(tipo == '4')
            for(j = 0;j<getc2();j++)
                for(k = 0;k<getf2();k++){
                    sum=m2[k][j];
                    for(i = 0;i<getf1();i++){
                        m3[i][j] += sum*m1[i][k];
                    }

                }
            //k-i-j
            if(tipo == '5')
            for( k = 0; k < getc1(); k ++){
                for( i = 0; i < getf1(); i ++){
                    for( j = 0; j < getc2(); j ++){
                        m3[i][j] += m1[i][k] * m2[k][j];
                    }
                }
            }

            //k-j-i
            if(tipo == '6')
            for(int k = 0; k < getc1(); k ++){
                for(int j = 0; j < getc2(); j ++){
                    for(int i = 0; i < getf1(); i ++){
                        m3[i][j] += m1[i][k] * m2[k][j];
                    }
                }
            }
			
			//i-j-k blocks
			if(tipo=='7'){
				int sum = 0,i1,j1,k1;
				int b = GetCorrectSize(getblock());
				for(i = 0;i<getf1();i+=b){
					for(j = 0;j<getc2();j+=b){
						for(k = 0;k<getf2();k+=b){
										
							for(i1 = i; i1 < i+b; i1++)
								for(j1 = j; j1 < j+b; j1++)
									for(k1 = k; k1 < k+b; k1++){
										sum=m1[i1][k1]*m2[k1][j1];
										m3[i1][j1] += sum;
									}
						}
					}
				}
			}
			
			//i-k-j blocks
			if(tipo=='8'){
				int sum = 0,i1,j1,k1;
				int b = GetCorrectSize(getblock());
				for(i = 0;i<getf1();i+=b)
					for(k = 0;k<getc1();k+=b){
						for(j = 0;j<getc2();j+=b){
										
							for(i1 = i; i1 < i+b; i1++)
								for(k1 = k; k1 < k+b; k1++){	
									sum=m1[i1][k1];
									for(j1 = j; j1 < j+b; j1++){
										m3[i1][j1] += sum*m2[k1][j1];
									}
								}
						}
					}
			}
		

        }

		
        //READ M1
        void ReadM1(char *y){
          FILE * pFile;
          pFile = fopen (y, "rb");
          T x;
          int i,j;
          char tipo;
          fread(&tipo,sizeof(char),1,pFile);
          fread(&f1,sizeof(int),1,pFile);
          fread(&c1,sizeof(int),1,pFile);

          m1 = (T**)malloc(f1*sizeof(T*));
            for(i=0;i<f1;i++)
                m1[i] = (T*)malloc(c1*sizeof(T));

          for(i=0;i<f1;i++)
            for(j=0;j<c1;j++){
                fread(&x,sizeof(T),1,pFile);
                setM1(i,j,x);
                }

        }

		int GetCorrectSize(int x){
			if(x==1){
				int ml = 3072;
				int b = sqrt(ml/3);
				for(int i=b;i<c1;i++)
					if(c1%i==0)return i;
			}
			if(c1%x==0)
				return x;
			else{
				for(int i=x+1;i<c1;i++)
					if(c1%i==0)return i;
			}
		}
        //READ M2
        void ReadM2(char * y){
          FILE * pFile;
          pFile = fopen (y, "rb");
          T x;
          int i,j;
          char tipo;
          fread(&tipo,sizeof(char),1,pFile);
          fread(&f2,sizeof(int),1,pFile);
          fread(&c2,sizeof(int),1,pFile);

          m2 = (T**)malloc(f2*sizeof(T*));
          for(i=0;i<f2;i++)
              m2[i] = (T*)malloc(c2*sizeof(T));

          for(i=0;i<f2;i++)
            for(j=0;j<c2;j++){
                fread(&x,sizeof(T),1,pFile);
                setM2(i,j,x);
                }
        }

        //Save M3
        void SaveM3(char * y){
          FILE * pFile;
          pFile = fopen (y, "wb");
          int i,j;
          char tipo = 'f';
          fwrite(&tipo,sizeof(char),1,pFile);
          fwrite(&f3,sizeof(int),1,pFile);
          fwrite(&c3,sizeof(int),1,pFile);

          for(i=0;i<f3;i++)
            for(j=0;j<c3;j++)
                fwrite(&getm3()[i][j],sizeof(T),1,pFile);
        }
};

#endif // MATRIZ_H
