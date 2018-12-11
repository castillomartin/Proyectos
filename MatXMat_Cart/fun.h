#include "Matrix.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <string>
#include <mpi.h>

using namespace std;

Matrix ReadFile(const string &strr){

	int n, m;
	double val;
	Matrix result;

	const char* cstr = strr.c_str();
	fstream f1(cstr, ios::binary | ios::in);
	f1.read((char*)&n, sizeof(int));
	f1.read((char*)&m, sizeof(int));
	result.setRows(n);
	result.setColumns(m);

	result.matriz.resize(n);
	for (int i = 0; i < n; ++i){
		result.matriz[i].resize(m);
		for (int j = 0; j < m; j++){
			f1.read((char*)&val, sizeof(double));
			result.matriz[i][j] = val;
		}
	}
	f1.close();
	return result;
}


double* ChangeArray (Matrix &m) {
	
	int cont = 0;
	int my_size;
	double * array = new double[m.getColumns() * m.getRows()];
	
	for (int i = 0; i < m.getRows(); ++i){
		for (int j =0; j < m.getColumns(); ++j){
			array[cont++] = m.matriz[i][j];	
		}
	}
	
	return array;
}


Matrix ChangeMatriz(int size , double *array) {
	int cont = 0;

	Matrix result;
	result.setRows(size);
	result.setColumns(size);
	
	result.matriz.resize(size);
	for ( int i = 0; i < size; i++ ){
		result.matriz[i].resize(size);
		for ( int j = 0; j < size; j++ ){
			result.matriz[i][j] = array[cont++];
		}
	}
	return result;
}

void SaveResult( char* filename, Matrix &m){
	
	MPI_File cFile;
	MPI_Status  status;
	int r1 = MPI_File_open( MPI_COMM_SELF, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE , MPI_INFO_NULL, &cFile);
	double aux;
	
    int AROW = m.getRows();
    int COLMS = m.getColumns();
	  
	MPI_File_write(cFile,&AROW,1,MPI_INT,&status);
	MPI_File_write(cFile,&COLMS,1,MPI_INT,&status);
	
	// for (int i = 0; i < AROW; ++i) {
		// for (int j = 0; j < COLMS; ++j) {
			// aux = m.matriz[i][j];
			// MPI_File_write(cFile,&aux,1,MPI_DOUBLE,&status);
        // }
    // }
	
	MPI_File_close( &cFile );
	
	ofstream ft(filename, ios::binary | ios::app);
    if (ft.is_open()) {
		double aux;
	  
		for (int i = 0; i < AROW; ++i) {
			for (int j = 0; j < COLMS; ++j) {
			aux = m.matriz[i][j];
			ft.write((char *)&aux, sizeof(double));
			}
		}
    }
    ft.close();
	  
}


void SaveComplement(double _max, int _size, double _total){
	
	float nodo1;
	double speedup, efficiency;
	ifstream in ("base_1", ios::in);
	in >> nodo1;
	in.close();
	speedup = (double)nodo1 / _max;
	efficiency = speedup / _size;
	ofstream out("result", ios::app);
	out << _size << " " << _total << " " << _max << " " << speedup << " " << efficiency <<endl; 
	out.close();
	
	
}

Matrix Mat_x_Mat(Matrix &m1, Matrix &m2){
	  
	 int x = m1.getRows();
	 int y = m2.getColumns();
	  
	 vector< vector<double> > aux(x, vector<double>(m2.getColumns(), 0));
	  
	 Matrix aux2(aux,x,y);
	 int block = 32;
	  
	  
	 //Block Multiplication 
	 for (int i = 0; i < m1.getRows(); i += block) 
		for (int k = 0; k < m2.getColumns(); k += block) 
			for (int j = 0; j < m1.getColumns(); j += block) 
				for (int i1 = i; i1 < i + block && i1 < m1.getRows(); i1++) 
					for (int k1 = k; k1 < k + block && k1 < m2.getColumns(); k1++) 
						for (int j1 = j; j1 < j + block && j1 < m1.getColumns(); j1++) 
							aux2.matriz[i1][j1] += m1.matriz[i1][k1] * m2.matriz[k1][j1];
				
	  
	  
	return aux2;
	  
}

