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
	double * array = new double[m.getColumns() * m.getRows()];
	
	for (int i = 0; i < m.getRows(); ++i){
		for (int j =0; j < m.getColumns(); ++j){
			array[cont] = m.matriz[i][j];
			cont++;
		}
	}
	
	return array;
}


Matrix ChangeMatriz(int r_size, int c_size, double *array) {
	int cont = 0;

	Matrix result;
	result.setRows(r_size);
	result.setColumns(c_size);
	
	result.matriz.resize(r_size);
	for ( int i = 0; i < r_size; i++ ){
		result.matriz[i].resize(c_size);
		for ( int j = 0; j < c_size; j++ ){
			result.matriz[i][j] = array[cont];
			cont++;
		}
	}
	return result;
}

double* VecXMat(Matrix &m, double* v){
	double *result;
	result = new double[m.getRows()];

	for (int i = 0; i < m.getRows(); ++i){
			result[i] = 0.0;
		for (int j = 0; j < m.getColumns(); ++j){
			result[i] += m.matriz[i][j] * v[j];
		}
	}
	return result;	
}


void SaveResult(const char* filename, double * result, int length){
	
		int aux3=1;
		ofstream out (filename, ios::binary | ios::out);
		out.write((char*)&length, sizeof(int));
		out.write((char*)&aux3, sizeof(int));
		for (int i = 0; i < length; i++)
			out.write((char*)&result[i], sizeof(double));
		out.close();
		
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


