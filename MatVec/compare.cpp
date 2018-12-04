#include "Matrix.h"
#include <iostream>
#include <fstream>

using namespace std;

Matrix ReadFile(const string &strr){

	int n, m;
	double val;
	Matrix var;

	const char* cstr = strr.c_str();
	fstream f1(cstr, ios::binary | ios::in);
	f1.read((char*)&n, sizeof(int));
	f1.read((char*)&m, sizeof(int));
	var.setRows(n);
	var.setColumns(m);

	var.matriz.resize(n);
	for (int i = 0; i < n; ++i){
		var.matriz[i].resize(m);
		for (int j = 0; j < m; j++){
			f1.read((char*)&val, sizeof(double));
			var.matriz[i][j] = val;
		}
	}
	f1.close();
	return var;
}



int main(int argc, char* argv[]){
	if(argc < 3){
		cout << "use ./compare A B" << endl;
		return 0;
	}
	
	bool f = true;
	const double eps = 1e-6;
	int n,m;
	
	string F_A, F_B;
	F_A = argv[1];
	F_B = argv[2];
	Matrix matrix1,matrix2;
	
	matrix1 = ReadFile(F_A);
	matrix2 = ReadFile(F_B);
	n = matrix1.getRows();
	m = matrix1.getColumns();
	for(int i=0;i<n;i++)
		for(int j=0;j<m;j++){
			if(matrix1.matriz[i][j] != matrix2.matriz[i][j]){
				f = false;
				break;
			}
		}
	
	if(f)
		cout<<"equals"<<endl;
	else cout<<"not equals"<<endl;
	
	return 0;
}
