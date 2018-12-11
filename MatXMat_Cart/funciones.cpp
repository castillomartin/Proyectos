#include <fstream>
#include <iostream>
#include <cstdlib>

using namespace std;

void Print(char * fil);
void Generate(int n,int m, char * f);

int main(int argc, char* argv[])
{	
	if(argc == 2)
	Print(argv[1]);
	
	else if(argc == 4){
	int n = atoi(argv[1]);
	int m = atoi(argv[2]);
	Generate(n,m,argv[3]);
	}
	
	else {
		cout<<"Need 1 or 3 argument"<<endl;
	}
	
	return 0;
}


void Print(char * fil){
	
	int n, m;
	fstream file(fil, ios::binary | ios::in);
	
	if (file.is_open()) {

	file.read((char*)&n, sizeof(int));
	file.read((char*)&m, sizeof(int));
	cout << n << endl;
	cout << m << endl;


	double val;
	for (int i = 0; i<n; i++) {
		for (int j = 0; j<m; j++) {
			file.read((char*)&val, sizeof(double));
			cout << val << "  ";
		}
		cout << endl;
	}
	
	file.close();
	}
    else  
		cout<<"Don't exist File"<<endl;

}


void Generate(int n,int m,char * f){
	double x;
		ofstream fout(f, ios::binary | ios::out);
		srand(time(0));
		if (fout.is_open()) {
			fout.write((char*)&n, sizeof(int));
			fout.write((char*)&m, sizeof(int));
			for (int i = 0; i<n; ++i) {
				for (int j = 0; j<m; ++j) {
					x = (double)(rand()%10) / 10;
					fout.write((char*)&x, sizeof(double));
				}
			}
		}
		fout.close();
}



