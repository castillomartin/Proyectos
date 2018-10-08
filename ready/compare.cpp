#include "Matriz.h"
#include <math.h>
#include <limits>
using namespace std;

bool Comparer(MatrizP * m);

int main(int argc, char* argv[])
{
    MatrizP * m;
    m = new Matriz<float>();

    m->ReadM1(argv[1]);
    m->ReadM2(argv[2]);
	
	if(Comparer(m))
		puts("equals");
	else
		puts("different");
	return 0;
}


bool Comparer(MatrizP * m){
	
	int i,j;
	for(i=0;i<m->getf1();i++)
		for(j=0;j<m->getc1();j++){
			if(abs(m->getm1pos(i,j) - m->getm1pos(i,j)) > numeric_limits<float>::epsilon())
				return false;
		} 
		return true;
}
