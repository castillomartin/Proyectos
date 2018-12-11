#include <iostream>
#include <vector>

using namespace std;

class Matrix
{
	private:
		int columns;
		int rows;
		
	public:
		Matrix(){}
		Matrix( vector < vector<double> > pmatriz , int prows, int pcolumns){
			matriz = pmatriz;
			rows = prows;
			columns = pcolumns;
		}
		~Matrix(){}

		int getColumns()const {return columns;}
		
		int getRows() const {return rows;}

		void setColumns(int columns) {this->columns = columns;}
		
		void setRows(int rows){this->rows = rows;}
		
		vector < vector<double> > matriz; 
};




