#include "fun.h"
using namespace std;


int main(int argc, char *argv[])
{
	
	int my_size, my_rank;
	int error,left, right, n_rows, m_columns, rows, columns;
	double t_total = 0.0, t_max = 0.0,t_start = 0.0, t_end = 0.0;
	string F_A, F_B, F_C;
	bool flag;

	F_A = argv[1];
	F_B = argv[2];
	F_C = argv[3];

	Matrix matrix1;
	Matrix vector1;
	
	if ((error = MPI_Init(&argc, &argv)) != MPI_SUCCESS){
		cout << " MPI_Init error " << endl;
		MPI_Abort(MPI_COMM_WORLD, error);
	}

	MPI_Comm_size(MPI_COMM_WORLD, &my_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	
	double *matrix2,*matrix3, *vector2, *vector3,*aux,*result;
	
	if (my_rank == 0){
		
		//READ FILE
		matrix1 = ReadFile(F_A);
		vector1 = ReadFile(F_B);
		
		n_rows = matrix1.getRows();
		m_columns = matrix1.getColumns();

		flag = 1;
		left = my_rank * m_columns / my_size;
		right = (my_rank + 1) * m_columns / my_size - 1;
		columns = right - left + 1 ;
			
		if (matrix1.getRows() >= matrix1.getColumns()){
			flag = 0;
			left = my_rank * n_rows / my_size;
			right = (my_rank + 1) * n_rows / my_size - 1;
			rows = right - left + 1 ;
		}
		
	}

	MPI_Bcast(&flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
		
	
	MPI_Bcast(&n_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&m_columns, 1, MPI_INT, 0, MPI_COMM_WORLD);
	
	
	//IF ROWS > COLUMNS MPI_Gather
	if ( flag == 0){
		
		MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Status status;
	
		if ( my_rank == 0 ){
			matrix2 = ChangeArray(matrix1);
			vector2 = ChangeArray(vector1);
		}
		else {
			vector2 = new double[m_columns];
		}

		MPI_Bcast(vector2, m_columns, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
			
		matrix3 = new double [m_columns * rows];

		MPI_Scatter(matrix2, m_columns * rows, MPI_DOUBLE, matrix3, m_columns * rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		Matrix aux2 = ChangeMatriz(rows, m_columns, matrix3); 
		
		//TIME Мultiplication
		t_start = MPI_Wtime();
		
		result = VecXMat(aux2, vector2);
		
		t_end = MPI_Wtime() - t_start;
		//END TIME
		
		if( my_rank != 0){
			MPI_Send(&t_end, 1, MPI_DOUBLE, 0, my_rank, MPI_COMM_WORLD);
		}
			
		if ( my_rank == 0 ){
			t_total+=t_end;
			aux = new double [n_rows];
			if( t_end > t_max)
				t_max = t_end;
			for (int i = 0; i < my_size -1; ++i){
				MPI_Recv(&t_end, 1, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
				if( t_end > t_max)
					t_max = t_end;
				t_total += t_end;
			}

		}
		
		//MPI_Reduce(&t_end, &t_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
		//MPI_Reduce(&t_end, &t_total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Gather(result, rows , MPI_DOUBLE, aux, rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		
	}
		
	//IF ROWS < COLUMNS MPI_Reduce
	else {
		
		MPI_Bcast(&columns, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Status status;
	
		if ( my_rank == 0 ){
			matrix2 = ChangeArray(matrix1);
			vector2 = ChangeArray(vector1);
		}
		else {
			vector2= new double[m_columns];
		}
			
		matrix3 = new double [n_rows * columns];
		vector3 = new double [columns];

		MPI_Scatter(matrix2, n_rows * columns, MPI_DOUBLE, matrix3, n_rows * columns, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Scatter(vector2, columns, MPI_DOUBLE, vector3, columns, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			
		Matrix aux2 = ChangeMatriz(n_rows, columns, matrix3);

		//TIME Мultiplication
		t_start = MPI_Wtime();
		
		result = VecXMat(aux2, vector3);
		
		t_end = MPI_Wtime() - t_start;
		//END TIME
			
		if(my_rank != 0){
			MPI_Send(&t_end, 1, MPI_DOUBLE, 0, my_rank, MPI_COMM_WORLD);
		}


		if ( my_rank == 0 ){
			t_total+=t_end;
			if (t_end > t_max)
				t_max = t_end;
			aux = new double [n_rows];
			for (int i = 0; i < my_size -1; ++i){
				MPI_Recv(&t_end, 1, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
				if( t_end > t_max)
					t_max = t_end;
				t_total += t_end;
			}
			
		}
		
		//MPI_Reduce(&t_end, &t_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
		//MPI_Reduce(&t_end, &t_total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(result, aux, n_rows, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		
	}
	
	if (my_rank ==0){
		
		//Save SpeedUp, Efficiency, MaxTime, TotalTime, NumProcces
		if (my_size > 1) 
			SaveComplement(t_max, my_size, t_total);
		
		else {
			ofstream out1 ("base_1", ios::out);
			out1 << t_max << " " << t_total << " " << my_size<<endl;
			out1.close();
		}
		
		//Save Result Multiplication AxB = C
		SaveResult(F_C.c_str(), aux, n_rows);
		
	}
	MPI_Finalize();
	return 0;
}


