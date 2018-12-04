#include "fun.h"
using namespace std;

int main(int argc, char *argv[])
{
	int error,left, right,length,n_rows, m_columns;
	int my_size, my_rank;
	double t_total = 0.0, t_max = 0.0,t_start = 0.0, t_end = 0.0;
	string F_A, F_B, F_C;


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
	
	double *matrix2, *vector2,*matrix3,*aux,*result;
	
	if (my_rank == 0){
		
		matrix1 = ReadFile(F_A);
		vector1 = ReadFile(F_B);
		
		n_rows = matrix1.getRows();
		m_columns = matrix1.getColumns();

		left = my_rank * n_rows / my_size;
		right = (my_rank + 1) * n_rows / my_size - 1;
		length = right - left + 1 ;
		
	}

	MPI_Bcast(&n_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&m_columns, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&length, 1, MPI_INT, 0, MPI_COMM_WORLD);
			
	MPI_Barrier(MPI_COMM_WORLD);
		
	
	if ( my_rank == 0 ){
		matrix2 = ChangeArray(matrix1);
		vector2 = ChangeArray(vector1);
	}
	else {
		vector2 = new double[m_columns];
	}

	MPI_Bcast(vector2, m_columns, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
			
	matrix3 = new double [m_columns * length];

	MPI_Scatter(matrix2, m_columns * length, MPI_DOUBLE, matrix3, m_columns * length, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	Matrix aux2 = ChangeMatriz(length, m_columns, matrix3); 
		
	//start time
	t_start = MPI_Wtime();

	result = VecXMat(aux2, vector2);  
		
	t_end = MPI_Wtime() - t_start;
	//end time
		
	if( my_rank != 0){
		MPI_Send(&t_end, 1, MPI_DOUBLE, 0, my_rank, MPI_COMM_WORLD);
	}
			
	delete[] matrix3;
	delete[] vector2;

	if (my_rank == 0) {
		delete[] matrix2;
	}


	MPI_Status status;


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
	
	
	MPI_Gather(result, length , MPI_DOUBLE, aux, length, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			
		delete[] result;

	
	
	if (my_rank ==0){
		
		if (my_size > 1) {
			
			float t_t_1;
			double speedup, efficiency;
			ifstream fin ("base_1", ios::in);
			fin >> t_t_1;
			fin.close();
			speedup = (double)t_t_1 / t_max;
			efficiency = speedup / my_size;
			ofstream fout("result", ios::app);
			fout << my_size << " " << t_total << " " << t_max << " " << speedup << " " << efficiency <<endl; 
			fout.close();
			
		}
		else {
			ofstream fout1p ("base_1", ios::out);
			fout1p << t_max << " " << t_total << " " << my_size<<endl;
			fout1p.close();
		}
		
		
		int one=1;
		const char* out_cstr = F_C.c_str();
		ofstream fvec (out_cstr, ios::binary | ios::out);
		fvec.write((char*)&n_rows, sizeof(int));
		fvec.write((char*)&one, sizeof(int));
		for (int i = 0; i < n_rows; i++)
			fvec.write((char*)&aux[i], sizeof(double));
		fvec.close();
		
	}
	MPI_Finalize();
	delete[] aux;
	return 0;
}
