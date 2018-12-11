#include "fun.h"
using namespace std;

MPI_Comm ij_comm;
int axis,N, length_block,aux,coord[3];

double * ReadBlock (char *file_name);

int main(int argc, char **argv) {

	int my_size, my_grid_rank, kk;
	double t_total = 0.0, t_max = 0.0,t_start = 0.0, t_end = 0.0;
  
	double *A1, *B1, *C1;

	MPI_Comm grid_comm, i_comm, j_comm, k_comm;
	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &my_size);

	axis = atoi(argv[4]);
  
	if (my_size == 1) {
		Matrix a = ReadFile(argv[1]);
		Matrix b = ReadFile(argv[2]);
	
		//TIME
		t_start = MPI_Wtime();
	
		Matrix c = Mat_x_Mat(a, b);
	
		t_end = MPI_Wtime() - t_start;
		//END TIME

		t_max = t_end;
		t_total = t_end;
		ofstream out1("base_1", ios::out);
		out1 << t_max << " " << t_total << " " << my_size << endl;
		out1.close();

		//Save Result A x B = C
		SaveResult(argv[3],c);
	
		MPI_Finalize();
		return 0;
	}

	int dims[3] = {axis, axis, axis};
	int period[3] = {0, 0, 0};
	int reorder = 0;

	MPI_Cart_create(MPI_COMM_WORLD, 3, dims, period, reorder, &grid_comm);

	int dims_ij[3] = {1, 1, 0};
	int dims_i[3] = {1, 0, 0};
	int dims_j[3] = {0, 1, 0};
	int dims_k[3] = {0, 0, 1};

	MPI_Cart_sub(grid_comm, dims_ij, &ij_comm);
	MPI_Cart_sub(grid_comm, dims_i, &i_comm);
	MPI_Cart_sub(grid_comm, dims_j, &j_comm);
	MPI_Cart_sub(grid_comm, dims_k, &k_comm);

	MPI_Comm_rank(grid_comm, &my_grid_rank);


	MPI_Cart_coords(grid_comm, my_grid_rank, 3, coord);

	if (coord[2] == 0) {
		A1 = ReadBlock(argv[1]);
		B1 = ReadBlock(argv[2]);
	}

	MPI_Barrier(grid_comm);
	MPI_Bcast(&length_block, 1, MPI_DOUBLE, 0, k_comm);

	MPI_Barrier(grid_comm);
	MPI_Bcast(&aux, 1, MPI_INT, 0, k_comm);

	if (coord[2] != 0) {
		A1 = new double[length_block];
		B1 = new double[length_block];
	}

	int sendCoord[3], recvCoord[3], bcoord[1];

	int i = coord[0], j = coord[1], k = coord[2];
	if (k == 0 && j != 0) {
		int rrank;
		recvCoord[0] = i;
		recvCoord[1] = j;
		recvCoord[2] = j;
		MPI_Cart_rank(grid_comm, recvCoord, &rrank);
		MPI_Send(A1, length_block, MPI_DOUBLE, rrank, my_grid_rank, grid_comm);
	}

	if (j == k && j != 0) {
		MPI_Status status;
		int srank;
		sendCoord[0] = i;
		sendCoord[1] = j;
		sendCoord[2] = 0;
		MPI_Cart_rank(grid_comm, sendCoord, &srank);
		MPI_Recv(A1, length_block, MPI_DOUBLE, srank, MPI_ANY_TAG, grid_comm,&status);
	}

	int rank;

	bcoord[0] = coord[2];
	MPI_Cart_rank(j_comm, bcoord, &rank);
	MPI_Bcast(A1, length_block, MPI_DOUBLE, rank, j_comm);

	if (k == 0 && i != 0) {
		int rrank;
		recvCoord[0] = i;
		recvCoord[1] = j;
		recvCoord[2] = i;
		MPI_Cart_rank(grid_comm, recvCoord, &rrank);
		MPI_Send(B1, length_block, MPI_DOUBLE, rrank, my_grid_rank, grid_comm);
	}
	if (i == k && i != 0) {
		MPI_Status status;
		int srank;
		sendCoord[0] = i;
		sendCoord[1] = j;
		sendCoord[2] = 0;
		MPI_Cart_rank(grid_comm, sendCoord, &srank);
		MPI_Recv(B1, length_block, MPI_DOUBLE, srank, MPI_ANY_TAG, grid_comm,&status);
	}

	bcoord[0] = coord[2];
	MPI_Cart_rank(i_comm, bcoord, &rank);
	MPI_Bcast(B1, length_block, MPI_DOUBLE, rank, i_comm);
	int aux1 = aux / axis;
	Matrix a = ChangeMatriz(aux1, A1);
	Matrix b = ChangeMatriz(aux1, B1);
 
	//time
	t_start = MPI_Wtime();
	
	Matrix c = Mat_x_Mat(a, b);
	
	t_end = MPI_Wtime() - t_start;
	//end time

	
	if (my_grid_rank != 0) {
		MPI_Send(&t_end, 1, MPI_DOUBLE, 0, rank, grid_comm);
	}


	if (my_grid_rank == 0) {
		t_total += t_end;
		if (t_end > t_max)
			t_max = t_end;
		for (int i = 0; i < my_size - 1; ++i) {
			MPI_Recv(&t_end, 1, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, grid_comm, &status);
			if (t_end > t_max) 
				t_max = t_end;
		t_total += t_end;
		}
	}

	C1 = ChangeArray(c);

	double *aux4;

	aux4 = new double[length_block];
	
	//REDUCE RESULT BLOCK 
	for (int i = 0; i < length_block; i++) aux4[i] = 0;
		MPI_Reduce(C1, aux4, length_block, MPI_DOUBLE, MPI_SUM, 0, k_comm);


	if ( coord[2] == 0){
	 	
		MPI_Status status;
		int lenght = N * axis;
		int aux6 = (coord[0] * lenght * N + coord[1] * N);
		int borderoffset = sizeof(double) *  aux6;
		
		MPI_File fh;
		MPI_File_open(ij_comm, argv[3], MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL,&fh);	
		
		//Save Data Block File
		if (coord[0] == coord[1] == coord[2] == 0) {
			MPI_File_write_at(fh, 0, &lenght, 1, MPI_INT, &status);
			MPI_File_write_at(fh, 4, &lenght, 1, MPI_INT, &status);
		}
		
		borderoffset+=  2 * sizeof(int);
		
		for (int i = 0; i < length_block; i++) {
			int block =  sizeof(double) * ((i / N) * lenght + i % N);
			MPI_File_write_at(fh, borderoffset + block, &aux4[i], 1, MPI_DOUBLE,&status);
		}
		
	}
	


	if (my_grid_rank == 0) {
		int rcol = 1;
		
		if (my_size == 1) {
			ofstream out1("base_1", ios::out);
			out1 << t_max << " " << t_total << " " << my_size << endl;
			out1.close();
		} else {
			//Save Max Time, Total Time, Efficience, SpeedUp 
			SaveComplement(t_max,  my_size,  t_total);
		}
		
	}


	MPI_Finalize();
	return 0;
}


double *ReadBlock (char *file_name){
	
	
	ifstream in(file_name, ios::binary | ios::in);
	int lenght;
	double *aux2, aux3;
	in.read((char*)&lenght, sizeof(int));
	in.read((char*)&lenght, sizeof(int));
	aux = lenght;

	N = lenght/axis;
	length_block = N * N;
	aux2 = new double [length_block];

	int borderoffset = sizeof(double) * (coord[1] * N + N * lenght * coord[0]) + 2*sizeof(int);

	in.seekg(borderoffset, in.beg);
	for ( int i = 0; i < N; i++ ){
		for ( int j = 0; j < N; j++ ){
			in.read((char*)&aux3, sizeof(double));
			aux2[i*N + j] = aux3;
		}
		borderoffset = sizeof(double)*(lenght - N);
		in.seekg(borderoffset, in.cur);
	}	
   
	return aux2;
}



