#include <iostream>
#include <cstring>
#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include <cstdlib>
#include <ctime>
#include <random> 
#include <chrono>
#include <limits>
#include <omp.h>
using namespace std;

string col_name[4] = {"x1", "x2", "x3", "x4"};
string row_name[6] = {"y1", "y2", "y3", "y4", "y5", "z"};

__global__ void update_row(float* matrix, int rows, int cols, int pivotRow, int pivotColume, float pivot){
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int j = bid * blockDim.x + tid;

    if ((j < cols) && (j != pivotColume)){
        matrix[pivotRow * cols + j] = -matrix[pivotRow * cols + j] * pivot;
    }

}

__global__ void update_col(float* matrix, int rows, int cols, int pivotRow, int pivotColume, float pivot){
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int i = bid * blockDim.x + tid;

    if ((i < rows) && (i != pivotRow)){
        matrix[i * cols + pivotColume] = matrix[i * cols + pivotColume] * pivot;
    }

}

__global__ void update_field(float* matrix, int rows, int cols, int pivotRow, int pivotColume){
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;

    if ((i < rows) && (i != pivotRow))
	if ((j < cols) && (j != pivotColume)){
             matrix[i * cols + j] = matrix[i * cols + j] + matrix[i * cols + pivotColume] * matrix[pivotRow * cols + j];
        }

}

class Simplex{

    private:
        //stores coefficients of all the variables
        float *A; // the matrix that stores the original data without kexi and w
        float *Aw;  // the matrix that stores the auxiliary data with kexi and w
	int rows, cols, rowsW, colsW;

        int x0;
        //stores constants of constraints
        //std::vector<float> B;
        //stores the coefficients of the objective function
        //std::vector<float> C;

        float maximum;
        int lastPivotC_w;
        bool isUnbounded;

        int printFreq;

        // record the performance of exchange
        double total_t_ex;
        int total_n_ex;


    public:
        Simplex(vector <vector<float> > matrix, int r, int c){
            maximum = 0;
            isUnbounded = false;
            lastPivotC_w = 0;
            printFreq = 1000;
            total_t_ex = 0.0;
            total_n_ex = 0;
            rows = r;
            cols = c;
	    rowsW = r + 1;
	    colsW = c + 1;
	
	    cudaMallocManaged(&A, rows * cols * sizeof(float));
	    cudaMallocManaged(&Aw, rowsW * colsW * sizeof(float));

	    //A = new float[rows * cols];
	    //Aw = new float[rowsW * colsW];

            cudaMallocManaged(&A, rows * cols * sizeof(float));
            cudaMallocManaged(&Aw, rowsW * colsW * sizeof(float));

            for(int i= 0;i<rows;i++){             //pass A[][] values to the metrix
                for(int j= 0; j< cols;j++ ){
                    A[i * cols + j] = matrix[i][j];
                    Aw[i * colsW + j+1] = matrix[i][j];
                }
            }

            // for(int i=0; i< c.size() ;i++ ){      //pass c[] values to the B vector
            //     C[i] = c[i] ;
            // }
            // for(int i=0; i< b.size();i++ ){      //pass b[] values to the B vector
            //     B[i] = b[i];
            // }
            constructW();
        }

        void constructW(){
            cout << "--------Construct auxiliary linear program with kexi and w.--------" << endl;
            cout << "A:" << endl;
            printM(A, rows, cols);
            x0 = A[0 * rows + cols - 1];
            for (int i = 0 ; i < rows - 1; i ++){
                if(A[i * rows + cols - 1] < x0)
                    x0 = A[i * rows + cols - 1];

                // set kexi
                Aw[i * colsW] = 1.0;
            }
            x0 = -x0;

            // set kexi
            Aw[(rows-1) * colsW + 0] = 0.0;

            // set W
            Aw[rows * colsW + 0] = -1.0;
            Aw[rows * colsW + cols] = -x0;

            // set the final column
            for (int i = 0 ; i < rows - 1; i ++){
                Aw[i * colsW + cols] += x0;
            }

            cout << "A_w:" << endl;
            printM(Aw, rowsW, colsW);
        }


        void ExchangeStep(float *matrix, int rows, int cols, int pivotRow, int pivotColume){

            auto start = std::chrono::high_resolution_clock::now();

            // update the pivot element
            matrix[pivotRow * cols + pivotColume] = 1.0 / matrix[pivotRow * cols + pivotColume];
            float pivot = matrix[pivotRow * cols + pivotColume];
            // update the pivot row
            //#pragma omp parallel for
            //for(int j = 0; j < cols; j++){
            //    if (j != pivotColume)
            //        matrix[pivotRow * cols + j] = -matrix[pivotRow * cols + j] * matrix[pivotRow * cols + pivotColume];
            //}
	    update_row<<<(cols - 1)/ 256 + 1, 256>>>(matrix, rows, cols, pivotRow, pivotColume, pivot);
	    cudaDeviceSynchronize();

            // update the field
            //#pragma omp parallel for
            //for(int i = 0; i < rows; i++)
            //    for(int j = 0; j < cols; j++){
            //        if ((i != pivotRow) && (j != pivotColume))
            //            matrix[i * cols + j] = matrix[i * cols + j] + matrix[i * cols + pivotColume] * matrix[pivotRow * cols + j];
            //}
	    dim3 grid((cols - 1)/32 + 1, (rows - 1)/32 + 1);
	    dim3 block(32, 32);
	    update_field<<<grid, block>>>(matrix, rows, cols, pivotRow, pivotColume);
	    cudaDeviceSynchronize();
	    //cudaError_t cudastatus = cudaGetLastError();
	    //printf("%s\n", cudaGetErrorString(cudastatus));

            // update the pivot column
            //#pragma omp parallel for
            //for(int i = 0; i < rows;i++){
            //    if (i != pivotRow)
            //        matrix[i * cols + pivotColume] = matrix[i * cols + pivotColume] * matrix[pivotRow * cols + pivotColume];
            //}
	    update_col<<<(rows - 1)/256 + 1, 256>>>(matrix, rows, cols, pivotRow, pivotColume, pivot);
	    cudaDeviceSynchronize();

            auto finish = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = finish - start;
            total_t_ex += elapsed.count();
            total_n_ex++;
        }

        void Reduce(){
            cout << "--------Remove kexi column and w row.--------" << endl;
            // exchange the free kexi with the last used pivot column
            ExchangeStep(Aw, rowsW, colsW, rowsW - 3, lastPivotC_w);
            for (int i = 0 ; i < rowsW ; i++)
                Aw[i * colsW + colsW - 1] = Aw[i * colsW + colsW - 1] + Aw[i * colsW + lastPivotC_w] * (-x0);

            // remove kexi column and w row
            for (int i = 0 ; i < rowsW - 1;i++){
                for (int j = 0 ; j < lastPivotC_w; j++)
                    A[i * cols + j] = Aw[i * colsW + j];
                for (int j = lastPivotC_w + 1; j < colsW; j++)
                    A[i * cols + j - 1] = Aw[i * colsW + j];
            }
            cout << "Aw:" << endl;
            printM(Aw, rowsW, colsW);
            cout << "A:" << endl;
            printM(A, rows, cols);
        }

        void SimplexW(){
            cout << "--------Conduct simplex for w=max until w>=0.--------" << endl;
            bool finished = false;
            int iter = 0;
            while(!finished){
                if (iter % printFreq == 0)
                    cout << "\tIter " << iter << "." << endl;
                if (iter > 20000)
                    break;
                if(Aw[(rowsW - 1) * colsW + (colsW - 1)] >= 0){
                    finished = true;
                    printf("\tfinal w:%f.\n", Aw[(rowsW - 1) * colsW + (colsW - 1)]);
                    continue;
                }
                int q = 0;
                while ((Aw[(rowsW - 1) * colsW + q] <= 0) && (q < (colsW - 1))){
                    q++;
                }

                finished = (q == (colsW - 1));

                if (!finished){

                    int p = -1;
                    if(rowsW < 1000000){
                        for(int l = 0; l < rowsW - 3; l++){
                            if (Aw[l * colsW + q] < 0)
                                if (p == -1)
                                    p = l;
                                else if((Aw[l * colsW + colsW - 1] / Aw[l * colsW + q]) > (Aw[p * colsW + colsW - 1] / Aw[p * colsW + q]))
                                    p = l;
                        }
                    }
                    else{

                        float max_local[50];
                        int max_idx[50];

                        int num_thrs = 1;
                        #pragma omp parallel shared(max_local, max_idx)
                        {
                            num_thrs = omp_get_num_threads();
                            int id;
                            int n, start, stop;
                            int amount = rowsW - 3;
                            n = amount / num_thrs;

                            id = omp_get_thread_num();

                            start = id * n;
                            if(id != (omp_get_num_threads() - 1))
                                stop = start + n;
                            else
                                stop = amount;

                            max_idx[id] = -1;
                            //printf("debug 1\n");
                            max_local[id] = -9999999999.0;
                            for(int l = start; l < stop; l++)
                                if (Aw[l * colsW + q] < 0)
                                    if ((Aw[l * colsW + colsW - 1] / Aw[l * colsW + q]) > max_local[id])
                                    {
                                        max_local[id] = Aw[l * colsW + colsW - 1] / Aw[l * colsW + q];
                                        max_idx[id] = l;
                                    }
			}
                        for (int i = 1 ; i < num_thrs; i ++)
                            if ((max_idx[i] != -1) && (max_local[i] > max_local[0])){
		        	max_idx[0] = max_idx[i];
		        	max_local[0] = max_local[i];
		            }
		        p = max_idx[0];

                    }
                    finished = (p == -1);
                    if (finished){
                        cout << "Error!" << endl;
                        exit (EXIT_FAILURE);
                    }
                    else{
                        if (iter % printFreq == 0)
                            printf("\tbefore exchange: w:%f.\n", Aw[(rowsW - 1) * colsW + (colsW - 1)]);
                        ExchangeStep(Aw, rowsW, colsW, p, q);
                        lastPivotC_w = q;
                        if (iter % printFreq == 0)
                            printf("\tafter exchange: w:%f.\n", Aw[(rowsW - 1) * colsW + (colsW - 1)]);
                        printM(Aw, rowsW, colsW);
                    }

                }
                iter++;

            }

        }

        void SimplexN(){
            bool finished = false;
            int iter = 0;
            while(!finished){
                if (iter % printFreq == 0)
                    cout << "\tIter " << iter << "." << endl;
                if (iter > 20000)
                    break;
                int q = 0;
                while ((A[(rows - 1) * cols + q] <= 0) && (q < (cols - 1))){
                    q++;
                }

                finished = (q == (cols - 1));

                if (!finished){
                    int p = -1;
                    for(int l = 0; l < rows - 1; l++){
                        if (A[l * cols + q] < 0)
                            if (p == -1)
                                p = l;
                            else if((A[l * cols + cols - 1] / A[l * cols + q]) > (A[p * cols + cols - 1] / A[p * cols + q]))
                                p = l;
                    }
                    finished = (p == -1);
                    if (finished){
                        cout << "Error!" << endl;
                        exit (EXIT_FAILURE);
                    }
                    else{
                        if (iter % printFreq == 0)
                            printf("\tchoose pivot: %d:%f.\n", q, A[(rows - 1) * cols + q]);
                        ExchangeStep(A, rows, cols, p, q);
                        lastPivotC_w = q;
                        if (iter % printFreq == 0)
                            printf("\tafter exchange: %d:%f.\n", q, A[(rows - 1) * cols + q]);
                        printM(A, rows, cols);
                    }

                }
                iter++;

            }

        }

        void freeEliminate(int col){
            cout << "--------Eliminate kexi.--------" << endl;
            int chosen_row = 0;
            int min = 999999999;
            if (Aw[(rowsW - 1) * colsW + col] > 0){
                for (int i = 0; i < rowsW - 2; i++){
                    if (Aw[i * colsW + col] < 0){
                        float bk_it = Aw[i * colsW + colsW - 1] / Aw[i * colsW + col];
                        if (bk_it < min){
                            min = bk_it;
                            chosen_row = i;
                        }
                    }
                
                }
            }
            else{
                for (int i = 0; i < rowsW - 2; i++){
                    if (Aw[i * colsW + col] > 0){
                        float bk_it = Aw[i * colsW + colsW - 1] / Aw[i * colsW + col];
                        if (bk_it < min){
                            min = bk_it;
                            chosen_row = i;
                        }
                    }
                
                }
            }

            cout << "chosen row:" << chosen_row << endl;
            ExchangeStep(Aw, rowsW, colsW, chosen_row, col);
            printM(Aw, rowsW, colsW);
            // change the position of kexi row
            if (chosen_row != rowsW - 3){
                float *t = new float[colsW];
		for (int j = 0 ; j < colsW; j++)
                    t[j] = Aw[(rowsW - 3) * colsW + j];
		for (int j = 0 ; j < colsW; j++)
                    Aw[(rowsW - 3) * colsW + j] = Aw[chosen_row * colsW + j];
		for (int j = 0 ; j < colsW; j++)
                    Aw[chosen_row * colsW + j] = t[j];
            }
        }

        void printM(float *matrix, int rows, int cols){
            if(rows > 30)
                return;
            for(int i= 0;i<rows;i++){             
                for(int j= 0; j< cols;j++ ){
                    cout << matrix[i * cols + j] << "\t";
                }
                cout << "\n";    
            }
            cout << "\n";
        }

        void printZ(float *matrix, int rows, int cols){
            for(int j= 0; j< cols;j++ ){
                cout << matrix[(rows - 1) * cols + j] << "\t";
            }
            cout << "\n";
        }

        void printP(){
            printf("ExchangeStep() total number: %d.\n", total_n_ex);
            printf("ExchangeStep() total time: %f s.\n", total_t_ex);
            printf("ExchangeStep() average time: %f s.\n", total_t_ex / total_n_ex);
        }

};

void rand_init(float **input, int rows, int cols){

    //srand(time(NULL) % 256);
    srand(8);
    //srand(123423424);
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0, 1.0);

    for (int i=0;i<rows;i++)
        for(int j = 0; j < cols; j++)
            //input[i][j] = distribution(generator) + static_cast <float> (rand()) / static_cast <float> (RAND_MAX) + 0.1;
            input[i][j] = distribution(generator);

    for (int j = 0; j < cols; j++)
        input[0][j]= static_cast <float> (rand()) / static_cast <float> (RAND_MAX) + 0.1;

    float *p = new float[cols - 1];
    for (int j = 0; j < cols - 1; j++)
        p[j]= static_cast <float> (rand()) / static_cast <float> (RAND_MAX) + 0.01;

    for (int i = 0; i < rows - 1; i++){
        input[i][cols - 1] = 0;
        for (int j = 0; j < cols - 1;j++)
            input[i][cols - 1] += input[i][j] * p[j];
        input[i][cols - 1] -= static_cast <float> (rand()) / static_cast <float> (RAND_MAX); 
    }

    for (int j = 0 ; j < cols - 1;j++)
        input[rows - 1][j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    input[rows - 1][cols -1] = 0.0;

}

int main()
{
    
    //int rows = 6;
    //int cols = 5;
    //float input[6][5] = {{1, 1, 0, 0, -7}, 
    //               {0, 0, 1, 1, -9},
    //               {-1, 0, 0, 0, 5},
    //               {0, -1, -1, 0, 10},
    //               {0, 0, 0, -1, 7},
    //               {-10, -8, -9, -12, 0}};

    //int rows = 4;
    //int cols = 4;
    //float input[4][4] = {{-2, -1, -1, 180}, 
    //               {-1, -3, -2, 300},
    //               {-2, -1, -2, 240},
    //               {6, 5, 4, 0}};
    
    //int rows = 5;
    //int cols = 3;
    //float input[5][3] = {{-1, -1, 18}, 
    //               {-1, 0, 11},
    //               {0, -1, 10},
    //               {1, 1, -9},
    //               {1, 3, -229}};
    
    int rows = 1000;
    int cols = rows / 2;
    float **input = new float*[rows];
    for (int i = 0; i < rows; i++)
        input[i] = new float[cols];

    rand_init(input, rows, cols);
    cout << "finish initialization." << endl;
    
    vector <vector<float> > A(rows, vector<float>(cols, 0));

    for(int i=0;i<rows;i++){         //make a vector from given array
        for(int j=0; j<cols;j++){
            A[i][j] = input[i][j];
        }
    }    


    Simplex simplex(A, rows, cols);

    auto start = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed;
    
    simplex.freeEliminate(0);
    auto fe = std::chrono::high_resolution_clock::now();
    elapsed = fe - start;
    printf("freeEliminate() time: %f s.\n", elapsed.count());
    
    simplex.SimplexW();
    auto sw = std::chrono::high_resolution_clock::now();
    elapsed = sw - fe;
    printf("SimplexW() time: %f s.\n", elapsed.count());
    
    simplex.Reduce();
    auto rd = std::chrono::high_resolution_clock::now();
    elapsed = rd - sw;
    printf("Reduce() time: %f s.\n", elapsed.count());
    
    simplex.SimplexN();
    auto sn = std::chrono::high_resolution_clock::now();
    elapsed = sn - rd;
    printf("SimplexN() time: %f s.\n", elapsed.count());
    
    simplex.printP();

    auto finish = std::chrono::high_resolution_clock::now();
    elapsed = finish - start;   
    printf("Total execution time: %f s.\n", elapsed.count());
}

