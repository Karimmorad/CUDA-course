#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#define S_LEN 512
#define N 1000

double get_time() // function to get the time of day in seconds
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int max(int n1, int n2, int n3, int n4) {
    int tmp1, tmp2;

    tmp1 = n1 > n2 ? n1 : n2;
    tmp2 = n3 > n4 ? n3 : n4;
    tmp1 = tmp1 > tmp2 ? tmp1 : tmp2;

    return tmp1;
}

int main(int argc, char *argv[]){

    srand(time(NULL));
    int ins=-2, del=-2;

    char query[N][S_LEN];
    char string[N][S_LEN];
    int res[N];

    char alphabet[5] = {'A','C','G','T','N'};
    int mat[S_LEN+1][S_LEN+1];
    for(int n = 0; n < N; n++){
        for(int i = 0; i < S_LEN; i++){
            query[n][i]=alphabet[rand()%5];
            string[n][i]=alphabet[rand()%5];
        } 
    }
    
    double start_cpu = get_time();
    for(int n = 0; n < N; n++){
        for(int i = 0; i < S_LEN+1; i++){
            for(int j = 0; j < S_LEN+1; j++){
                mat[i][j]=0;
            }
        }

        int A_LEN = 2;                                      //1 over the real antidiag length (it has to start from the first cell)
        int MIN_COL = 0;

        for (int i=1; i<S_LEN+S_LEN+1; i++){                //compute antidiags per matrix
            for(int j=1; j<A_LEN; j++){
                int row, col;
                
                col = MIN_COL + j;
                row = i - col + 1;
                mat[row][col]= max(mat[row-1][col-1]+(query[n][row-1]==string[n][col-1]), mat[row-1][col]+del, mat[row][col-1]+ins, 0);
            }
            if(i < S_LEN)
                A_LEN++;                                    //increase antidiag len
            else{
                A_LEN--;                                    //decrease antidiag len
                MIN_COL++;
            }
        }
        res[n] = mat[S_LEN][S_LEN];
    }
    double end_cpu = get_time();

    printf("CPU Time: %lf\n", end_cpu-start_cpu);

    return 0;
}