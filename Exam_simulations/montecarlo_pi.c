/*
 * This C program estimates PI value by using Monte Carlo simulations.
 * You can find an quick explanation of the method at the following link:
 * https://www.geeksforgeeks.org/estimating-value-pi-using-monte-carlo/
 *  
 * The goal of the exercise is to accelerate the algorithm onto GPU
 */

#include<stdio.h>
#include<stdlib.h>
#include<time.h>

#define N (1<<30)

int main(int argc, char* argv[]){
  int i, count; 
  double x, y, pi;
  srand(time(NULL));
  for(i=0, count=0; i<N; i++){
    x = (double)rand()/(RAND_MAX);
    y = (double)rand()/(RAND_MAX);
    count += (x*x+y*y<=1);
  }
  pi = ((double) count)/N*4;

  printf("%f\n", pi);

  return 0;
}