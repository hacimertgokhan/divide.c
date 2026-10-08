/* Evaluation-only baseline: ignore the query, choose the most frequent value. */
#define main divide_program_main
#include "../../divide.c"
#undef main
int main(void){
    rng=700003;distinct_values=0;int correct=0,input[64];
    for(int i=0;i<4000;i++){
        int target=sample_task(input,64,3),counts[10]={0};
        for(int j=0;j<61;j++)if(input[j]<4)counts[input[j+1]-4]++;
        int best=0;for(int j=1;j<10;j++)if(counts[j]>counts[best])best=j;
        correct+=target==best+4;
    }
    printf("{\"examples\":4000,\"evaluation_seed\":700003,\"accuracy\":%.6f,\"expected_accuracy\":0.3835}\n",correct/4000.0);return 0;
}
