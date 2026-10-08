#define main divide_program_main
#ifndef SOURCE
#define SOURCE "../divide.c"
#endif
#include SOURCE
#undef main
static int cmp(const void *a,const void *b){double x=*(const double*)a,y=*(const double*)b;return (x>y)-(x<y);}
int main(int argc,char **argv){if(argc!=2)fail("usage: benchmark checkpoint");Model m;init(&m,42);load_model(&m,argv[1]);int tokens[64],pred,blocks;double samples[9],nll=0;int correct=0;for(int rep=-1;rep<9;rep++){rng=987654321;double start=now();for(int s=0;s<4000;s++){int target=sample(tokens,64);int l=forward(&m,tokens,64,8,1,target,&pred,&blocks);if(rep==0){nll+=t.n[l].v;correct+=pred==target;}}if(rep>=0)samples[rep]=(now()-start)*1000/4000;}qsort(samples,9,sizeof(double),cmp);printf("%s median_ms=%.6f min_ms=%.6f max_ms=%.6f accuracy=%.6f nll=%.6f\n",argv[1],samples[4],samples[0],samples[8],correct/4000.0,nll/4000);free(t.n);return 0;}
