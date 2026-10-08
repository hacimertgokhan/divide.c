#define main divide_program_main
#ifndef SOURCE
#define SOURCE "../../divide.c"
#endif
#include SOURCE
#undef main
#ifdef PREVIOUS_ENGINE
#include "../../data.h"
#endif
static int cmp(const void *a,const void *b){double x=*(const double*)a,y=*(const double*)b;return (x>y)-(x<y);}
int main(int argc,char **argv){
    if(argc!=5)fail("usage: benchmark checkpoint context task_id seed");
    int length=number(argv[2]),task=atoi(argv[3]),seed=number(argv[4]);if(length<16||length>MAX_CONTEXT||task<0||task>3)fail("invalid benchmark configuration");
    Model m;init(&m,42);load_model(&m,argv[1]);int tokens[MAX_CONTEXT],pred,blocks;double times[9],nll=0;int correct=0;size_t peak=0;
    for(int rep=-1;rep<9;rep++){
        rng=(uint64_t)seed;t.peak=0;double start=now();
        for(int i=0;i<4000;i++){int target=sample_task(tokens,length,task);int l=forward(&m,tokens,length,8,1,target,&pred,&blocks);if(rep==0){nll+=t.n[l].v;correct+=pred==target;}}
        if(rep>=0)times[rep]=(now()-start)*1000/4000;peak=t.peak;
    }
    qsort(times,9,sizeof(double),cmp);
    printf("{\"median_ms\":%.9f,\"min_ms\":%.9f,\"max_ms\":%.9f,\"accuracy\":%.9f,\"nll\":%.9f,\"blocks\":%d,\"parameters\":%d,\"tape_bytes\":%zu}\n",times[4],times[0],times[8],correct/4000.0,nll/4000,blocks,m.count,peak*sizeof(Node));free(t.n);return 0;
}
