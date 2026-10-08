#define main divide_program_main
#ifndef SOURCE
#define SOURCE "../../divide.c"
#define NEW_ENGINE 1
#endif
#include SOURCE
#undef main
static int cmp(const void *a,const void *b){double x=*(const double*)a,y=*(const double*)b;return (x>y)-(x<y);}
int main(int argc,char **argv){
    if(argc!=2)fail("usage: benchmark checkpoint");
#ifdef NEW_ENGINE
    score_cache=1;
#endif
    Model m;init(&m,42);load_model(&m,argv[1]);int input[64],pred,blocks,correct=0;double times[9],nll=0;size_t peak=0;
    for(int rep=-1;rep<9;rep++){
        rng=700003;t.peak=0;double start=now();
        for(int i=0;i<4000;i++){int target=sample_task(input,64,3);int l=forward(&m,input,64,8,1,target,&pred,&blocks);if(rep==0){correct+=pred==target;nll+=t.n[l].v;}}
        if(rep>=0)times[rep]=(now()-start)*1000/4000;peak=t.peak;
    }
    qsort(times,9,sizeof(double),cmp);
    printf("{\"median_ms\":%.9f,\"min_ms\":%.9f,\"max_ms\":%.9f,\"accuracy\":%.9f,\"nll\":%.9f,\"blocks\":%d,\"parameters\":%d,\"tape_bytes\":%zu}\n",times[4],times[0],times[8],correct/4000.0,nll/4000,blocks,m.count,peak*sizeof(Node));free(t.n);return 0;
}
