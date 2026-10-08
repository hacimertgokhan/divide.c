#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <errno.h>

/* Small reverse-mode autodiff engine. All memory is owned by one arena. */
#define D 12
#define V 20
#define MAX_CONTEXT 1024
#define MAX_PARAMS 4096
static const char alphabet[] = "abcd0123456789;=?xyz";
typedef struct { double v,g,da,db; int a,b; } Node;
typedef struct { Node *n; size_t used,cap,peak; } Tape;
static Tape t;
static int gated_readout = 1;
static int compressor = 1; /* 0: legacy average, 1: ordered cyclic mix */
static void fail(const char *s) { fprintf(stderr,"error: %s\n",s); exit(1); }
static int node(double v,int a,int b,double da,double db) {
    if(t.used==t.cap) { size_t c=t.cap?t.cap*2:65536; Node *p=realloc(t.n,c*sizeof(*p)); if(!p)fail("out of memory");t.n=p;t.cap=c; }
    int i=(int)t.used++;t.n[i]=(Node){v,0,da,db,a,b};if(t.used>t.peak)t.peak=t.used;return i;
}
static int constant(double x){return node(x,-1,-1,0,0);}
static int add(int a,int b){return node(t.n[a].v+t.n[b].v,a,b,1,1);}
static int mul(int a,int b){return node(t.n[a].v*t.n[b].v,a,b,t.n[b].v,t.n[a].v);}
static int scale(int a,double x){return node(t.n[a].v*x,a,-1,x,0);}
static int act(int a){double x=tanh(t.n[a].v);return node(x,a,-1,1-x*x,0);}
static int exponential(int a){double x=exp(t.n[a].v);return node(x,a,-1,x,0);}
static int logarithm(int a){return node(log(t.n[a].v),a,-1,1/t.n[a].v,0);}
static int inverse(int a){double x=t.n[a].v;return node(1/x,a,-1,-1/(x*x),0);}
static void backward(int root){t.n[root].g=1;for(size_t j=t.used;j-->0;){Node n=t.n[j];if(n.a>=0)t.n[n.a].g+=n.g*n.da;if(n.b>=0)t.n[n.b].g+=n.g*n.db;}}
static uint64_t rng;
static uint32_t random32(void){rng^=rng<<13;rng^=rng>>7;rng^=rng<<17;return (uint32_t)(rng>>16);}
static double uniform(void){return (random32()+0.5)/4294967296.0;}
static double now(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec+ts.tv_nsec*1e-9;}
typedef struct { double p[MAX_PARAMS],m[MAX_PARAMS],v[MAX_PARAMS];int count,emb,merge,bias,q,k,val,pos,span,out,ob,step; } Model;
static int reserve(Model *m,int n){int a=m->count;m->count+=n;if(m->count>MAX_PARAMS)fail("parameter limit");return a;}
static void init(Model *m,uint64_t seed){memset(m,0,sizeof(*m));rng=seed;m->emb=reserve(m,V*D);m->merge=reserve(m,D*D);m->bias=reserve(m,D);m->q=reserve(m,D*D);m->k=reserve(m,D*D);m->val=reserve(m,D*D);m->pos=reserve(m,D);m->span=reserve(m,D);m->out=reserve(m,V*D);m->ob=reserve(m,V);for(int i=0;i<m->count;i++)m->p[i]=(uniform()*2-1)*sqrt(3.0/D);}
static void linear(int offset,const int *x,int *y,int rows){for(int j=0;j<rows;j++){int s=constant(0);for(int k=0;k<D;k++)s=add(s,mul(offset+j*D+k,x[k]));y[j]=s;}}
typedef struct {int h[D],start,len;} Block;
/* Shared merge matrix is also the leaf encoder in both architectures. */
static Block leaf(Model *m,int token,int pos){Block b={.start=pos,.len=1};int x[D],y[D];for(int j=0;j<D;j++)x[j]=m->emb+token*D+j;linear(m->merge,x,y,D);for(int j=0;j<D;j++)b.h[j]=act(add(y[j],m->bias+j));return b;}
static Block combine(Model *m,Block a,Block b){Block r={.start=a.start,.len=a.len+b.len};int x[D],y[D];for(int j=0;j<D;j++)x[j]=scale(add(a.h[j],b.h[compressor ? (j+1)%D : j]),0.5);linear(m->merge,x,y,D);for(int j=0;j<D;j++)r.h[j]=act(add(y[j],m->bias+j));return r;}
static int token_id(char c){const char *p=strchr(alphabet,c);if(!p)fail("unsupported character");return (int)(p-alphabet);}
/* Prefix-only task: a digit is predicted after an earlier key/value mapping. */
static int sample(int *tokens,int length){int digits[4];for(int i=0;i<4;i++){digits[i]=random32()%10;tokens[i*3]=i;tokens[i*3+1]=4+digits[i];tokens[i*3+2]=token_id(';');}for(int i=12;i<length-3;i++)tokens[i]=token_id("xyz"[random32()%3]);int key=random32()%4;tokens[length-3]=token_id('?');tokens[length-2]=key;tokens[length-1]=token_id('=');return 4+digits[key];}
static int forward(Model *m,const int *tokens,int n,int recent,int divided,int target,int *prediction,int *blocks){
    t.used=0;for(int i=0;i<m->count;i++)constant(m->p[i]);
    Block cache[V];int seen[V]={0};
    Block stack[MAX_CONTEXT];int count=0,old=divided?n-recent:0;if(old<0)old=0;
    for(int i=0;i<n;i++){int id=tokens[i];if(!seen[id]){cache[id]=leaf(m,id,0);seen[id]=1;}Block b=cache[id];b.start=i;if(divided && i<old){while(count>0 && stack[count-1].len==b.len){b=combine(m,stack[--count],b);} }stack[count++]=b;}
    *blocks=count;
    /* Query includes the requested key and final separator; no target enters it. */
    int x[D],query[D];Block a=cache[tokens[n-2]],b=cache[tokens[n-1]];
    for(int j=0;j<D;j++)x[j]=scale(add(a.h[j],b.h[j]),0.5);linear(m->q,x,query,D);
    int scores[MAX_CONTEXT],values[MAX_CONTEXT][D];double maxscore=-1e300;
    for(int i=0;i<count;i++){
        int enriched[D],key[D];double pos=(double)stack[i].start/n,span=log2(1.0+stack[i].len)/10.0;
        for(int j=0;j<D;j++)enriched[j]=add(stack[i].h[j],add(scale(m->pos+j,pos),scale(m->span+j,span)));
        linear(m->k,enriched,key,D);linear(m->val,enriched,values[i],D);
        int s=constant(0);for(int j=0;j<D;j++)s=add(s,mul(query[j],key[j]));scores[i]=scale(s,1/sqrt(D));if(t.n[scores[i]].v>maxscore)maxscore=t.n[scores[i]].v;
    }
    int denom=constant(0);for(int i=0;i<count;i++){scores[i]=exponential(add(scores[i],constant(-maxscore)));denom=add(denom,scores[i]);}int inv=inverse(denom),context[D];
    for(int i=0;i<count;i++)scores[i]=mul(scores[i],inv);
    for(int j=0;j<D;j++){int s=constant(0);for(int i=0;i<count;i++)s=add(s,mul(scores[i],values[i][j]));context[j]=gated_readout?add(s,mul(s,query[j])):add(s,query[j]);}
    int logits[V];linear(m->out,context,logits,V);double best=-1e300;*prediction=0;
    for(int j=0;j<V;j++){logits[j]=add(logits[j],m->ob+j);if(t.n[logits[j]].v>best){best=t.n[logits[j]].v;*prediction=j;}}
    int sum=constant(0);for(int j=0;j<V;j++)sum=add(sum,exponential(add(logits[j],constant(-best))));return add(add(logarithm(sum),constant(best)),scale(logits[target],-1));
}
static void update(Model *m,double lr){m->step++;double norm=0;for(int i=0;i<m->count;i++)norm+=t.n[i].g*t.n[i].g;double clip=norm>1?1/sqrt(norm):1;double c1=1-pow(0.9,m->step),c2=1-pow(0.999,m->step);for(int i=0;i<m->count;i++){double g=t.n[i].g*clip;m->m[i]=.9*m->m[i]+.1*g;m->v[i]=.999*m->v[i]+.001*g*g;m->p[i]-=lr*(m->m[i]/c1)/(sqrt(m->v[i]/c2)+1e-8);}}
static void save(Model *m,const char *path){FILE *f=fopen(path,"w");if(!f)fail("cannot open checkpoint");fprintf(f,"DIVIDE3 %d %d %d %d %d\n",D,V,m->count,compressor,gated_readout);for(int i=0;i<m->count;i++)fprintf(f,"%.17g\n",m->p[i]);if(fclose(f))fail("checkpoint write failed");}
static void load_model(Model *m,const char *path){FILE *f=fopen(path,"r");if(!f)fail("cannot read checkpoint");char magic[32];int d,v,n;if(fscanf(f,"%31s %d %d %d",magic,&d,&v,&n)!=4||d!=D||v!=V||n!=m->count)fail("invalid checkpoint");if(!strcmp(magic,"DIVIDE1")){compressor=0;gated_readout=0;}else if(!strcmp(magic,"DIVIDE2")){gated_readout=0;if(fscanf(f,"%d",&compressor)!=1||(compressor!=0&&compressor!=1))fail("invalid compressor in checkpoint");}else if(!strcmp(magic,"DIVIDE3")){if(fscanf(f,"%d %d",&compressor,&gated_readout)!=2||(compressor!=0&&compressor!=1)||(gated_readout!=0&&gated_readout!=1))fail("invalid checkpoint architecture");}else fail("invalid checkpoint version");for(int i=0;i<n;i++)if(fscanf(f,"%lf",&m->p[i])!=1||!isfinite(m->p[i]))fail("invalid checkpoint weights");fclose(f);}
static int number(const char *s){char *end;errno=0;long n=strtol(s,&end,10);if(errno||*end||n<1||n>10000000)fail("invalid positive integer");return (int)n;}
int main(int argc,char **argv){
    int steps=1000,tests=200,length=64,recent=8,seed=42;const char *csv=NULL,*prefix=NULL,*load=NULL;int divided_predict=1;const char *prompt=NULL;
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"--help")){puts("divide: C hierarchical-memory experiment\n--steps N --tests N --context N --recent N --seed N --readout legacy|gated --compressor legacy|ordered --csv FILE --save PREFIX\n--load FILE --predict TEXT [--baseline]\nAlphabet: abcd0123456789;=?xyz; prompt ends in ?a= (or b/c/d).\nExample: ./divide --steps 2000 --context 64 --save model --csv results.csv");return 0;}
        if(!strcmp(argv[i],"--baseline")){divided_predict=0;continue;}
        if(i+1>=argc)fail("missing option value");const char *opt=argv[i++],*value=argv[i];
        if(!strcmp(opt,"--readout")){if(!strcmp(value,"legacy"))gated_readout=0;else if(!strcmp(value,"gated"))gated_readout=1;else fail("invalid readout");}else if(!strcmp(opt,"--seed"))seed=number(value);else if(!strcmp(opt,"--compressor")){if(!strcmp(value,"legacy"))compressor=0;else if(!strcmp(value,"ordered"))compressor=1;else fail("invalid compressor");}else if(!strcmp(opt,"--steps"))steps=number(value);else if(!strcmp(opt,"--tests"))tests=number(value);else if(!strcmp(opt,"--context"))length=number(value);else if(!strcmp(opt,"--recent"))recent=number(value);else if(!strcmp(opt,"--csv"))csv=value;else if(!strcmp(opt,"--save"))prefix=value;else if(!strcmp(opt,"--load"))load=value;else if(!strcmp(opt,"--predict"))prompt=value;else fail("unknown option");
    }
    if(length<16||length>MAX_CONTEXT||recent>length)fail("context must be 16..1024 and recent <= context");
    Model m;int tokens[MAX_CONTEXT];
    if(prompt){size_t n=strlen(prompt);if(n<3||n>MAX_CONTEXT||prompt[n-3]!='?'||prompt[n-2]<'a'||prompt[n-2]>'d'||prompt[n-1]!='=')fail("prompt must end in ?a=, ?b=, ?c= or ?d=");if(!load)fail("prediction requires --load");init(&m,42);load_model(&m,load);for(size_t i=0;i<n;i++)tokens[i]=token_id(prompt[i]);int pred,blocks;forward(&m,tokens,(int)n,recent,divided_predict,4,&pred,&blocks);printf("prediction=%c blocks=%d\n",alphabet[pred],blocks);free(t.n);return 0;}
    if(load)fail("--load requires --predict");
    FILE *f=csv?fopen(csv,"w"):NULL;if(csv&&!f)fail("cannot open CSV");if(f)fprintf(f,"model,parameters,train_seconds,eval_ms_per_example,nll,accuracy,attention_blocks,representation_bytes,autodiff_peak_bytes,seed,compressor,readout,context,recent,steps,tests\n");
    printf("CPU scalar prototype | context=%d recent=%d steps=%d tests=%d dim=%d seed=%d compressor=%s readout=%s\n",length,recent,steps,tests,D,seed,compressor?"ordered":"legacy",gated_readout?"gated":"legacy");
    for(int mode=0;mode<2;mode++){
        init(&m,(uint64_t)seed);rng=12345;t.peak=0;double start=now(),running=0;
        for(int s=0;s<steps;s++){int target=sample(tokens,length),pred,blocks;int loss=forward(&m,tokens,length,recent,mode,target,&pred,&blocks);running+=t.n[loss].v;backward(loss);update(&m,.003);if((s+1)%250==0){fprintf(stderr,"%s step %d mean_nll %.4f\n",mode?"divide":"baseline",s+1,running/250);running=0;}}
        double training=now()-start;size_t peak=t.peak;double loss=0,blocksum=0;int correct=0;rng=987654321;start=now();
        for(int s=0;s<tests;s++){int target=sample(tokens,length),pred,blocks;int l=forward(&m,tokens,length,recent,mode,target,&pred,&blocks);loss+=t.n[l].v;correct+=pred==target;blocksum+=blocks;}
        double ms=(now()-start)*1000/tests,avgblocks=blocksum/tests;size_t bytes=(size_t)avgblocks*(D*sizeof(double)+2*sizeof(int));
        printf("%-8s params=%d train=%.3fs eval=%.3fms nll=%.4f accuracy=%.2f%% blocks=%.0f representation=%zuB tape_peak=%zuB\n",mode?"divide":"baseline",m.count,training,ms,loss/tests,100.0*correct/tests,avgblocks,bytes,peak*sizeof(Node));
        if(f)fprintf(f,"%s,%d,%.6f,%.6f,%.6f,%.6f,%.0f,%zu,%zu,%d,%s,%s,%d,%d,%d,%d\n",mode?"divide":"baseline",m.count,training,ms,loss/tests,(double)correct/tests,avgblocks,bytes,peak*sizeof(Node),seed,compressor?"ordered":"legacy",gated_readout?"gated":"legacy",length,recent,steps,tests);
        if(prefix){char path[4096];if(snprintf(path,sizeof(path),"%s.%s.weights",prefix,mode?"divide":"baseline")>=(int)sizeof(path))fail("checkpoint path too long");save(&m,path);}
    }
    if(f&&fclose(f))fail("CSV write failed");free(t.n);return 0;
}
