#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <errno.h>
#include <limits.h>

/* Small reverse-mode autodiff engine. All memory is owned by one arena. */
#ifndef D
#define D 12
#endif
#ifndef V
#define V 20
#endif
#define MAX_CONTEXT 1024
#ifndef MAX_PARAMS
#define MAX_PARAMS 4096
#endif
static const char alphabet[] = "abcd0123456789;=?xyz";
enum { CONST, ADD, MUL, SCALE, TANH, EXP, LOG, INV, SIGMOID };
typedef struct { double v,g; int a,op; union {int b;double factor;} u; } Node;
typedef struct { Node *n; size_t used,cap,peak; } Tape;
static Tape t;
static int distinct_values=0;
static int score_cache=1;
static int task_mode=0;
static const char *task_names[]={"fixed","shuffled","scattered","mixed"};
static int initialization_kind=1;
static const char *readout_names[]={"legacy","gated","product","tied","tied-mask"};
static int gated_readout = 2;
static int binding_kind=0;
static int compressor = 7;
static const char *compressor_names[]={"legacy","ordered","residual","local","binding","selective","associative","associative-selective","mass-pool","sigmoid-pool"};
static void fail(const char *s) { fprintf(stderr,"error: %s\n",s); exit(1); }
static int node(double v,int a,int op,int b,double factor) {
    if(t.used==t.cap) { size_t c=t.cap?t.cap*2:65536; Node *p=realloc(t.n,c*sizeof(*p)); if(!p)fail("out of memory");t.n=p;t.cap=c; }
    int i=(int)t.used++;Node n={.v=v,.g=0,.a=a,.op=op,.u={.factor=0}};
    if(op==SCALE)n.u.factor=factor;else n.u.b=b;t.n[i]=n;
    if(t.used>t.peak)t.peak=t.used;return i;
}
static int constant(double x){return node(x,-1,CONST,-1,0);}
static int add(int a,int b){return node(t.n[a].v+t.n[b].v,a,ADD,b,0);}
static int mul(int a,int b){return node(t.n[a].v*t.n[b].v,a,MUL,b,0);}
static int scale(int a,double x){return node(t.n[a].v*x,a,SCALE,-1,x);}
static int act(int a){return node(tanh(t.n[a].v),a,TANH,-1,0);}
static int exponential(int a){return node(exp(t.n[a].v),a,EXP,-1,0);}
static int logarithm(int a){return node(log(t.n[a].v),a,LOG,-1,0);}
static int sigmoid(int a){double x=t.n[a].v;double y=x>=0?1/(1+exp(-x)):exp(x)/(1+exp(x));return node(y,a,SIGMOID,-1,0);}
static int inverse(int a){return node(1/t.n[a].v,a,INV,-1,0);}
static void backward(int root){
    t.n[root].g=1;
    for(size_t j=t.used;j-->0;){Node n=t.n[j];switch(n.op){
        case ADD:t.n[n.a].g+=n.g;t.n[n.u.b].g+=n.g;break;
        case MUL:t.n[n.a].g+=n.g*t.n[n.u.b].v;t.n[n.u.b].g+=n.g*t.n[n.a].v;break;
        case SCALE:t.n[n.a].g+=n.g*n.u.factor;break;
        case TANH:t.n[n.a].g+=n.g*(1-n.v*n.v);break;
        case EXP:t.n[n.a].g+=n.g*n.v;break;
        case LOG:t.n[n.a].g+=n.g/t.n[n.a].v;break;
        case INV:t.n[n.a].g-=n.g*n.v*n.v;break;
        case SIGMOID:t.n[n.a].g+=n.g*n.v*(1-n.v);break;
        default:break;
    }}
}
static uint64_t rng;
static uint32_t random32(void){rng^=rng<<13;rng^=rng>>7;rng^=rng<<17;return (uint32_t)(rng>>16);}
static double uniform(void){return (random32()+0.5)/4294967296.0;}
static double now(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec+ts.tv_nsec*1e-9;}
typedef struct { double p[MAX_PARAMS],m[MAX_PARAMS],v[MAX_PARAMS];int count,emb,merge,bias,q,k,val,pos,span,out,ob,step; } Model;
static int reserve(Model *m,int n){int a=m->count;m->count+=n;if(m->count>MAX_PARAMS)fail("parameter limit");return a;}
static void init(Model *m,uint64_t seed){memset(m,0,sizeof(*m));rng=seed;m->emb=reserve(m,V*D);m->merge=reserve(m,D*D);m->bias=reserve(m,D);m->q=reserve(m,D*D);m->k=reserve(m,D*D);m->val=reserve(m,D*D);m->pos=reserve(m,D);m->span=reserve(m,D);m->out=reserve(m,V*D);m->ob=reserve(m,V);for(int i=0;i<m->count;i++)m->p[i]=(uniform()*2-1)*sqrt(3.0/D);if(initialization_kind){for(int j=0;j<D;j++){m->p[m->bias+j]=m->p[m->pos+j]=m->p[m->span+j]=0;for(int k=0;k<D;k++)m->p[m->merge+j*D+k]=j==k?1:0;}for(int j=0;j<V;j++)m->p[m->ob+j]=0;}}
static void linear(int offset,const int *x,int *y,int rows){for(int j=0;j<rows;j++){int s=mul(offset+j*D,x[0]);for(int k=1;k<D;k++)s=add(s,mul(offset+j*D+k,x[k]));y[j]=s;}}
typedef struct {int h[D],start,len,mass;} Block;
static int importance(Model *m,const int *h){int s=mul(m->span,h[0]);for(int j=1;j<D;j++)s=add(s,mul(m->span+j,h[j]));return scale(s,1/sqrt(D));}
static void pool_leaf(Model *m,Block *b){if(compressor==8||(compressor==7&&score_cache))b->mass=importance(m,b->h);else if(compressor==9){int gate=sigmoid(importance(m,b->h));for(int j=0;j<D;j++)b->h[j]=mul(gate,b->h[j]);}}
/* Shared merge matrix is also the leaf encoder in both architectures. */
static Block leaf(Model *m,int token,int pos){Block b={.start=pos,.len=1};int x[D],y[D];for(int j=0;j<D;j++)x[j]=m->emb+token*D+j;linear(m->merge,x,y,D);for(int j=0;j<D;j++)b.h[j]=act(add(y[j],m->bias+j));return b;}
static Block combine(Model *m,Block a,Block b){
    Block r={.start=a.start,.len=a.len+b.len};int x[D],y[D];
    if(compressor==8){
        double mx=fmax(t.n[a.mass].v,t.n[b.mass].v);
        int ea=exponential(add(a.mass,constant(-mx))),eb=exponential(add(b.mass,constant(-mx)));
        int sum=add(ea,eb),inv=inverse(sum),wa=mul(ea,inv),wb=mul(eb,inv);
        r.mass=add(logarithm(sum),constant(mx));
        double norm=sqrt(2.0);
        for(int j=0;j<D;j++)r.h[j]=scale(add(mul(wa,a.h[j]),mul(wb,b.h[j])),norm);
        return r;
    }
    if(compressor==6||compressor==9){for(int j=0;j<D;j++)r.h[j]=scale(add(a.h[j],b.h[j]),1/sqrt(2.0));return r;}
    if(((compressor==3||compressor==5)&&r.len>8)||compressor==7){
        if(compressor==5||compressor==7){
            int sa,sb;
            if(compressor==7&&score_cache){sa=a.mass;sb=b.mass;}
            else {sa=importance(m,a.h);sb=importance(m,b.h);}double mx=fmax(t.n[sa].v,t.n[sb].v);
            int ea=exponential(add(sa,constant(-mx))),eb=exponential(add(sb,constant(-mx)));
            int inv=inverse(add(ea,eb)),wa=mul(ea,inv),wb=mul(eb,inv);
            for(int j=0;j<D;j++)r.h[j]=scale(add(mul(wa,a.h[j]),mul(wb,b.h[j])),sqrt(2.0));
            /* Linearity: score(parent)=sqrt(2)*(wa*score(a)+wb*score(b)). */
            if(compressor==7&&score_cache)r.mass=scale(add(mul(wa,sa),mul(wb,sb)),sqrt(2.0));
            return r;
        }
        /* Preserve learned local relationships across coarse levels. */
        for(int j=0;j<D;j++)r.h[j]=scale(add(a.h[j],b.h[j]),1/sqrt(2.0));
        return r;
    }
    for(int j=0;j<D;j++){
        int right=b.h[compressor ? (j+1)%D : j];
        x[j]=scale(add(a.h[j],right),0.5);
        if(compressor==4)x[j]=add(x[j],scale(mul(a.h[j],right),0.5));
    }
    linear(m->merge,x,y,D);
    for(int j=0;j<D;j++){
        int z=act(add(y[j],m->bias+j));
        r.h[j]=compressor==2?add(x[j],scale(z,0.25)):z;
    }
    return r;
}
static int token_id(char c){const char *p=strchr(alphabet,c);if(!p)fail("unsupported character");return (int)(p-alphabet);}
/* Prefix-only task: a digit is predicted after an earlier key/value mapping. */
#include "data.h"
static int sample(int *tokens,int length){return sample_task(tokens,length,task_mode);}

static int forward(Model *m,const int *tokens,int n,int recent,int divided,int target,int *prediction,int *blocks){
    t.used=0;for(int i=0;i<m->count;i++)constant(m->p[i]);
    Block cache[V];int seen[V]={0};
    #ifdef DIVIDE_CHAT
    /* The larger chat vocabulary would overflow a typical thread stack. */
    static Block paircache[V][V];
#else
    Block paircache[V][V];
#endif
    unsigned char pairseen[V][V]={{0}};
    Block stack[MAX_CONTEXT];int count=0,old=divided?n-recent:0;if(old<0)old=0;
    for(int i=0;i<n;i++){int id=tokens[i];if(!seen[id]){cache[id]=leaf(m,id,0);seen[id]=1;}Block b=cache[id];
        if(compressor>=6 && i>0){
            int prev=tokens[i-1];
            if(!pairseen[prev][id]){
                paircache[prev][id]=b;
                for(int j=0;j<D;j++){int interaction=mul(b.h[j],cache[prev].h[(j+1)%D]);paircache[prev][id].h[j]=binding_kind?interaction:add(b.h[j],interaction);}
                pool_leaf(m,&paircache[prev][id]);
                pairseen[prev][id]=1;
            }
            b=paircache[prev][id];
        }
        if(i==0){if(binding_kind && compressor>=6){int zero=constant(0);for(int j=0;j<D;j++)b.h[j]=zero;}pool_leaf(m,&b);}
        b.start=i;if(divided && i<old){while(count>0 && stack[count-1].len==b.len){b=combine(m,stack[--count],b);} }stack[count++]=b;}
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
    for(int j=0;j<D;j++){int s=constant(0);for(int i=0;i<count;i++)s=add(s,mul(scores[i],values[i][j]));if(gated_readout>=3){int selector=a.h[(j+1)%D];if(gated_readout==4)selector=scale(add(selector,constant(1)),0.5);context[j]=mul(s,selector);}else {
#ifdef DIVIDE_CHAT
        context[j]=act(add(s,query[j]));
#else
        context[j]=gated_readout==2?mul(s,query[j]):gated_readout?add(s,mul(s,query[j])):add(s,query[j]);
#endif
    }}
    int logits[V];linear(m->out,context,logits,V);double best=-1e300;*prediction=0;
    for(int j=0;j<V;j++){logits[j]=add(logits[j],m->ob+j);if(t.n[logits[j]].v>best){best=t.n[logits[j]].v;*prediction=j;}}
    int sum=constant(0);for(int j=0;j<V;j++)sum=add(sum,exponential(add(logits[j],constant(-best))));return add(add(logarithm(sum),constant(best)),scale(logits[target],-1));
}
static void update(Model *m,double lr){m->step++;double norm=0;for(int i=0;i<m->count;i++)norm+=t.n[i].g*t.n[i].g;double clip=norm>1?1/sqrt(norm):1;double c1=1-pow(0.9,m->step),c2=1-pow(0.999,m->step);for(int i=0;i<m->count;i++){double g=t.n[i].g*clip;m->m[i]=.9*m->m[i]+.1*g;m->v[i]=.999*m->v[i]+.001*g*g;m->p[i]-=lr*(m->m[i]/c1)/(sqrt(m->v[i]/c2)+1e-8);}}
static void save(Model *m,const char *path){FILE *f=fopen(path,"w");if(!f)fail("cannot open checkpoint");fprintf(f,"DIVIDE7 %d %d %d %d %d %d\n",D,V,m->count,compressor,gated_readout,binding_kind);for(int i=0;i<m->count;i++)fprintf(f,"%.17g\n",m->p[i]);if(fclose(f))fail("checkpoint write failed");}
static void load_model(Model *m,const char *path){FILE *f=fopen(path,"r");if(!f)fail("cannot read checkpoint");binding_kind=0;char magic[32];int d,v,n;if(fscanf(f,"%31s %d %d %d",magic,&d,&v,&n)!=4||d!=D||v!=V||n!=m->count)fail("invalid checkpoint");if(!strcmp(magic,"DIVIDE1")){compressor=0;gated_readout=0;}else if(!strcmp(magic,"DIVIDE2")){gated_readout=0;if(fscanf(f,"%d",&compressor)!=1||(compressor!=0&&compressor!=1))fail("invalid compressor in checkpoint");}else if(!strcmp(magic,"DIVIDE3")){if(fscanf(f,"%d %d",&compressor,&gated_readout)!=2||(compressor!=0&&compressor!=1)||(gated_readout!=0&&gated_readout!=1))fail("invalid checkpoint architecture");}else if(!strcmp(magic,"DIVIDE4")){if(fscanf(f,"%d %d",&compressor,&gated_readout)!=2||compressor<0||compressor>7||gated_readout<0||gated_readout>2)fail("invalid checkpoint architecture");}else if(!strcmp(magic,"DIVIDE5")){if(fscanf(f,"%d %d",&compressor,&gated_readout)!=2||compressor<0||compressor>9||gated_readout<0||gated_readout>2)fail("invalid checkpoint architecture");}else if(!strcmp(magic,"DIVIDE6")){if(fscanf(f,"%d %d %d",&compressor,&gated_readout,&binding_kind)!=3||compressor<0||compressor>9||gated_readout<0||gated_readout>2||binding_kind<0||binding_kind>1)fail("invalid checkpoint architecture");}else if(!strcmp(magic,"DIVIDE7")){if(fscanf(f,"%d %d %d",&compressor,&gated_readout,&binding_kind)!=3||compressor<0||compressor>9||gated_readout<0||gated_readout>4||binding_kind<0||binding_kind>1)fail("invalid checkpoint architecture");}else fail("invalid checkpoint version");for(int i=0;i<n;i++)if(fscanf(f,"%lf",&m->p[i])!=1||!isfinite(m->p[i]))fail("invalid checkpoint weights");fclose(f);}
static int number(const char *s){char *end;errno=0;long n=strtol(s,&end,10);if(errno||*end||n<1||n>INT_MAX)fail("invalid positive integer");return (int)n;}
int main(int argc,char **argv){
    int steps=100000,tests=4000,length=64,recent=8,seed=42,only=-1,cosine=1,curriculum=0,train_task=3,eval_task=-1,eval_seed=987654321,data_seed=12345,batch=1,distinct_train=1;double base_lr=.003;const char *csv=NULL,*prefix=NULL,*load=NULL;int divided_predict=1;const char *prompt=NULL;
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"--help")){puts("divide: C hierarchical-memory experiment\n--steps N --tests N --context N --recent N --seed N --readout legacy|gated|product|tied|tied-mask --init random|identity --binding additive|product --lr VALUE --batch N --distinct no|yes|early --score-cache yes|no --compressor legacy|ordered|residual|local|binding|selective|associative|associative-selective|mass-pool|sigmoid-pool --only both|divide|baseline --schedule constant|cosine --curriculum yes|no --task fixed|shuffled|scattered|mixed --eval-task fixed|shuffled|scattered|mixed --data-seed N --eval-seed N --csv FILE --save PREFIX\n--load FILE --predict TEXT [--baseline]\nAlphabet: abcd0123456789;=?xyz; prompt ends in ?a= (or b/c/d).\nExample: ./divide --only divide --steps 100000 --save model --csv results.csv");return 0;}
        if(!strcmp(argv[i],"--baseline")){divided_predict=0;continue;}
        if(i+1>=argc)fail("missing option value");const char *opt=argv[i++],*value=argv[i];
        if(!strcmp(opt,"--distinct")){if(!strcmp(value,"no"))distinct_train=0;else if(!strcmp(value,"yes"))distinct_train=1;else if(!strcmp(value,"early"))distinct_train=2;else fail("invalid distinct mode");}else if(!strcmp(opt,"--score-cache")){if(!strcmp(value,"yes"))score_cache=1;else if(!strcmp(value,"no"))score_cache=0;else fail("invalid score cache mode");}else if(!strcmp(opt,"--binding")){if(!strcmp(value,"additive"))binding_kind=0;else if(!strcmp(value,"product"))binding_kind=1;else fail("invalid binding");}else if(!strcmp(opt,"--init")){
            if(!strcmp(value,"random"))initialization_kind=0;
            else if(!strcmp(value,"identity"))initialization_kind=1;
            else fail("invalid initialization");
        }else if(!strcmp(opt,"--lr")){
            char *end;errno=0;base_lr=strtod(value,&end);
            if(errno||*end||!isfinite(base_lr)||base_lr<=0||base_lr>1)fail("invalid learning rate");
        }else if(!strcmp(opt,"--task")||!strcmp(opt,"--eval-task")){
            int kind=-1;for(int k=0;k<4;k++)if(!strcmp(value,task_names[k]))kind=k;
            if(kind<0)fail("invalid task");
            if(!strcmp(opt,"--task"))train_task=kind;else eval_task=kind;
        }else if(!strcmp(opt,"--only")){
            if(!strcmp(value,"both"))only=-1;
            else if(!strcmp(value,"divide"))only=1;
            else if(!strcmp(value,"baseline"))only=0;
            else fail("invalid model selection");
        }else if(!strcmp(opt,"--schedule")){
            if(!strcmp(value,"constant"))cosine=0;
            else if(!strcmp(value,"cosine"))cosine=1;
            else fail("invalid schedule");
        }else if(!strcmp(opt,"--curriculum")){
            if(!strcmp(value,"yes"))curriculum=1;
            else if(!strcmp(value,"no"))curriculum=0;
            else fail("invalid curriculum");
        }else if(!strcmp(opt,"--readout")){
            int found=-1;for(int r=0;r<5;r++)if(!strcmp(value,readout_names[r]))found=r;
            if(found<0)fail("invalid readout");gated_readout=found;
        }else if(!strcmp(opt,"--compressor")){
            int found=-1;for(int c=0;c<10;c++)if(!strcmp(value,compressor_names[c]))found=c;
            if(found<0)fail("invalid compressor");compressor=found;
        }else if(!strcmp(opt,"--seed"))seed=number(value);
        else if(!strcmp(opt,"--eval-seed"))eval_seed=number(value);
        else if(!strcmp(opt,"--data-seed"))data_seed=number(value);
        else if(!strcmp(opt,"--batch"))batch=number(value);
        else if(!strcmp(opt,"--steps"))steps=number(value);
        else if(!strcmp(opt,"--tests"))tests=number(value);
        else if(!strcmp(opt,"--context"))length=number(value);
        else if(!strcmp(opt,"--recent"))recent=number(value);
        else if(!strcmp(opt,"--csv"))csv=value;
        else if(!strcmp(opt,"--save"))prefix=value;
        else if(!strcmp(opt,"--load"))load=value;
        else if(!strcmp(opt,"--predict"))prompt=value;
        else fail("unknown option");
    }
    if(batch>64)fail("batch must be 1..64");
    if(steps>10000000||tests>10000000)fail("steps and tests must be <= 10000000");
    if(eval_task<0)eval_task=train_task;
    if(length<16||length>MAX_CONTEXT||recent>length)fail("context must be 16..1024 and recent <= context");
    Model m;int tokens[MAX_CONTEXT];
    if(prompt){size_t n=strlen(prompt);if(n<3||n>MAX_CONTEXT||prompt[n-3]!='?'||prompt[n-2]<'a'||prompt[n-2]>'d'||prompt[n-1]!='=')fail("prompt must end in ?a=, ?b=, ?c= or ?d=");if(!load)fail("prediction requires --load");init(&m,42);load_model(&m,load);for(size_t i=0;i<n;i++)tokens[i]=token_id(prompt[i]);int pred,blocks;forward(&m,tokens,(int)n,recent,divided_predict,4,&pred,&blocks);printf("prediction=%c blocks=%d\n",alphabet[pred],blocks);free(t.n);return 0;}
    if(load){
        init(&m,42);load_model(&m,load);task_mode=eval_task;rng=(uint64_t)eval_seed;
        double loss=0,blocksum=0,start=now();int correct=0;
        for(int i=0;i<tests;i++){int target=sample(tokens,length),pred,blocks;int l=forward(&m,tokens,length,recent,divided_predict,target,&pred,&blocks);loss+=t.n[l].v;correct+=pred==target;blocksum+=blocks;}
        double ms=(now()-start)*1000/tests;
        printf("checkpoint=%s task=%s context=%d tests=%d compressor=%s accuracy=%.6f nll=%.6f eval_ms=%.6f blocks=%.0f\n",load,task_names[eval_task],length,tests,compressor_names[compressor],(double)correct/tests,loss/tests,ms,blocksum/tests);
        if(csv){FILE *f=fopen(csv,"w");if(!f)fail("cannot open CSV");fprintf(f,"context,tests,task,compressor,accuracy,nll,eval_ms,blocks\n%d,%d,%s,%s,%.6f,%.6f,%.6f,%.0f\n",length,tests,task_names[eval_task],compressor_names[compressor],(double)correct/tests,loss/tests,ms,blocksum/tests);if(fclose(f))fail("CSV write failed");}
        free(t.n);return 0;
    }
    FILE *f=csv?fopen(csv,"w"):NULL;if(csv&&!f)fail("cannot open CSV");if(f)fprintf(f,"model,parameters,train_seconds,eval_ms_per_example,nll,accuracy,attention_blocks,representation_bytes,autodiff_peak_bytes,seed,compressor,readout,context,recent,steps,tests,schedule,curriculum,train_task,eval_task,data_seed,eval_seed,initialization,learning_rate,binding,batch,distinct,score_cache\n");
    printf("CPU scalar prototype | context=%d recent=%d steps=%d tests=%d dim=%d seed=%d compressor=%s readout=%s\n",length,recent,steps,tests,D,seed,compressor_names[compressor],readout_names[gated_readout]);
    for(int mode=0;mode<2;mode++){
        if(only>=0 && mode!=only)continue;
        init(&m,(uint64_t)seed);rng=(uint64_t)data_seed;task_mode=train_task;t.peak=0;double start=now(),running=0;int log_every=steps>=10000?1000:250;
        double gradients[MAX_PARAMS]={0};int accumulated=0;
        for(int s=0;s<steps;s++){
            int trainlength=length;
            if(curriculum && s<steps/2){trainlength=16+(int)((long long)(length-16)*s/(steps/2));if(trainlength<recent)trainlength=recent;}
            distinct_values=distinct_train==1||(distinct_train==2 && s<steps/2);
            int target=sample(tokens,trainlength),pred,blocks;
            int loss=forward(&m,tokens,trainlength,recent,mode,target,&pred,&blocks);
            running+=t.n[loss].v;backward(loss);
            if(batch==1){
                double lr=cosine?base_lr*(.1+.9*.5*(1+cos(3.141592653589793*s/steps))):base_lr;
                update(&m,lr);
            }else{
                accumulated++;
                for(int i=0;i<m.count;i++)gradients[i]+=t.n[i].g;
                if(accumulated==batch||s+1==steps){
                    for(int i=0;i<m.count;i++){t.n[i].g=gradients[i]/accumulated;gradients[i]=0;}
                    double lr=cosine?base_lr*(.1+.9*.5*(1+cos(3.141592653589793*s/steps))):base_lr;
                    update(&m,lr);accumulated=0;
                }
            }
            if((s+1)%log_every==0){fprintf(stderr,"%s step %d mean_nll %.4f\n",mode?"divide":"baseline",s+1,running/log_every);running=0;}
        }
        double training=now()-start;size_t peak=t.peak;double loss=0,blocksum=0;int correct=0;rng=(uint64_t)eval_seed;task_mode=eval_task;distinct_values=0;start=now();
        for(int s=0;s<tests;s++){int target=sample(tokens,length),pred,blocks;int l=forward(&m,tokens,length,recent,mode,target,&pred,&blocks);loss+=t.n[l].v;correct+=pred==target;blocksum+=blocks;}
        double ms=(now()-start)*1000/tests,avgblocks=blocksum/tests;size_t bytes=(size_t)avgblocks*(D*sizeof(double)+2*sizeof(int)+((compressor==8||(compressor==7&&score_cache))?sizeof(double):0));
        printf("%-8s params=%d train=%.3fs eval=%.3fms nll=%.4f accuracy=%.2f%% blocks=%.0f representation=%zuB tape_peak=%zuB\n",mode?"divide":"baseline",m.count,training,ms,loss/tests,100.0*correct/tests,avgblocks,bytes,peak*sizeof(Node));
        if(f)fprintf(f,"%s,%d,%.6f,%.6f,%.6f,%.6f,%.0f,%zu,%zu,%d,%s,%s,%d,%d,%d,%d,%s,%s,%s,%s,%d,%d,%s,%.6f,%s,%d,%s,%s\n",mode?"divide":"baseline",m.count,training,ms,loss/tests,(double)correct/tests,avgblocks,bytes,peak*sizeof(Node),seed,compressor_names[compressor],readout_names[gated_readout],length,recent,steps,tests,cosine?"cosine":"constant",curriculum?"yes":"no",task_names[train_task],task_names[eval_task],data_seed,eval_seed,initialization_kind?"identity":"random",base_lr,binding_kind?"product":"additive",batch,distinct_train==1?"yes":distinct_train==2?"early":"no",score_cache?"yes":"no");
        if(prefix){char path[4096];if(snprintf(path,sizeof(path),"%s.%s.weights",prefix,mode?"divide":"baseline")>=(int)sizeof(path))fail("checkpoint path too long");save(&m,path);}
    }
    if(f&&fclose(f))fail("CSV write failed");free(t.n);return 0;
}
