#define main divide_program_main
#include "divide.c"
#undef main
#include <assert.h>
int main(void){
    Model m;int tokens[32];init(&m,42);rng=77;int target=sample(tokens,32);
    for(int bind=0;bind<2;bind++)for(int readout=0;readout<5;readout++)for(int comp=0;comp<10;comp++)for(int mode=0;mode<2;mode++){
        binding_kind=bind;gated_readout=readout;compressor=comp;
        int pred,blocks;int loss=forward(&m,tokens,32,4,mode,target,&pred,&blocks);
        assert(blocks==(mode?7:32));assert(isfinite(t.n[loss].v));backward(loss);
        int indices[]={m.emb,m.merge+3,m.bias+2,m.q+4,m.k+6,m.val+8,m.pos+2,m.span+1,m.out+4*D,m.ob+target,m.emb+17*D,m.emb+18*D};
        double analytic[12];for(int j=0;j<12;j++)analytic[j]=t.n[indices[j]].g;
        for(int j=0;j<12;j++){
            int idx=indices[j];double original=m.p[idx],eps=1e-5;
            m.p[idx]=original+eps;loss=forward(&m,tokens,32,4,mode,target,&pred,&blocks);double plus=t.n[loss].v;
            m.p[idx]=original-eps;loss=forward(&m,tokens,32,4,mode,target,&pred,&blocks);double minus=t.n[loss].v;m.p[idx]=original;
            double numeric=(plus-minus)/(2*eps);if(fabs(numeric-analytic[j])>1e-5*(1+fabs(numeric))){fprintf(stderr,"gradient mismatch mode=%d index=%d analytic=%g numeric=%g\n",mode,idx,analytic[j],numeric);return 1;}
        }
    }
    for(int comp=0;comp<10;comp++){
        compressor=comp;t.used=0;for(int i=0;i<m.count;i++)constant(m.p[i]);
        Block a=leaf(&m,0,0),b=leaf(&m,4,1);pool_leaf(&m,&a);pool_leaf(&m,&b);Block ab=combine(&m,a,b),ba=combine(&m,b,a);
        double diff=0;for(int j=0;j<D;j++)diff+=fabs(t.n[ab.h[j]].v-t.n[ba.h[j]].v);
        assert((comp==0||comp>=6)?diff<1e-12:diff>1e-6);
    }
    /* Score reuse must preserve every parameter gradient, not just predictions. */
    compressor=7;binding_kind=0;gated_readout=2;
    for(int j=0;j<D;j++)m.p[m.span+j]=0.07*(j+1);
    int pred0,pred1,blocks;score_cache=0;
    int root=forward(&m,tokens,32,4,1,target,&pred0,&blocks);
    double reference_loss=t.n[root].v,grads[MAX_PARAMS];backward(root);
    for(int j=0;j<m.count;j++)grads[j]=t.n[j].g;
    score_cache=1;root=forward(&m,tokens,32,4,1,target,&pred1,&blocks);backward(root);
    assert(pred0==pred1&&fabs(t.n[root].v-reference_loss)<1e-12);
    for(int j=0;j<m.count;j++)assert(fabs(t.n[j].g-grads[j])<1e-10*(1+fabs(grads[j])));
    /* The mass summary is associative for fixed encoded leaves. */
    compressor=8;t.used=0;for(int j=0;j<m.count;j++)constant(m.p[j]);
    Block leaves[4];for(int j=0;j<4;j++){leaves[j]=leaf(&m,j,j);pool_leaf(&m,&leaves[j]);}
    Block abcd=combine(&m,combine(&m,leaves[0],leaves[1]),combine(&m,leaves[2],leaves[3]));
    Block acbd=combine(&m,combine(&m,leaves[0],leaves[2]),combine(&m,leaves[1],leaves[3]));
    for(int j=0;j<D;j++)assert(fabs(t.n[abcd.h[j]].v-t.n[acbd.h[j]].v)<1e-12);
    assert(fabs(t.n[abcd.mass].v-t.n[acbd.mass].v)<1e-12);
    distinct_values=0;
    for(int kind=0;kind<4;kind++)for(int n=16;n<=128;n*=2){
        int input[128];task_mode=kind;rng=123;
        for(int iter=0;iter<100;iter++){
            int answer=sample(input,n),found[4]={0},values[4]={0};
            for(int i=0;i<n;i++)assert(input[i]>=0&&input[i]<V);
            for(int i=0;i<n-5;i++)if(input[i]<4){assert(input[i+1]>=4&&input[i+1]<14);assert(input[i+2]==token_id(';'));found[input[i]]++;values[input[i]]=input[i+1];}
            for(int key=0;key<4;key++)assert(found[key]==1);
            assert(input[n-3]==token_id('?')&&input[n-1]==token_id('='));assert(values[input[n-2]]==answer);
        }
    }
    distinct_values=1;task_mode=3;rng=99;
    for(int iter=0;iter<100;iter++){
        int input[64],seen_values[10]={0};sample(input,64);
        for(int j=0;j<59;j++)if(input[j]<4){int digit=input[j+1]-4;assert(digit>=0&&digit<10);assert(!seen_values[digit]);seen_values[digit]=1;}
    }
    distinct_values=0;
    free(t.n);puts("gradient and block-count checks passed");return 0;
}
