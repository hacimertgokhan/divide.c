#define main divide_program_main
#include "divide.c"
#undef main
#include <assert.h>
int main(void){
    Model m;int tokens[32];init(&m,42);rng=77;int target=sample(tokens,32);
    for(int readout=0;readout<2;readout++)for(int comp=0;comp<2;comp++)for(int mode=0;mode<2;mode++){
        gated_readout=readout;compressor=comp;
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
    for(int comp=0;comp<2;comp++){
        compressor=comp;t.used=0;for(int i=0;i<m.count;i++)constant(m.p[i]);
        Block a=leaf(&m,0,0),b=leaf(&m,4,1),ab=combine(&m,a,b),ba=combine(&m,b,a);
        double diff=0;for(int j=0;j<D;j++)diff+=fabs(t.n[ab.h[j]].v-t.n[ba.h[j]].v);
        assert(comp?diff>1e-6:diff<1e-12);
    }
    free(t.n);puts("gradient and block-count checks passed");return 0;
}
