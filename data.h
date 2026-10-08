/* Shared causal key/value data generator. No target is placed in the prefix. */
static int sample_task(int *tokens,int length,int kind){
    if(kind==3)kind=random32()%3;
    int digits[4];for(int i=0;i<4;i++){int duplicate;do{digits[i]=random32()%10;duplicate=0;if(distinct_values)for(int j=0;j<i;j++)if(digits[i]==digits[j])duplicate=1;}while(duplicate);}
    if(kind==0){
        for(int i=0;i<4;i++){tokens[i*3]=i;tokens[i*3+1]=4+digits[i];tokens[i*3+2]=token_id(';');}
        for(int i=12;i<length-3;i++)tokens[i]=token_id("xyz"[random32()%3]);
    }else{
        int order[4]={0,1,2,3},gaps[5]={0};
        for(int i=3;i>0;i--){int j=random32()%(i+1),tmp=order[i];order[i]=order[j];order[j]=tmp;}
        if(kind==2)for(int i=0;i<length-15;i++)gaps[random32()%5]++;
        int pos=0;
        for(int i=0;i<4;i++){
            for(int j=0;j<gaps[i];j++)tokens[pos++]=token_id("xyz"[random32()%3]);
            int key=order[i];tokens[pos++]=key;tokens[pos++]=4+digits[key];tokens[pos++]=token_id(';');
        }
        while(pos<length-3)tokens[pos++]=token_id("xyz"[random32()%3]);
    }
    int key=random32()%4;tokens[length-3]=token_id('?');tokens[length-2]=key;tokens[length-1]=token_id('=');return 4+digits[key];
}
