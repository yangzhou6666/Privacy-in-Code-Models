VICTIM_MODE2MODEL_MAP = {
    # 'victim_CodeGPT-small-java-adaptedGPT2':'CodeGPT-small-java-adaptedGPT2',
    # 'victim_CodeGPT-small-java':"CodeGPT-small-java",
    # # 'victim_gpt2':"gpt2",
    # "victim_trasformer":'transformer'
}
def victim_maps(epoch=10):
    for i in range(epoch):
        VICTIM_MODE2MODEL_MAP['victim_'+str(i)] = 'checkpoint-epoch-'+str(i)+'_victim'
    return VICTIM_MODE2MODEL_MAP
        
