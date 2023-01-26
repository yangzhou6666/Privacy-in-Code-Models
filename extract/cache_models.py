'''Download all the necessary models from HuggingFace'''

from transformers import AutoTokenizer, AutoModelForCausalLM
from extract import get_model_and_tokenizer

if __name__ == '__main__':
    models = [
        'facebook/incoder-6B',
        'facebook/incoder-1B',
        'Salesforce/codegen-350M-multi',
        'Salesforce/codegen-350M-nl',
        'Salesforce/codegen-350M-mono',
        'Salesforce/codegen-2B-multi',
        'Salesforce/codegen-2B-nl',
        'Salesforce/codegen-2B-mono',
        'Salesforce/codegen-6B-multi',
        'Salesforce/codegen-6B-nl',
        'Salesforce/codegen-6B-mono',
        'codeparrot/codeparrot-small',
        'codeparrot/codeparrot'
    ]

    for model_name in models:
        get_model_and_tokenizer(model_name)