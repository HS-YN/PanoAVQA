import torch
from transformers import AutoTokenizer

from exp import ex


token_dict = {
    'bert': 'bert-base-uncased'
}


@ex.capture()
def get_tokenizer(cache_path, transformer_name, rebuild_cache):
    # Provide tokenizer with caching
    tokenizer_path = cache_path / 'tokenizer'
    tokenizer_path.mkdir(parents=True, exist_ok=True)
    tokenizer_file = f"{transformer_name}.pkl"
    path = tokenizer_path / tokenizer_file

    if rebuild_cache and path.is_file():
        path.unlink()
    if path.is_file():
        tokenizer = torch.load(path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(token_dict[transformer_name],
                                                  do_lower_case=True)
        torch.save(tokenizer, path)
    return tokenizer