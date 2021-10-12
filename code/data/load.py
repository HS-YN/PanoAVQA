import torch

from exp import ex
from .qa import get_qa
from .video import get_video
from .tokenizer import get_tokenizer


@ex.capture()
def load(modes, cache_path, transformer_name, rebuild_cache):
    if isinstance(modes, str):
        modes = modes
    modes = sorted(list(modes))
    cache_path.mkdir(parents=True, exist_ok=True)
    cache_files = [f"{mode}_{transformer_name}.pkl" for mode in modes]

    data = {}
    tokenizer = get_tokenizer()
    print(f'[LOG] Loading cached QA from {cache_path}...', end='', flush=True)
    for mode, cache_file in zip(modes, cache_files):
        path = cache_path / cache_file
        if rebuild_cache and path.is_file():
            path.unlink()
        if path.is_file():
            data[mode] = torch.load(path)
        else:
            qa = get_qa(tokenizer, data=None, mode=mode)
            torch.save(qa, path)
            data[mode] = qa
    print('Complete!') 
    video = get_video(mode=mode)

    return data, video, tokenizer