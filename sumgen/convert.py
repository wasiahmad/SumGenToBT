import os
import torch
import numpy as np


def convert(input_ckpt, out_dir, outfile, model_dim):
    model = torch.load(input_ckpt)

    # PLBART's dictionary has 49,997 tokens
    # +4 special tokens (<s>, <pad>, </s>, <unk>)
    # +3 language id tokens (<java>, <python>, <en_XX>)
    # +1 <mask> token
    # total = 50,005

    additional_languages = 1  # <cpp>
    model["args"].langs = 'java,python,en_XX,cpp'

    def append_embedding(key):
        extra_weight_tensors = torch.Tensor(
            np.random.uniform(0, 1, size=(additional_languages, model_dim))
        )
        # Last token is mask token, we must keep it at the last index
        dest = torch.cat((
            model['model'][key][:-1, :],
            extra_weight_tensors,
            model['model'][key][-1:, :],  # mask token embedding
        ), 0)
        model['model'][key] = dest

    append_embedding('encoder.embed_tokens.weight')
    append_embedding('decoder.embed_tokens.weight')
    append_embedding('decoder.output_projection.weight')

    torch.save(model, os.path.join(out_dir, outfile))


def sanity_check(model_name_or_path, checkpoint_file):
    from fairseq.models.bart import BARTModel

    # by default model["args"].task will be used for initialization
    # model["args"].task is multilingual_denoising
    bart = BARTModel.from_pretrained(
        model_name_or_path,
        checkpoint_file=checkpoint_file,
        data_name_or_path='../sentencepiece',  # for dict.txt file
    )

    assert len(bart.task.source_dictionary) == 50006
    assert bart.task.source_dictionary[0] == '<s>'
    assert bart.task.source_dictionary[1] == '<pad>'
    assert bart.task.source_dictionary[2] == '</s>'
    assert bart.task.source_dictionary[3] == '<unk>'
    assert bart.task.source_dictionary[50001] == '[java]'
    assert bart.task.source_dictionary[50002] == '[python]'
    assert bart.task.source_dictionary[50003] == '[en_XX]'
    assert bart.task.source_dictionary[50004] == '[cpp]'
    assert bart.task.source_dictionary[50005] == '<mask>'


if __name__ == '__main__':
    convert('../plbart/plbart_base.pt', '.', 'multilingual_plbart_base.pt', 768)
    sanity_check('.', 'multilingual_plbart_base.pt')
    convert('../plbart/plbart_large.pt', '.', 'multilingual_plbart_large.pt', 1024)
    sanity_check('.', 'multilingual_plbart_large.pt')
