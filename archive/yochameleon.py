# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Chameleon License found in the
# LICENSE file in the root directory of this source tree.

import argparse

from chameleon.inference.chameleon import ChameleonInferenceModel

# def main():
#     model = ChameleonInferenceModel(
#         "./data/models/7b/",
#         "./data/tokenizer/text_tokenizer.json",
#         "./data/tokenizer/vqgan.yaml",
#         "./data/tokenizer/vqgan.ckpt",
#     )

#     tokens = model.generate(
#         prompt_ui=[
#             {"type": "image", "value": "file:/mnt/localssd/code/YoLLaVA/yollava-data/train/bo/0.png"},
#             {"type": "text", "value": "What is the breed of the dog?"},
#             # {"type": "sentinel", "value": "<END-OF-TURN>"},
#         ]
#     )
#     breakpoint()
#     print(model.decode_text(tokens)[0])


if __name__ == "__main__":

    model = ChameleonInferenceModel(
        "./data/models/7b/",
        "./data/tokenizer/text_tokenizer.json",
        "./data/tokenizer/vqgan.yaml",
        "./data/tokenizer/vqgan.ckpt",
    )

    # tokens = model.generate(
    #     prompt_ui=[
    #         {"type": "image", "value": "file:/mnt/localssd/code/YoLLaVA/yollava-data/train/bo/0.png"},
    #         {"type": "text", "value": "What is the breed of the dog?"},
    #         # {"type": "sentinel", "value": "<END-OF-TURN>"},
    #     ]
    # )
    # print(model.decode_text(tokens)[0])

    # Add tokens
    num_ori_tokens = model.token_manager.tokenizer.get_vocab_size()
    personalized_tokens = ['<thao>']
    num_of_added_tokens = model.token_manager.tokenizer.add_tokens(personalized_tokens)
    num_ori_tokens_after = model.token_manager.tokenizer.get_vocab_size()

    print('Original tokens: ', num_ori_tokens, '\nAdded: ', num_of_added_tokens, '\nNew tokens: ', num_ori_tokens_after)
    
    # Check if sentence are encoded correctly
    test_sentence = 'Hi, this is <thao>'
    encoded_ids = model.token_manager.tokenizer.encode(test_sentence).ids
    decoded_sentence = model.token_manager.tokenizer.decode(encoded_ids)
    print('Original sentence: ', test_sentence)
    print('Encoded ids: ', encoded_ids)
    print('Decoded sentence: ', decoded_sentence)

    breakpoint()

    
    # main()
