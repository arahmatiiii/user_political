def word_tokenizer(texts: list, tokenizer_obj) -> list:
    """

    :param texts: list of sents ex: ["first sent", "second sent"]
    :param tokenizer_obj:
    :return: list of tokenized words ex: [["first", "sent"], ["second", "sent"]]
    """
    return [tokenizer_obj(text) for text in texts]
