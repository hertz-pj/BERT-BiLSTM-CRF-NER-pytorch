import json

def _read_json(input_file, mode="train"):
    lines = []
    with open(input_file,'r') as f:
        for line in f:
            line = json.loads(line.strip())
            text = line['text']
            label_entities = line.get('label',None)
            words = list(text)
            labels = ['O'] * len(words)
            if label_entities is not None:
                for key,value in label_entities.items():
                    for sub_name,sub_index in value.items():
                        for start_index,end_index in sub_index:
                            assert  ''.join(words[start_index:end_index+1]) == sub_name
                            if start_index == end_index:
                                labels[start_index] = 'B-'+key
                            else:
                                labels[start_index] = 'B-'+key
                                labels[start_index+1:end_index+1] = ['I-'+key]*(len(sub_name)-1)
            lines.append({"words": words, "labels": labels})
    
    with open(f"/raid/ypj/openSource/cluener_public/{mode}.txt", "w") as f:
        for line in lines:
            for w, l in zip(line["words"], line["labels"]):
                f.write(f"{w}\t{l}\n")
            f.write("\n")

def get_entity_bio(seq):
    """Gets entities from sequence.
    note: BIO
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        get_entity_bio(seq)
        #output
        [['PER', 0,1], ['LOC', 3, 3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        # if not isinstance(tag, str):
        #     tag = id2label[tag]
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
            chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx

            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks

if __name__ == "__main__":
    _read_json("/raid/ypj/openSource/cluener_public/train.json", "train")
    _read_json("/raid/ypj/openSource/cluener_public/dev.json", "dev")
    _read_json("/raid/ypj/openSource/cluener_public/test.json", "test")

    # with open("./model/clue/token_labels_.txt") as f:
    #     lines = [line.strip().split(" ") for line in f.readlines()]
    
    # label_seq = []
    # token_seq = []

    # texts = []
    # labels = []  

    # for id_, line in enumerate(lines):

    #     if len(line) == 3:
    #         token_seq.append(line[0])
    #         label_seq.append(line[2])
    
    #     if len(line) != 3:
    #         texts.append(token_seq)
    #         labels.append(label_seq)

    #         token_seq = []
    #         label_seq = []

    # if token_seq:
    #     texts.append(token_seq)
    #     labels.append(label_seq)

    # test_submit = []
    # for id_, (token_seq, label_seq) in enumerate(zip(texts, labels)):
    #     json_d = {}
    #     json_d['id'] = str(id_)
    #     json_d['label'] = {}

    #     chunks = get_entity_bio(label_seq)

    #     if len(chunks) != 0:
    #         for subject in chunks:
    #             tag = subject[0]
    #             start = subject[1]
    #             end = subject[2]
    #             word = "".join(token_seq[start:end + 1])
    #             if tag in json_d['label']:
    #                 if word in json_d['label'][tag]:
    #                     json_d['label'][tag][word].append([start, end])
    #                 else:
    #                     json_d['label'][tag][word] = [[start, end]]
    #             else:
    #                 json_d['label'][tag] = {}
    #                 json_d['label'][tag][word] = [[start, end]]
    #     test_submit.append(json_d)
    
    # with open("./model/clue/submit.json", "w") as f:
    #     for line in test_submit:
    #         f.write(json.dumps(line, ensure_ascii=False)+"\n")
    