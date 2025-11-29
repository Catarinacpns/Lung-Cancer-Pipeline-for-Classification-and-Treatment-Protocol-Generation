import tiktoken

def chunk_text(text, max_tokens, overlap):
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    num_tokens = len(tokens)

    if num_tokens <= max_tokens:
        return [text]

    chunks = []
    step_size = max(1, int(max_tokens * (1 - overlap)))

    for i in range(0, num_tokens, step_size):
        chunk_tokens = tokens[i: i + max_tokens]
        chunks.append(enc.decode(chunk_tokens))

    return chunks

def chunk_text_openai(text, chunk_sizes=[1000], overlaps=[0.2]):
    tokens = enc.encode(text)
    num_tokens = len(tokens)

    if num_tokens < min(chunk_sizes):
        #print(f"Small doc stored but tagged as chunk size {min(chunk_sizes)}")
        return {f"size_{min(chunk_sizes)}_overlap_{overlaps[0]}": [text]}  


    chunked_texts = {}
    for max_tokens in chunk_sizes:
        for overlap in overlaps:
            chunks = []
            step = max(1, int(max_tokens * (1 - overlap)))

            for i in range(0, num_tokens, step):
                chunk_tokens = tokens[i: i + max_tokens]
                chunks.append(enc.decode(chunk_tokens))

            #print(f"Chunk Size: {max_tokens}, Overlap: {overlap}, Chunks Created: {len(chunks)}")
            chunked_texts[f"size_{max_tokens}_overlap_{overlap}"] = chunks  

    return chunked_texts