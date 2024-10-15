import glob

def load_context(fpath="eval/needle/PaulGrahamEssays/*.txt", ctx_len=100000):
    context = ""
    for file in glob.glob(fpath):
        with open(file, 'r') as f: 
            context += f.read()
    LLAMA_CHAR_TO_TOKEN_RATIO = 3.66
    context = context[: int(ctx_len * LLAMA_CHAR_TO_TOKEN_RATIO)]
    return context

def insert_needle(context, needle, depth):
    context = context.split(".")
    c_len = len(context)
    needle_place = int(depth * c_len)
    context = ".".join(context[:needle_place]) + "." + needle + ".".join(context[needle_place:])
    return context

