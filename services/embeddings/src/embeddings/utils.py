import numpy as np
import torch


def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = np.expand_dims(attention_mask, axis=-1).astype(float)
    return np.sum(token_embeddings * input_mask_expanded, axis=1) / np.clip(
        np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None
    )


def encode_texts(
    texts: list[str], model, tokenizer, device
) -> tuple[np.ndarray, np.ndarray]:
    encoded_input = tokenizer(
        texts, padding=True, truncation=True, return_tensors="pt", max_length=512
    ).to(device)

    with torch.no_grad():
        model_output = model(**encoded_input, output_hidden_states=True)

    last_hidden = model_output.last_hidden_state.cpu().numpy()
    attention_mask = encoded_input["attention_mask"].cpu().numpy()

    embeddings = mean_pooling(last_hidden, attention_mask)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    final_embeddings = embeddings / norms

    l6_hidden = model_output.hidden_states[6].cpu().numpy()
    l6_embeddings = mean_pooling(l6_hidden, attention_mask)
    l6_norms = np.linalg.norm(l6_embeddings, axis=1, keepdims=True)
    l6_final = l6_embeddings / l6_norms

    return final_embeddings, l6_final
