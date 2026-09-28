"""Versioned character vocabulary shared by new V3 training and inference."""
CHARACTERS="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?'\"-:;()"
CHAR_TO_ID={char:i+2 for i,char in enumerate(CHARACTERS)}

def encode_char_v1(text, max_tokens=1024):
    # 0 is padding, 1 is unknown; avoid interpreting unknown characters as 'a'.
    return [CHAR_TO_ID.get(char,1) for char in str(text).strip()][:max_tokens]
