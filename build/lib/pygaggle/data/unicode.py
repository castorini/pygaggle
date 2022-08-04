def convert_to_unicode(text):
    """
    Converts `text` to Unicode (if it's not already) assuming utf-8 input."""
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError(f"Unsupported string type: f{type(text)}")
