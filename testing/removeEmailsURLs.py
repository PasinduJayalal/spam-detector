import re

def clean_text(text, lower=True, remove_urls=True, remove_emails=True):
    if remove_urls:
        text = re.sub(r'https?://\S+|www\.\S+', '', text, flags=re.IGNORECASE)

    if remove_emails:
        text = re.sub(r'\b[\w\.-]+?@\w+?\.\w{2,4}\b', '', text)

    if lower:
        text = text.lower()

    # Optional: remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Example usage
text = "Contact us at support@example.com or visit https://example.com for more info."
print(clean_text(text))
# print(clean_text(text, remove_urls=False, remove_emails=False))  
