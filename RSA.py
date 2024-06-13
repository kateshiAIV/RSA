from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
from PIL import Image
import os

def load_private_key(file_path):
    with open(file_path, "rb") as key_file:
        private_key = serialization.load_pem_private_key(
            key_file.read(),
            password=None,
            backend=default_backend()
        )
    return private_key

def sign_image(private_key, image_path):
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()

    signature = private_key.sign(
        image_data,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256()
    )

    return signature

def save_signature(signature, file_path):
    with open(file_path, "wb") as signature_file:
        signature_file.write(signature)

def main():
    # Load the private key
    private_key_path = "Nadawca\private_key123123.pem"
    private_key = load_private_key(private_key_path)

    # Sign the image
    image_path = "TESTIMAGE_copy.jpg"
    signature = sign_image(private_key, image_path)

    # Save the signature
    signature_path = "TESTIMAGE_copy.signature"
    save_signature(signature, signature_path)

    print(f"Image signed successfully. Signature saved to {signature_path}")

if __name__ == "__main__":
    main()
