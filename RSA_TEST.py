from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend

def load_public_key(file_path):
    with open(file_path, "rb") as key_file:
        public_key = serialization.load_pem_public_key(
            key_file.read(),
            backend=default_backend()
        )
    return public_key

def load_signature(file_path):
    with open(file_path, "rb") as signature_file:
        signature = signature_file.read()
    return signature

def verify_signature(public_key, image_path, signature):
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()

    try:
        public_key.verify(
            signature,
            image_data,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return True
    except:
        return False

def main():
    # Load the public key
    public_key_path = "Odbiorca/public_key123123.pem"
    public_key = load_public_key(public_key_path)

    # Load the signature
    signature_path = "TESTIMAGE.signature"
    signature = load_signature(signature_path)

    # Verify the image signature
    image_path = "TESTIMAGE_copy.jpg"
    is_valid = verify_signature(public_key, image_path, signature)

    if is_valid:
        print("Signature is valid.")
    else:
        print("Signature is invalid.")

if __name__ == "__main__":
    main()
