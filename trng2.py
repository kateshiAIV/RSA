from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
from Cryptodome.PublicKey import RSA
from PIL import Image
import cv2
import pyaudio
import wave
import numpy as np
import os
import mss
import pyautogui
import time



def capture_video_frame(top,left,width,height,filename):
    with mss.mss() as sct:
        # Capture a region of the screen (change the coordinates and size as needed)
        monitor = {"top": top, "left": left, "width": width, "height": height}
        # Capture a single screenshot
        screenshot = sct.grab(monitor)
        # Convert the screenshot to PIL Image format
        frame = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
        frame.save(filename)
        return frame
    
def generate_random_numbers_from_live_image(image):
    # Convert image to grayscale
    image = image.convert('L')
    pixel_data = np.array(image)
    random_numbers = []
    for row in pixel_data:
        for pixel in row:
            voltage = pixel / 255.0  # Normalize pixel intensity to [0, 1]
            random_number = voltage + np.random.normal(scale=0.1)
            random_numbers.append(random_number)
    return random_numbers


def capture_audio():
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 512
    RECORD_SECONDS = 5
    WAVE_OUTPUT_FILENAME = "recordedFile.wav"
    device_index = 0
    audio = pyaudio.PyAudio()

    print("----------------------record device list---------------------")
    info = audio.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    for i in range(0, numdevices):
            if (audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                print("Input Device id ", i, " - ", audio.get_device_info_by_host_api_device_index(0, i).get('name'))

    print("-------------------------------------------------------------")

    index = int(device_index)
    print("recording via index "+str(index))

    stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,input_device_index = index,
                    frames_per_buffer=CHUNK)
    print ("recording started")
    Recordframes = []
    
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        Recordframes.append(data)
    print ("recording stopped")
    
    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(Recordframes))
    waveFile.close()
















def read_binary_file(file_path, max_bytes=1000000):
    with open(file_path, 'rb') as file:
        binary_data = file.read(max_bytes)
    return binary_data

def display_binary_stream(binary_data):
    binary_stream = ' '.join(format(byte, '08b') for byte in binary_data)
    print("Strumień binarny:")
    print(binary_stream)

def save_to_file(data, filename):
    with open(filename, 'ab') as file:  # Open file in binary mode
        file.write(data)


def resize_image(image_path, target_size=(1024, 1024)):
    """
    Resize the image to the target size while keeping the center of the image.
    """
    image = Image.open(image_path)
    width, height = image.size
    left = (width - target_size[0]) / 2
    top = (height - target_size[1]) / 2
    right = (width + target_size[0]) / 2
    bottom = (height + target_size[1]) / 2
    cropped_image = image.crop((left, top, right, bottom))
    return cropped_image


def save_to_file(data, filename):
    with open(filename, 'wb') as file:
        file.write(data)

def save_to_file2(data, filename):
    with open(filename, 'ab') as file:
        file.write(data)

def trng(random_numbers1, random_numbers2, sound_binary, video_binary):

    random_numbers1_result_text = ''.join(str(bit) for bit in random_numbers1)
    save_to_file(random_numbers1_result_text.encode(), "random_numbers1.txt")
    random_numbers2_result_text = ''.join(str(bit) for bit in random_numbers2)
    save_to_file(random_numbers2_result_text.encode(), "random_numbers2.txt")
    sound_binary_result_text = ''.join(str(bit) for bit in sound_binary)
    save_to_file(sound_binary_result_text.encode(), "sound_binary.txt")
    video_binary_result_text = ''.join(str(bit) for bit in sound_binary)
    save_to_file(video_binary_result_text.encode(), "video_binary.txt")

    M1 = []
    M2 = []
    N1 = []
    N2 = [] 

    min_length = min(len(sound_binary), len(random_numbers1), len(random_numbers2), len(video_binary))

    for i in range(min_length):
        M1_i = (sound_binary[i] * sound_binary[i]) % (random_numbers1[i] * 15485863)
        M2_i = (video_binary[i] * video_binary[i]) % (random_numbers2[i] * 15485863)
        N1_i = M1_i % 2
        N2_i = M2_i % 2

        M1.append(M1_i)
        M2.append(M2_i)
        N1.append(int(N1_i))  # Convert to integer explicitly
        N2.append(int(N2_i))  # Convert to integer explicitly

    xor_result = [int(N1_i) ^ int(N2_i) for N1_i, N2_i in zip(N1, N2)]  # Convert to integer explicitly

    # Zapis wyniku do pliku tekstowego
    result_text = ''.join(str(bit) for bit in xor_result)
    save_to_file2(result_text.encode(), "random_output2.txt")
    save_to_file2(result_text.encode(), "random_output2.bin")
    return result_text


def read_image(file_path):
    image = Image.open(file_path)
    # Convert image to grayscale if needed
    image = image.convert('L')
    pixel_data = np.array(image)
    return pixel_data

# Function to generate random numbers from image data
# def generate_random_numbers(image_data):
#     random_numbers = []
#     for row in image_data:
#         for pixel in row:
#             # Simulate voltage generation from pixel intensity
#             voltage = pixel / 255.0  # Normalize pixel intensity to [0, 1]
#             # Introduce randomness using dark noise
#             random_number = voltage + np.random.normal(scale=0.1)  # Add Gaussian noise
#             random_numbers.append(random_number)
#     return random_numbers

def generate_random_numbers(image_data):
    random_numbers = []
    for row in image_data:
        for pixel in row:
            # Simulate voltage generation from pixel intensity
            voltage = pixel / 255.0  # Normalize pixel intensity to [0, 1]
            # Introduce randomness using dark noise
            random_number = voltage + np.random.normal(scale=0.1)  # Add Gaussian noise
            random_numbers.append(random_number)
    return random_numbers


def record_screen_part(top, left, width, height, duration, output_file):
    # Zdefiniuj rozmiar i częstotliwość klatek dla nagranego wideo
    screen_size = (width, height)
    frame_rate = 30.0

    # Utwórz obiekt VideoWriter do zapisywania wideo
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_file, fourcc, frame_rate, screen_size)

    # Nagrywaj przez określony czas
    start_time = time.time()
    while (time.time() - start_time) < duration:
        # Przechwyć zrzut ekranu
        screenshot = pyautogui.screenshot(region=(left, top, width, height))
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Zapisz klatkę do wideo
        out.write(frame)

    # Zwolnij zasoby i zakończ nagrywanie
    out.release()
    cv2.destroyAllWindows()


def generate_rsa_key_pair(seed_data):
    np.random.seed(seed_data)
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend()
    )
    return private_key

def generate_rsa_key_from_binary_file(binary_file):
    with open(binary_file, "rb") as f:
        binary_data = f.read()
    seed_data = int.from_bytes(binary_data, byteorder='big') % (2**32)
    private_key = generate_rsa_key_pair(seed_data)
    return private_key






class CustomRandom:
    def __init__(self):
        self.random_bits = run_TRNG()
        self.index = 0

    def read(self, N):
        # If we run out of random bits, generate more
        if self.index + N*8 >= len(self.random_bits):
            print("Generating more random bits...")
            self.random_bits += run_TRNG()
            
        result = self.random_bits[self.index:self.index + (N*8)]
        self.index += N*8
        
        # Convert the bits to bytes
        bytes_result = bytes(int(''.join(map(str, result[i:i+8])), 2) for i in range(0, len(result), 8))
        return bytes_result

def generate_rsa_key_from_binary_file():
    custom_random = CustomRandom()
    
    # Generate RSA private key
    key = RSA.generate(2048, randfunc=lambda N: custom_random.read(N))
    return key.publickey(), key





def save_key_to_file(key, file_path, key_type):
    with open(file_path, 'wb') as file:
        if key_type == 'public':
            file.write(key.export_key(format='PEM'))
        elif key_type == 'private':
            file.write(key.export_key(format='PEM'))

def run_TRNG():
    random_sequence = ""
    live_video_frame = capture_video_frame(0,0,960,520,"captured_frame.png")
    if live_video_frame:
        random_numbers1 = generate_random_numbers_from_live_image(live_video_frame)

        # Capture another live video frame
        another_live_video_frame = capture_video_frame(0,960,960,520,"captured_frame2.png")
        if another_live_video_frame:
            random_numbers2 = generate_random_numbers_from_live_image(another_live_video_frame)
            capture_audio()
            sound_binary = read_binary_file("recordedFile.wav")
            record_screen_part(520,960,960,520, 5, "recorded_screen_part.avi")
            video_binary = read_binary_file("recorded_screen_part.avi")
            random_sequence += trng(random_numbers1, random_numbers2, sound_binary, video_binary)

    live_video_frame = capture_video_frame(520,0,960,520,"captured_frame.png")
    if live_video_frame:
        random_numbers1 = generate_random_numbers_from_live_image(live_video_frame)

        # Capture another live video frame
        another_live_video_frame = capture_video_frame(520,960,960,520,"captured_frame2.png")
        if another_live_video_frame:
            random_numbers2 = generate_random_numbers_from_live_image(another_live_video_frame)
            capture_audio()
            sound_binary = read_binary_file("recordedFile.wav")
            record_screen_part(0,0,960,520, 5, "recorded_screen_part.avi")
            video_binary = read_binary_file("recorded_screen_part.avi")
            random_sequence+=trng(random_numbers1, random_numbers2, sound_binary, video_binary)

    live_video_frame = capture_video_frame(520,0,960,520,"captured_frame.png")
    if live_video_frame:
        random_numbers1 = generate_random_numbers_from_live_image(live_video_frame)

        # Capture another live video frame
        another_live_video_frame = capture_video_frame(0,960,960,520,"captured_frame2.png")
        if another_live_video_frame:
            random_numbers2 = generate_random_numbers_from_live_image(another_live_video_frame)
            capture_audio()
            sound_binary = read_binary_file("recordedFile.wav")
            record_screen_part(520,0,960,520, 5, "recorded_screen_part.avi")
            video_binary = read_binary_file("recorded_screen_part.avi")
            random_sequence+=trng(random_numbers1, random_numbers2, sound_binary, video_binary)

    live_video_frame = capture_video_frame(0,960,960,520,"captured_frame.png")
    if live_video_frame:
        random_numbers1 = generate_random_numbers_from_live_image(live_video_frame)

        # Capture another live video frame
        another_live_video_frame = capture_video_frame(520,0,960,520,"captured_frame2.png")
        if another_live_video_frame:
            random_numbers2 = generate_random_numbers_from_live_image(another_live_video_frame)
            capture_audio()
            sound_binary = read_binary_file("recordedFile.wav")
            record_screen_part(0,960,960,520, 5, "recorded_screen_part.avi")
            video_binary = read_binary_file("recorded_screen_part.avi")
            random_sequence+=trng(random_numbers1, random_numbers2, sound_binary, video_binary)

    return random_sequence


def main():

    #open("random_output.txt", 'w')  # Open file in binary mode   
    # Read image data
    # image_data1 = read_image('image1.jpg')
    # # Generate random numbers from image data

    # random_numbers1 = generate_random_numbers(image_data1)

    # # Read another image data
    # image_data2 = read_image('image2.jpg')
    # # Generate random numbers from the second image data
    # random_numbers2 = generate_random_numbers(image_data2)
    
    
    # # Read binary files for sound and video
    # sound_binary = read_binary_file('sound2.mp3')
    # video_binary = read_binary_file('video.mov')

    # # Call the trng function to perform the random number generation
    # trng(random_numbers1, random_numbers2, sound_binary, video_binary)

    # image_data1 = read_image('picture2.jpg')
    # # Generate random numbers from image data
    

    
    # random_numbers1 = generate_random_numbers(image_data1)

    # # Read another image data
    # image_data2 = read_image('picture2_2.jpg')
    # # Generate random numbers from the second image data
    # random_numbers2 = generate_random_numbers(image_data2)
    
    
    # # Read binary files for sound and video
    # sound_binary = read_binary_file('sound.mp3')
    # video_binary = read_binary_file('video2.mp4')

    # # Call the trng function to perform the random number generation
    # trng(random_numbers1, random_numbers2, sound_binary, video_binary)







    binary_file_path = "random_output.bin"
    public_key, private_key = generate_rsa_key_from_binary_file()

    save_key_to_file(private_key, 'Nadawca/private_key123123.pem', 'private')
    save_key_to_file(public_key, 'Odbiorca/public_key123123.pem', 'public')

    # private_key = generate_rsa_key_from_binary_file("random_output2.bin")


    # # Save the private key
    # with open("private_key2.pem", "wb") as f:
    #     f.write(private_key.private_bytes(
    #         encoding=serialization.Encoding.PEM,
    #         format=serialization.PrivateFormat.PKCS8,
    #         encryption_algorithm=serialization.NoEncryption()
    #     ))

    # # Save the public key
    # public_key = private_key.public_key()
    # with open("public_key2.pem", "wb") as f:
    #     f.write(public_key.public_bytes(
    #         encoding=serialization.Encoding.PEM,
    #         format=serialization.PublicFormat.SubjectPublicKeyInfo
    #     ))



            






if __name__ == "__main__":
    main()






