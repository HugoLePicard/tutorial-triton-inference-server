import cv2
import numpy
import tritonclient.grpc as grpcclient

from utils import *

def preprocess_image(path_image):
    image = cv2.imread(path_image)
    image = cv2.resize(image, (512, 512))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(numpy.float32) / 255.0 
    image = (image - 0.5) / 0.5              
    image = numpy.transpose(image, (2, 0, 1))   
    image = numpy.expand_dims(image, axis=0)    
    return image.astype(numpy.float16)

def main():
    path_image = os.path.join(BASE_PATH, "data", "images", "input.jpg")
    input_image_numpy = preprocess_image(path_image)

    client = grpcclient.InferenceServerClient(url="localhost:8001")

    input_tensor = grpcclient.InferInput("image", input_image_numpy.shape, "FP16")
    input_tensor.set_data_from_numpy(input_image_numpy)

    result = client.infer(
        model_name="vae_encoder_decoder",
        inputs=[input_tensor],
        outputs=[grpcclient.InferRequestedOutput("image_out")]
    )

    output = result.as_numpy("image_out")[0] 
    output = numpy.transpose(output, (1, 2, 0)) 
    output = ((output * 0.5 + 0.5) * 255).clip(0, 255).astype(numpy.uint8)

    path_output = os.path.join(BASE_PATH, "data", "images", "output.jpg")
    cv2.imwrite(path_output, cv2.cvtColor(output, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    main()