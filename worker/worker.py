import pika
import json
import cv2  # Assuming you're using OpenCV for image processing
import base64
import controller
import save


# RabbitMQ connection details (replace with your own)
connection_parameters = pika.ConnectionParameters(host='35.239.236.31')


def save_image(filename):
    with open(filename, 'rb') as f:
        image_data = f.read()
    return image_data

def on_message_received(channel, method, properties, body):
  try:
    # Parse task message
    
    # processing_type = task_message['processing_type']  # Add processing type key if used
    
    # cv2.imdecode(np.fromstring(image_data, np.uint8), cv2.IMREAD_COLOR)
    # Process image
    # processed_image_data = process_image(image_data, "grayscale")
    request = body.decode('utf-8').split('\n')
    id = body.decode('utf-8').split('\n\n')[-1]
    print(id)
    image_data = base64.b64decode(request[0])
    process = request[1]
    print(f"Processing image with type: {process}")
    filename ='images/recieved_image.jpg'
    with open (filename, 'wb') as f:
        f.write(image_data)
    
    if process == 'blur':
      controller.service_blur(filename, int(request[2]), id)
    elif process == 'grayscale':
      controller.service_grayscale(filename, id)
    elif process == 'rotate':
      controller.service_rotate(filename, int(request[2]), id)
    elif process == 'canny':
      controller.service_canny(filename, int(request[2]), int(request[3]), id)
    elif process == 'flip':
      controller.service_flip_image(filename, int(request[2]), id)
    # elif process == 'dilation':
    #   controller.service_apply_dilation(filename, int(request[2]), int(request[3]))
    # elif process == 'erosion':
    #   controller.service_apply_erosion(filename, int(request[2]), int(request[3]))
    # elif process == 'contours':
    #   controller.service_draw_contours(filename, int(request[2]))
    elif process == 'median blur':
      controller.service_apply_median_blur(filename, int(request[2]), id)
    elif process == 'sobel':
      controller.service_apply_sobel(filename, int(request[2]), int(request[3]), int(request[4]), id)
    elif process == 'histogram':
      controller.service_equalize_histogram(filename, id)
    elif process == 'hough':
      controller.service_find_and_draw_hough_lines(filename, int(request[2]), int(request[3]), int(request[4]), id)
    elif process == 'harris':
      controller.service_apply_harris_corner_detection(filename, int(request[2]), int(request[3]), float(request[4]), id)

    save.upload_blob('focussnap', f'images/{id}.jpg', f'{id}.jpg')

    # Acknowledge task completion
    channel.basic_ack(delivery_tag=method.delivery_tag)
    print(f"Finished processing image with type: {process}")
  except Exception as e:
    print(f"Error processing image: {e}")
    # Handle errors (e.g., retry failed tasks)

def main():
  connection = pika.BlockingConnection(connection_parameters)
  channel = connection.channel()

  # Declare the queue for image processing tasks
  channel.queue_declare(queue='image_processing_tasks')

  # Consume messages from the queue
  channel.basic_consume(queue='image_processing_tasks', on_message_callback=on_message_received)

  print("Worker is ready to process images...")
  channel.start_consuming()

if __name__ == '__main__':
  import cv2  # Import OpenCV if you're using it
  import numpy as np  # Import NumPy if you need it for image decoding
  main()
