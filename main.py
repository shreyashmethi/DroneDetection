import torch
import numpy as np
import cv2
from time import time
from flask import Flask,render_template,Response
import telepot
from multiprocessing import Process


token = '5763724136:AAEyw9r1uEoghVTQ6X43xdvycV7ZM93rEtI'
receiver_id = 942215952

bot = telepot.Bot(token)

class ObjectDetection:
    """
    Class implements Yolo5 model to make inferences on a youtube video using OpenCV.
    """

    def __init__(self):
        """
        Initializes the class with youtube url and output file.
        :param url: Has to be as youtube URL,on which prediction is made.
        :param out_file: A valid output file name.
        """
        self.model = self.load_model()
        self.classes = self.model.names
        self.generate_frames = self.generate_frames()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("\n\nDevice Used:", self.device)

    def run(self, tasks):
        running_tasks = [Process(target=task) for task in tasks]
        for running_task in running_tasks:
            running_task.start()
        for running_task in running_tasks:
            running_task.join()

    def get_video_from_url(self):
        """
        Creates a new video streaming object to extract video frame by frame to make prediction on.
        :return: opencv2 video capture object, with lowest quality frame available for video.
        """
        return cv2.VideoCapture(0)

    def load_model(self):
        """
        Loads Yolo5 model from pytorch hub.
        :return: Trained Pytorch model.
        """
        device = torch.device('cuda')

        model = torch.load('best.pt', map_location=device)
        # stride = int(model.stride.max())
        cudnn.benchmark = True

        return model

    def score_frame(self, frame):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)

        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def labels(self,frame):
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels = results.xyxyn[0][:, -1]
        return labels


    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]


    def plot_boxes(self, results, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.5:
                x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(
                    row[3] * y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

        return frame

    def alarm(self):
        player = self.get_video_from_url()
        assert player.isOpened()
        while True:
            ret, frame = player.read()
            if ret is True:
                print(self.labels(frame)[0])



    def generate_frames(self):
        while True:
            player = self.get_video_from_url()
            assert player.isOpened()
            while True:
                start_time = time()
                ret, frame = player.read()
                if not ret:
                    break
                results = self.score_frame(frame)
                frame = self.plot_boxes(results, frame)
                end_time = time()
                fps = 1 / np.round(end_time - start_time, 3)
                #print(f"Frames Per Second : {fps}")

            ## read the camera frame
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()

                yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



    def launch(self):
        app = Flask(__name__)


        @app.route('/')
        def index():
            return render_template('index.html')



        @app.route('/video')
        def video():
            return Response(self.generate_frames, mimetype='multipart/x-mixed-replace; boundary=frame')

        app.run(debug=True)


# Create a new object and execute.
detection = ObjectDetection()
detection.launch()