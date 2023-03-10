import cv2
import os

def main(video_file : str,
         output_directory : str,
         frame_name : str = 'frame'
        ):

    print('Reading Video File')

    if not os.path.exists(video_file):
        print('No such file: ' + video_file)
        print('Stopping')
        return

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    vidcap = cv2.VideoCapture(video_file)
    success,image = vidcap.read()
    count = 0
    print('Start Extracting Frames')
    while success:
        output_path = os.path.join(output_directory, frame_name + str(count) + ".jpg")
        print('Extracting Frame: ' + str(count) + '\t to: ' + output_path)
        cv2.imwrite(output_path, image)     # save frame as JPEG file      
        success,image = vidcap.read()
        count += 1

    print('Finished Extracting Frames')


###############
#    main     #
###############
if __name__ == '__main__':

    ### SET THESE ###
    video_file = '/Users/Synhelion/Desktop/UC Principle.mp4'
    output_directory = '/Users/Synhelion/Desktop/Frame Output'

    ### CODE ###
    main(video_file=video_file, output_directory=output_directory)