
import numpy as np
import os
import cv2

class EpisodeRecorder():

    def __init__(self,save_frames=False,save_frames_dir = './frames'):
        self.frames=[]
        self.actions=[]
        self.save_frames=save_frames
        self.save_frames_dir = save_frames_dir

        if self.save_frames:
            os.makedirs('./frames',exist_ok=True)

    def record_frame(self,pov):
        # copy to be sure!
        self.frames.append(pov.copy())



    def record_action(self,action):
        self.actions.append(action.copy())


    def _manual_save_frames(self,dir):
        pass

    # filepath m,ust end in .avi!
    def save_vid(self,filepath):
        if self.frames:
            out = cv2.VideoWriter(filepath, cv2.VideoWriter_fourcc(*'DIVX'), 60, self.frames[0].shape[0:2])
            for frame in self.frames:
                out.write(frame)

            out.release()
        else:
            self.__error_no_record()



    def save_pov_np(self,filepath):
        frames = np.array(self.frames)
        np.savez(filepath, frames=frames)

    def save_actions(self,filepath):
        pass

    def __error_no_record(self):
        print('No frames to save!')