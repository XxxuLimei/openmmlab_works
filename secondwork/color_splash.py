import numpy as np
import cv2
import mmcv

from mmdet.apis import inference_detector, init_detector

def main():
    config_file = "./balloon.py"
    checkpoint_file = "./work_dirs/balloon/epoch_10.pth"

    model = init_detector(config_file, checkpoint_file)

    video_reader = mmcv.VideoReader("test_video.mp4")
    video_writer = None
    if True:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            "test_out.mp4", fourcc, video_reader.fps,
            (video_reader.width, video_reader.height))

    for frame in mmcv.track_iter_progress(video_reader):
        result = inference_detector(model, frame)
        mask = None
        masks = result[1][0]
        for i in range(len(masks)):
            if result[0][0][i][-1] >= 0.8:
                if not mask is None:
                    mask = mask | masks[i]
                else:
                    mask = masks[i]

        masked_b = frame[:, :, 0] * mask
        masked_g = frame[:, :, 1] * mask
        masked_r = frame[:, :, 2] * mask
        masked = np.concatenate([masked_b[:,:,None],masked_g[:,:,None],masked_r[:,:,None]],axis=2)

        un_mask = 1 - mask
        frame_b = frame[:, :, 0] * un_mask
        frame_g = frame[:, :, 1] * un_mask
        frame_r = frame[:, :, 2] * un_mask
        frame = np.concatenate([frame_b[:, :, None], frame_g[:, :, None], frame_r[:, :, None]], axis=2).astype(np.uint8)
        frame = mmcv.bgr2gray(frame, keepdim=True)
        frame = np.concatenate([frame, frame, frame], axis=2)

        frame += masked
        
        if True:
            cv2.namedWindow('video', 0)
            mmcv.imshow(frame, 'video', 1)
        if True:
            video_writer.write(frame)

    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()