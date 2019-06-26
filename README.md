# Lifting-from-the-Deep-release-for-video
Change "Lifting from the Deep: Convolutional 3D Pose Estimation from a Single Image" to video input implementation

This project changes from [Lifting from the Deep: Convolutional 3D Pose Estimation from a Single Image](http://openaccess.thecvf.com/content_cvpr_2017/papers/Tome_Lifting_From_the_CVPR_2017_paper.pdf), CVPR 2017

## Dependencies

The code is compatible with python3.5
- [Tensorflow 1.0](https://www.tensorflow.org/)
- [OpenCV](http://opencv.org/)

## Run Video Part
- First, run `setup.sh` to retreive the trained models and to install the external utilities.
- Run `video.py` to receive the video passed from redis and save the results to the databases.
- Or run `video_local.py` to read the local video and save the results to the databases.

## Citation

    @InProceedings{Tome_2017_CVPR,
    author = {Tome, Denis and Russell, Chris and Agapito, Lourdes},
    title = {Lifting From the Deep: Convolutional 3D Pose Estimation From a Single Image},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {July},
    year = {2017}
    }

## References

- [Lifting from the Deep: Convolutional 3D Pose Estimation from a Single Image](https://github.com/DenisTome/Lifting-from-the-Deep-release).
