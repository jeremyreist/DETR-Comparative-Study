# Trackformer
Due to Trackformer explicitly requiring Python3.7, we have separated the trackformer testing section into its own
folder. To execute the Trackformer test:

1. Start a new virtual environment ON PYTHON3.7 and follow the instructions
[here]{https://github.com/timmeinhardt/trackformer/blob/main/docs/INSTALL.md} (only iv is necessary in step 3
as we are testing on MOT20). 

2. In the Trackformer folder create a new directory called output

3. Still in the Trackformer folder, run:
```
python src/track.py with \
    reid \
    dataset_name=MOT20-ALL \
    obj_detect_checkpoint_file=models/mot20_crowdhuman_deformable_multi_frame/checkpoint_epoch_50.pth  \
    output_dir=output
```

If jpg output is desired, append `write_images=pretty` to the end. This is only recommended for singular videos, so
run it in conjunction with `data_root_dir=/folder/with/images/`.

For convenience, the ffmpeg command to convert the images into video is provided

```ffmpeg -r 30 -f image2 -s 1544x1080 -i %06d.jpg -vcodec libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -crf 25  -pix_fmt yuv420p test.mp4```
