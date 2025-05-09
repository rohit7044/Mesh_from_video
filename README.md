# Mesh from Video
A short project on creating mesh from video using 3DDFA-V3

Follow the [3DDFA-V3](https://github.com/wang-zidu/3DDFA-V3) repository for installation

## Example
~~~
cd 3DDFA-V3
~~~

~~~
python demo_video.py -i "Input Video/Obama/xyz.mp4" -s "Output Video"
~~~

I was trying for blendshape estimation by following this paper [A Greedy Pursuit Approach for Fitting 3D Facial Expression Models](https://ieeexplore.ieee.org/document/9214483).
But I realized that blendshape estimation is actually not required for 3DMM and FLAME as they already have identity and expression coefficients. By using those data we can easily estimate the the expression and facial features.

Wasted few days only to realize it doesn't work :(

But I have the paper implementation of the blendshape code and also a sanity check. Maybe it can be of use for someone who wants to use classical way to estimate blendshape.
