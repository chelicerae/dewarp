# Barrel and pincussion distortion models

### Use different models to apply barrel and pincussion distortion to photos and video

I was challanged with task of fisheye distortion correction. The photos needed to be undistorted had no patterns like checkerboard, so I could not just calibrate the lens and pass the calibration matrix to OpenCV method. Regarding that, I came up with an idea of using pincussion distortion to fisheye distorted photos. As far as fisheye distortion can be simplified to barrel distortion (normally it consists of radial and tangential distortion, but here we omit the second one) we could just use reverse barrel distortion, which is a pincussion one. 

### Architectural outline

The pipeline can be discribed as follows: 
1. The original picture is passed.
2. Two remaping index matrix are created, for y and x axis respectivly. Their size is equall to the original image size, or may be bigger in case when paddings are used. Paddings are needed if the full picture is required. By default (with 0 padding), the resulting image is cropped due to the "streching of image" with pincussion distortion.
3. Every value of remapping matrix (lets think of it as tensor containing coordinates of x and y) is normalized, so that middle point becomes (0,0). It will be trated as principal point of the lens.
4. Each normalized index is transformed to polar coordinates and passed (in a vectorized manner for speeding up the process) to distortion (or redistortion if you will) function, changing to a new pair of index which _represents the position of pixel on original image with index equal to the value of remapping matrix cell that is to be moved to the this position on remap matrix_. This may be tricky at first glance, so take some time to understand how `cv2.remap()` works, hope [this](https://stackoverflow.com/questions/46520123/how-do-i-use-opencvs-remap-function) helps.
5. The obtained matrix (something like a blueprint for the transformed image construction) is then renormalized with respect to size of original image and the padding, if used, and passed to `cv2.remap()` gaining the final transformed image.

Remaping matrix calculation takes some time, but regarding that it is the same for all the pictures taken with the same lens (it is independant from the content of image) it needs to be calculated only once.

### Used models

The are plenty of models here and there, but here are the ones I tried for now:

- Logarithmic model [[1](https://www.researchgate.net/publication/47510646_Lens_Distortion_Models_Evaluation)]
- Field of view model [[1](https://www.researchgate.net/publication/47510646_Lens_Distortion_Models_Evaluation)][[2](https://hal-enpc.archives-ouvertes.fr/hal-01556898/document)]
- Fitgibbon model (simplified radial) [[3](http://marcodiiga.github.io/radial-lens-undistortion-filtering)]
- Radial model [[1](https://www.researchgate.net/publication/47510646_Lens_Distortion_Models_Evaluation)][[2](https://hal-enpc.archives-ouvertes.fr/hal-01556898/document)]
- Division model [[1](https://www.researchgate.net/publication/47510646_Lens_Distortion_Models_Evaluation)][[2](https://hal-enpc.archives-ouvertes.fr/hal-01556898/document)]
- Sterad model [[4](http://www.kscottz.com/fish-eye-lens-dewarping-and-panorama-stiching/)]

Every model has some distortion paramethers in different forms. For now I don't have a method to find the "correct" values for them computationally, so all of them where just approximated ~~by hand~~ (lets say it was grid search :) ). Feel free to play with them by yourself, maybe you'll get better results

In fututre maybe I will implement some kind of distortion paramether "calibration", but that is not for shure. Every paramether depends on each individual lens and picture taken with it, so is to some extent unique in each case.

### More on models

Here I will dive into practical details about models, so that it would be easier for you to use them. All of them use polar coordinate system because in this case it is much more handfull and demands less code.

Here is the original image took from [here](http://paulbourke.net/dome/fish2/):
![original image](https://raw.githubusercontent.com/chelicerae/dewarp/master/imgs/original.jpg)

#### Logarithmic model 

![original image](https://raw.githubusercontent.com/chelicerae/dewarp/master/imgs/log.jpg)

The math behind the model can be found in corresponding paper [[1](https://www.researchgate.net/publication/47510646_Lens_Distortion_Models_Evaluation)]. It had two paramethers: _s_, or scaling factor, and _lambda_ that controlls the amount of distortion. The distortion changes the next way due to the change of this paramethers: the bigger the _s_ is the less there are pincussion distortion and the smaller the _lambda_ is the closer is the resulting image. So, I guess I somehow confused this to paramethers because _s_ behaves exactly like the scalling paramether and visa versa. The results prooved to be not the ones I was expecting, maybe I made some mistake in implementing the model, so if you find one, I would be grateful if you report me. 

#### Field of view (FOV) model 

![original image](https://raw.githubusercontent.com/chelicerae/dewarp/master/imgs/fov_complete.jpg)

![original image](https://raw.githubusercontent.com/chelicerae/dewarp/master/imgs/fov.jpg)

This one can bo found in [[1](https://www.researchgate.net/publication/47510646_Lens_Distortion_Models_Evaluation)[2](https://hal-enpc.archives-ouvertes.fr/hal-01556898/document)]. I use the reverce function to obtain pincussion distortion in place of barrel one. The only paramether it takes is and angle in radians. The model showed the best results so far and it is said [[2](https://hal-enpc.archives-ouvertes.fr/hal-01556898/document)] that it can be combined with radial model (didn't try it out yet).

#### Fitzgibbon model 

![original image](https://raw.githubusercontent.com/chelicerae/dewarp/master/imgs/fitz.jpg)

This is the simplest one, so it was the first to be implemented. Results are better than nothing but still not as good as FOV model. Further information about the model is found here [[3](http://marcodiiga.github.io/radial-lens-undistortion-filtering)].

#### Steradian model

![original image](https://raw.githubusercontent.com/chelicerae/dewarp/master/imgs/sterad.jpg)

Steradian model is actually a quite creative idea that came to mind of [this person](http://www.kscottz.com/fish-eye-lens-dewarping-and-panorama-stiching/). I didn't find this approach in any of papers, so you can get the code form [the original repository](https://github.com/kscottz/dewarp/blob/master/fisheye/defish.py). Results are quite unusual, but may come in handy in certain situation. 

### Reference

- [1] - [Lens Distortion Models Evaluation](https://www.researchgate.net/publication/47510646_Lens_Distortion_Models_Evaluation)
- [2] - [A Precision Analysis of Camera Distortion Models](https://hal-enpc.archives-ouvertes.fr/hal-01556898/document)
- [3] - [Radial lens undistortion filtering](http://marcodiiga.github.io/radial-lens-undistortion-filtering)
- [4] - [Fish Eye Lens Dewarping and Panorama Stiching](http://www.kscottz.com/fish-eye-lens-dewarping-and-panorama-stiching/)













