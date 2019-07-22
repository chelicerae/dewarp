import cv2
import time
import numpy as np
from math import sqrt, atan, exp, sin, cos, log, tan, pi


def un_log(yx, params):
	ynd, xnd = yx[0], yx[1]
	alpha = np.arctan2(ynd, xnd)
	rd = sqrt(xnd**2 + ynd**2)
	ru = params['s'] * log(1 + params['lambda']*rd)
	ynu = ru*sin(alpha)
	xnu = ru*cos(alpha)
	return (ynu, xnu)


# A field of view reverce model function. 
def un_fov(yx, fov):
	ynd, xnd = yx[0], yx[1]
	alpha = np.arctan2(ynd, xnd)
	rnd = sqrt(ynd**2 + xnd**2)
	rnu = atan(rnd*tan(fov))/tan(fov)
	ynu = rnu*sin(alpha) 
	xnu = rnu*cos(alpha) 
	return (ynu, xnu)


def un_fitzgibbon(yx, params):
	ynd, xnd = yx[0]/(params['h']/2), yx[1]/(params['w']/2)
	alpha = np.arctan2(ynd, xnd)
	rnd = sqrt((xnd)**2 + (ynd)**2)
	rnu = rnd/(1. - params['k']*(rnd**2))
	ynu = rnu*sin(alpha) 
	xnu = rnu*cos(alpha) 
	return (ynu*(params['h']/2), xnu*(params['w']/2))


def dist_remap(h, w, py, px, dist_func, dist_params):

	# Create indeces remap matrix. 
	mapxy = np.indices((h+2*py, w+2*px))

	# Calculate img center for denormalized and normalized imgs.
	cy, cx = h/2, w/2
	cny, cnx = (h+2*py)/2., (w+2*px)/2.

	# Stack two matrixes into tensor.
	mapnxy = np.stack(((mapxy[0] - cny).astype(int), (mapxy[1] - cnx).astype(int)), axis=2)
	dist_mapnxy = np.apply_along_axis(dist_func, 2, mapnxy, dist_params)

	# Denormalize the obtained matrix with respecr to original image ratio.
	dist_mapxy = (dist_mapnxy + [cy, cx]).astype(np.float32)
	print(dist_mapxy[:, :, 0].shape, dist_mapxy[:, :, 1].shape)
	return dist_mapxy[:, :, 0], dist_mapxy[:, :, 1]


def dewarp_video(video_dir, save_dir, dist_func, dist_params):
	cap = cv2.VideoCapture(video_dir)
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	# out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
	
	cv2.namedWindow('video', cv2.WINDOW_NORMAL)
	cv2.resizeWindow('video', 900, 600)

	h = int(cap.get(4)) 
	w = int(cap.get(3))  

	start = time.time()
	mapy, mapx = dist_remap(h, w, 0, 0, dist_func, dist_params)
	end = time.time()
	print('Remap time:', end - start)

	while(cap.isOpened()):
		ret, frame = cap.read()
		if ret==True:
			frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
			cv2.imshow('video', frame)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break


def dist_img(src_dir, dist_func, dist_params):
	# Resize image window for consistancy.
	cv2.namedWindow('fisheye', cv2.WINDOW_NORMAL)
	cv2.resizeWindow('fisheye', 900, 600)

	img = cv2.imread(src_dir)
	# img = cv2.copyMakeBorder(img, 128, 128, 0, 0, cv2.BORDER_CONSTANT)
	h, w = img.shape[0], img.shape[1]
	mapy, mapx = dist_remap(h, w, 0, 0, dist_func, dist_params)

	start = time.time()
	res_img = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
	end = time.time()
	print('Remap time:', end - start)
	
	cv2.imshow('fisheye', cv2.flip(res_img, 0))
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	cv2.imwrite('vectorized.jpg', cv2.flip(res_img, 0))



k = -0.45
# dist_img('../img/image-resize.jpg', un_fitzgibbon, k)
dewarp_video('../20190409AM/Camera 1 - Pano-20190409-114743-1554803263.mp4', 'save_dir', un_fov, pi*0.0005)
# log_params = {'lambda': 0.09, 's': 150}
# dist_img('../img/image-resize.jpg', unlog, log_params)
fov = pi*0.00085
# dist_img('../img/image-resize.jpg', un_fov, fov)


