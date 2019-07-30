import cv2
import time
import numpy as np
from matplotlib import pyplot
from math import sqrt, atan, exp, sin, cos, log, tan, pi


def cartesian_to_polar(y, x):
	return np.arctan2(y, x), sqrt(x**2 + y**2)

def polar_to_cartesian(alpha, r):
	return r*sin(alpha), r*cos(alpha)

def un_sterad():
	pass


def un_log(yx, params):
	"""
	Logarithm model distortion function.
	yx : np.array
		(y, x) pixel coordinate.
	params : {'lambda': float, 's': float}
		Distortion paramethers dict.
	"""

	ynd, xnd = yx[0], yx[1]
	alpha, rd = cartesian_to_polar(ynd, xnd)
	ru = params['s'] * log(1 + params['lambda']*rd)
	return polar_to_cartesian(alpha, ru)


def un_fov(yx, fov):
	"""
	A field of view reverce model function. 
	yx : np.array
		(y, x) pixel coordinate.
	fov : float
		Field of view angle in radians.
	"""

	ynd, xnd = yx[0], yx[1]
	alpha, rd = cartesian_to_polar(ynd, xnd)
	ru = atan(rd*tan(fov))/tan(fov)
	return polar_to_cartesian(alpha, ru)


def un_fitzgibbon(yx, params):
	"""
	Fitzgibbon distortion model function.
	yx : np.array
		(y, x) pixel coordinate.
	params : {'h': int, 'w': int, 'k': float}
		Distortion parameters. 'h' and 'w': width and height of the image respectivly.
		'k': distortion parameter.
	"""

	# Normaize to (-1, 1) range.
	ynd, xnd = yx[0]/(params['h']/2), yx[1]/(params['w']/2)

	alpha, rd = cartesian_to_polar(ynd, xnd)
	ru = rd/(1. - params['k']*(rd**2))
	ynu, xnu = polar_to_cartesian(alpha, ru)

	# Denormalize from (-1, 1) range.
	return (ynu*(params['h']/2), xnu*(params['w']/2))


# params format {'degree' : degree of polynome, 'ks' : vector of dist coefs equal to degree in length}
def un_radial(yx, params):
	"""
	Radial distortion model function.
	yx : np.array
		(y, x) pixel coordinate.
	params : {'h': int, 'w': int, 'k': float}
		Distortion parameters. 'h' and 'w': width and height of the image respectivly.
		'k': distortion parameter.
	"""

	ynd, xnd = yx[0], yx[1]
	alpha, rd = cartesian_to_polar(ynd, xnd)

	# Count r pawer series and dot product of k's and r power series.
	dgr = params['degree']
	powers = np.array(range(dgr))
	rd_vec = np.power(np.array([rd]*dgr), powers)
	ru = rd * np.dot(params['ks'].T, rd_vec)

	return polar_to_cartesian(alpha, ru)


# params format {'degree' : degree of polynome, 'ks' : vector of dist coefs equal to degree in length}
def un_division(yx, params):
	"""
	Division distortion model function.
	yx : np.array
		(y, x) pixel coordinate.
	params : {'h': int, 'w': int, 'k': float}
		Distortion parameters. 'h' and 'w': width and height of the image respectivly.
		'k': distortion parameter.
	"""

	ynd, xnd = yx[0], yx[1]
	alpha, rd = cartesian_to_polar(ynd, xnd)

	# Count r pawer series and dot product of k's and r power series.
	dgr = params['degree']
	powers = np.array(range(dgr))
	rd_vec = np.power(np.array([rd]*dgr), powers)
	ru = rd / np.dot(params['ks'].T, rd_vec)

	return polar_to_cartesian(alpha, ru)


def dist_remap(h, w, dist_func, dist_params, py=0, px=0):
	"""
	Returns two remap function for x and y respectively using dist_func to indexes.
	h, w : int, int
		Heinghts and width of the image.
	py, px : int, int
		h and w padding for displaying bigger part og an image in case it is strached beyond original size.
	dist_func : function
		Distortion model function that creates the mapping from one image to another.
	dist_params : any
		Model specific distortion parameters. 
	"""

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
	return dist_mapxy[:, :, 0], dist_mapxy[:, :, 1]


def dewarp_video(video_dir, save_dir, dist_func, dist_params, py=0, px=0):
	"""
	Calculates remapping matrix and displays the video dewarping it in real-time.
	video_dir : string
		Directory of video
	save_dir : string
		Save directory of dewarped video.
	dist_func : function
		Distortion model function that creates the mapping from one image to another.
	dist_params : any
		Model specific distortion parameters. 
	py, px : int, int
		Images h and w padding for displaying bigger part of an image in case it is strached beyond original size.
	"""

	cap = cv2.VideoCapture(video_dir)
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	# out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
	
	cv2.namedWindow('video', cv2.WINDOW_NORMAL)
	cv2.resizeWindow('video', 900, 600)

	h = int(cap.get(4)) 
	w = int(cap.get(3))  

	start = time.time()
	mapy, mapx = dist_remap(h, w, dist_func, dist_params, py, px)
	end = time.time()
	print('Remap time:', end - start)

	while(cap.isOpened()):
		ret, frame = cap.read()
		if ret==True:
			frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
			cv2.imshow('video', frame)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break


def dist_img(src_dir, dist_func, dist_params, py=0, px=0):
	"""
	Calculates remapping matrix and uses it to the given image.
	src_dir : string
		Directory of image
	save_dir : string
		Save directory of dewarped image.
	dist_func : function
		Distortion model function that creates the mapping from one image to another.
	dist_params : any
		Model specific distortion parameters. 
	py, px : int, int
		Images h and w padding for displaying bigger part of an image in case it is strached beyond original size.
	"""

	# Resize image window for consistancy.
	cv2.namedWindow('fisheye', cv2.WINDOW_NORMAL)
	cv2.resizeWindow('fisheye', 900, 600)

	img = cv2.imread(src_dir)
	h, w = img.shape[0], img.shape[1]
	mapy, mapx = dist_remap(h, w, dist_func, dist_params, py, px)

	start = time.time()
	res_img = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
	end = time.time()
	print('Remap time:', end - start)
	
	cv2.imshow('fisheye', res_img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	cv2.imwrite('imgs/fitz.jpg', res_img)

def plotTC(fnc, dist_params, mindim, maxdim, inc):
	"""
	Build plot of time spent on remaping function building dependant on square image size.
	fnc : function
		Dewarping function used for pixel remaping.
	dist_params : any
		Distortion paramethers passed to fnc. Data type is distortion function specific.
	mindim : int
		Starting image size (mindim, mindim).
	maxdim : int
		Final image size (maxdim, maxdim).	
	inc : int
		Step of size progression from mindim to maxdim.
	"""
	x = []
	y = []
	for i in range(mindim, maxdim, inc):
		start = time.time()
		dist_remap(i, i, fnc, dist_params)
		end = time.time()
		x.append(i)
		y.append(end - start)
	pyplot.plot(x, y, 'ro-')
	pyplot.show()


# Testing models h/16 w/16
# log_params = {'lambda': 0.09, 's': 150}
# dist_img('imgs/original.jpg', un_log, log_params)

# Paddings: py, px = int(h+1500), int(w+1500)
# fov = pi*0.00085
# dist_img('imgs/original.jpg', un_fov, fov)

# Paddings: py, px = int(h+1500), int(w+1500)
# fov = pi*0.00085
# dist_img('imgs/original.jpg', un_fov, fov)

# params = {'h': 1144, 'w': 1024, 'k': -0.27}
# dist_img('imgs/original.jpg', un_fitzgibbon, params)

plotTC(un_fov, pi*0.00085, 100, 500, 100)


