from PIL import Image, ImageOps
import glob, os

def convert_to_grayscale(filename):
	"""
	"""
	try:
		img = Image.open('test.jpg').convert('L')
		img.save('gray.jpg')
	except Exception as e:
		logger.error(e, exc_info=True)

def downsize():
	"""
	"""
	try:
		im = Image.open('test.jpg')
		im.thumbnail((128, 128))
		im.save("test_thumbnail.jpg")
	except Exception as e:
		logger.error(e, exc_info=True)

def downsample():
	"""
	"""
	try:
		im = ImageOps.posterize(Image.open('test.jpg').convert('L'), 5)
		im.save("test_depth_thumbnail.jpg")
	except Exception as e:
		logger.error(e, exc_info=True)

#convert_to_grayscale()
#downsize()
downsample()